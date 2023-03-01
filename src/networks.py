# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


class DenseBlock(hk.Module):
  """A 2-layer MLP which widens then narrows the input."""

  def __init__(self,
               init_scale: float,
               widening_factor: int = 4,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._init_scale = init_scale
    self._widening_factor = widening_factor

  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    hiddens = x.shape[-1]
    initializer = hk.initializers.VarianceScaling(self._init_scale)
    x = hk.Linear(self._widening_factor * hiddens, w_init=initializer)(x)
    x = jax.nn.gelu(x)
    return hk.Linear(hiddens, w_init=initializer)(x)


class Transformer(hk.Module):
  """A transformer stack."""

  def __init__(self,
               num_heads: int,
               num_layers: int,
               dropout_rate: float,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._num_layers = num_layers
    self._num_heads = num_heads
    self._dropout_rate = dropout_rate

  def __call__(self, query: jnp.ndarray, key: jnp.ndarray, value: jnp.ndarray,
               mask: Optional[jnp.ndarray], is_training: bool) -> jnp.ndarray:
    """Connects the transformer.

    Args:
      query: Inputs, [B, T, H].
      key: Inputs, [B, T, H].
      value: Inputs, [B, T, H].
      mask: Padding mask, [B, T].
      is_training: Whether we're training or not.

    Returns:
      Array of shape [B, T, H].
    """

    init_scale = 2. / self._num_layers
    dropout_rate = self._dropout_rate if is_training else 0.
    if mask is not None:
      mask = mask[:, None, None, :]

    h = query
    # Note: names chosen to approximately match those used in the GPT-2 code;
    # see https://github.com/openai/gpt-2/blob/master/src/model.py.
    for i in range(self._num_layers):
      h_norm = layer_norm(h, name=f'h{i}_ln_0')
      h_attn = hk.MultiHeadAttention(
          num_heads=self._num_heads,
          key_size=64,
          w_init_scale=init_scale,
          name=f'h{i}_selfattn')(h_norm, h_norm, h_norm)  #, mask=mask)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = layer_norm(h, name=f'h{i}_ln_1')
      h_attn = hk.MultiHeadAttention(
          num_heads=self._num_heads,
          key_size=64,
          w_init_scale=init_scale,
          name=f'h{i}_attn')(h_norm, key, value)  #, mask=mask)
      h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
      h = h + h_attn
      h_norm = layer_norm(h, name=f'h{i}_ln_2')
      h_dense = DenseBlock(init_scale, name=f'h{i}_mlp')(h_norm)
      h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
      h = h + h_dense
    h = layer_norm(h, name='ln_f')

    return h


def layer_norm(x: jnp.ndarray, name: Optional[str] = None) -> jnp.ndarray:
  """Apply a unique LayerNorm to x with default settings."""
  return hk.LayerNorm(
      axis=-1, create_scale=True, create_offset=True, name=name)(
          x)


class ConvBlock(hk.Module):
  """Module containing convolution/transposed convolution, optional batchnorm, and activation."""

  def __init__(self,
               conv_fn,
               out_channels,
               activation,
               do_norm=True,
               stride=2,
               name=None,
               **kwargs):
    super(ConvBlock, self).__init__(name=name)
    self._conv = conv_fn(
        output_channels=out_channels,
        kernel_shape=4,
        stride=stride,
        data_format='NCHW',
        w_init=hk.initializers.VarianceScaling(2.0, 'fan_in', 'uniform'),
        **kwargs)
    self._norm = hk.InstanceNorm(
        True, True, data_format='NCHW') if do_norm else None
    self._activation = activation

  def __call__(self, x, is_training):
    x = self._conv(x)
    if self._norm is not None:
      x = self._norm(x)  #, is_training)
    if self._activation is not None:
      x = self._activation(x)
    return x


class Encoder(hk.Module):

  def __init__(self, num_filters=64, name=None):
    super(Encoder, self).__init__(name=name)

    leaky_relu = lambda x: jax.nn.leaky_relu(x, 0.2)

    self._encoder = [
        ConvBlock(hk.Conv2D, num_filters, leaky_relu, do_norm=False),
        ConvBlock(hk.Conv2D, num_filters * 2, leaky_relu),
        ConvBlock(hk.Conv2D, num_filters * 4, leaky_relu),
        ConvBlock(hk.Conv2D, num_filters * 4, leaky_relu),
    ]

  def __call__(self, x, is_training):
    x = self._encoder[0](x, is_training)
    skip = x
    for layer in self._encoder[1:]:
      x = layer(x, is_training)
    return x, skip


class Decoder(hk.Module):

  def __init__(self, num_filters=64, name=None):
    super(Decoder, self).__init__(name=name)
    self._decoder = [
        ConvBlock(hk.Conv2DTranspose, num_filters * 4, jax.nn.relu),
        ConvBlock(hk.Conv2DTranspose, num_filters * 2, jax.nn.relu),
        ConvBlock(hk.Conv2DTranspose, num_filters, jax.nn.relu),
        ConvBlock(hk.Conv2DTranspose, num_filters, jax.nn.relu)
    ]

    self._final_layer = hk.Conv2D(
        output_channels=4,
        kernel_shape=(3, 3),
        stride=(1, 1),
        data_format='NCHW',
        name='final_rgba')

  def __call__(self, x, is_training, skip):
    for layer in self._decoder[:-1]:
      x = layer(x, is_training)
    x = self._decoder[-1](jnp.concatenate((x, skip), 1), is_training)
    pre_tan = self._final_layer(x)
    pre_tan = jnp.clip(pre_tan, a_min=-3, a_max=3)
    rgba = jnp.tanh(pre_tan)
    return rgba, pre_tan


class OmnimatteSP(hk.Module):
  """Transformer architecture that predicts an RGBA layer for each input mask."""

  def __init__(self,
               num_filters=64,
               name=None,
               im_height=128,
               im_width=128):
    super(OmnimatteSP, self).__init__(name=name)
    self.im_encoder = Encoder(name='im_encoder')
    self.mask_encoder = Encoder(name='mask_encoder')
    self.layer_decoder = Decoder(name='layer_decoder')

    num_heads = 4
    t_layers = 2
    dropout = .1
    self._transformer = Transformer(num_heads, t_layers, dropout)  # decoder

    self._max_frames = 10
    self._feat_width = im_width // 16
    self._feat_height = im_height // 16
    self._emb_len = (im_height // 16) * (im_width // 16)

  def _get_pos_embs(self, name):
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    x_pos = hk.get_parameter(
        name + '_x', [self._feat_width, 96], init=embed_init
    )
    y_pos = hk.get_parameter(
        name + '_y', [self._feat_height, 96], init=embed_init
    )
    t_pos = hk.get_parameter(
        name + '_t', [self._max_frames, 64], init=embed_init
    )
    pos_embs = jnp.concatenate(
        [
            jnp.tile(
                x_pos[jnp.newaxis, jnp.newaxis],
                [self._max_frames, self._feat_height, 1, 1],
            ),
            jnp.tile(
                y_pos[jnp.newaxis, :, jnp.newaxis],
                [self._max_frames, 1, self._feat_width, 1],
            ),
            jnp.tile(
                t_pos[:, jnp.newaxis, jnp.newaxis],
                [1, self._feat_height, self._feat_width, 1],
            ),
        ],
        -1,
    )  # T, H, W, 256
    return pos_embs

  def _transform(self, im_feat, mask_feat, data, is_training=True):
    B, D, T, H, W = im_feat.shape
    im_feat_reshape = jnp.einsum('bdn->bnd', im_feat.reshape([B, D, -1]))

    positional_embeddings = self._get_pos_embs('pos_embs')
    mask_positional_embeddings = self._get_pos_embs('mask_pos_embs')
    positional_embeddings = jnp.clip(positional_embeddings, a_min=-1, a_max=1)
    mask_positional_embeddings = jnp.clip(
        mask_positional_embeddings, a_min=-1, a_max=1)

    L = mask_feat.shape[1]
    mask_feat_reshape = jnp.einsum('bldthw->blthwd', mask_feat)

    positional_embeddings = positional_embeddings[:T, :H, :W]
    mask_positional_embeddings = mask_positional_embeddings[:T, :H, :W]
    mask_feat_with_emb = mask_feat_reshape + mask_positional_embeddings
    mask_feat_with_emb = mask_feat_with_emb.reshape([B * L, -1, D])
    im_feat_with_emb = (
        jnp.einsum('bdthw->bthwd', im_feat) + positional_embeddings
    )
    im_feat_with_emb = im_feat_with_emb.reshape([B, -1, D])

    # repeat for each object layer
    im_feat_with_emb = jnp.tile(im_feat_with_emb[:, jnp.newaxis], (1, L, 1, 1))
    im_feat_reshape = jnp.tile(im_feat_reshape[:, jnp.newaxis], (1, L, 1, 1))

    # clip features
    mask_feat_with_emb = jnp.clip(mask_feat_with_emb, a_min=-2, a_max=2)
    im_feat_with_emb = jnp.clip(im_feat_with_emb, a_min=-2, a_max=2)
    im_feat_reshape = jnp.clip(im_feat_reshape, a_min=-2, a_max=2)

    im_feat_with_emb = im_feat_with_emb.reshape(B * L, -1, D)
    im_feat_reshape = im_feat_reshape.reshape(B * L, -1, D)

    out = self._transformer(
        mask_feat_with_emb,
        im_feat_with_emb,
        im_feat_reshape,
        mask=None,
        is_training=is_training)
    out = jnp.einsum('bnd->bdn', out).reshape([B, L, -1, T, H, W])

    return out

  def __call__(self, data, is_training):
    """Passes input layers individually through net and composites outputs."""
    # 1. cnn encode frames and masks
    # 2. encoded frames (K,V), masks (Q) -> transformer decoder
    # 3. cnn decode to rgba

    image = data['frame']
    mask = data['mask']  # B, L-1, T, H, W

    B, _, T, H, W = image.shape

    image = jnp.einsum('bdthw->btdhw', image).reshape((B * T, 3, H, W))
    im_feat, im_feat_skip = self.im_encoder(image, is_training)
    H2, W2 = im_feat.shape[2:]
    L = mask.shape[1]
    inp_mask = mask.reshape(B * L * T, -1, H, W)  # BLT, D, H, W
    mask_feat, _ = self.mask_encoder(inp_mask, is_training)

    im_feat = jnp.einsum('btdhw->bdthw', im_feat.reshape((B, T, -1, H2, W2)))
    # add masked embedding to im_feat (indicate which frames are masked)
    valid = data['valid_frames'][..., 0:1, 0:1] * 0.5 + 0.5  # B, 1, T, 1, 1
    embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
    valid_emb = hk.get_parameter('valid_emb', [256], init=embed_init)
    invalid_emb = hk.get_parameter('invalid_emb', [256], init=embed_init)
    im_feat += valid * valid_emb.reshape(1, -1, 1, 1, 1)
    im_feat += (1 - valid) * invalid_emb.reshape(1, -1, 1, 1, 1)
    mask_feat = jnp.einsum(
        'bltdhw->bldthw', mask_feat.reshape(B, L, T, -1, H2, W2)
    )
    layer_feats = self._transform(
        im_feat, mask_feat, data, is_training=is_training
    )
    layer_feats = jnp.einsum('bldthw->bltdhw', layer_feats).reshape(
        B * L * T, -1, H2, W2
    )
    H3, W3 = im_feat_skip.shape[-2:]
    im_feat_skip = jnp.tile(
        im_feat_skip.reshape(B, 1, T, -1, H3, W3),
        (1, L, 1, 1, 1, 1),
    )
    im_feat_skip = im_feat_skip.reshape(B * L * T, -1, H3, W3)
    rgba, _ = self.layer_decoder(layer_feats, is_training, im_feat_skip)
    rgba = jnp.einsum('bltdhw->bdlthw', rgba.reshape(B, L, T, -1, H, W))
    outputs = {
        'rgba': rgba,  # B, 4, L, T, H, W
        'inp_mask': mask,
    }
    return outputs

