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

from functools import lru_cache, partial

from . import losses
import haiku as hk
import jax
from .networks import OmnimatteSP
import optax


@lru_cache()
def forward(config):

  def _forward(batch, is_training):
    """Builds model."""
    model = OmnimatteSP(
        im_height=config.input_height,
        im_width=config.input_width)
    return model(batch, is_training)

  return hk.transform_with_state(_forward)


@lru_cache()
def optimizer(learning_rate):
  return optax.adamw(learning_rate)


@partial(jax.pmap, static_broadcasted_argnums=(2), axis_name='i')
def make_initial_state(rng, batch, config):
  """Computes the initial network and optimizer states."""
  params, state = forward(config).init(rng, batch, is_training=True)
  opt_state = optimizer(config.learning_rate).init(params)
  return params, state, opt_state


def loss_fn(params, state, batch, loss_weights, key, config):
  """Computes unsupervised losses."""

  # mask out input target frames before passing to model
  orig_image = batch['frame']
  valid_mask = batch['valid_frames'] * .5 + .5
  masked_image = orig_image * valid_mask
  masked_image = masked_image - (1 - valid_mask
                                )  # masked out pixels have val -1
  batch['frame'] = masked_image

  model_output, state = forward(config).apply(
      params, state, key, batch, is_training=True)

  # replace masked image with original image
  batch['frame'] = orig_image

  # reconstruction loss
  err_recon, recon = losses.compute_recon_loss(
      batch['mask'], batch['valid_frames'], batch['frame'], batch['bg'],
      model_output['rgba'], batch.get('valid_pixels')
  )

  # alpha regularizer
  alpha_reg_l0, alpha_reg_l1 = losses.compute_alpha_reg(
      batch['mask'], batch['valid_frames'], model_output['rgba']
  )
  reg_alpha = (
      loss_weights['lambda_alpha_l0'] * alpha_reg_l0
      + loss_weights['lambda_alpha_l1'] * alpha_reg_l1
  )

  # mask bootstrapping
  err_mask = loss_weights['lambda_mask'] * losses.compute_mask_loss(
      batch['mask'], batch['valid_frames'], model_output['rgba']
  )

  loss = err_recon + reg_alpha + err_mask

  scalars = {
      'loss': loss,
      'err_recon': err_recon,
      'reg_alpha': reg_alpha,
      'err_mask': err_mask,
  }
  model_output['recon'] = recon

  return loss, (scalars, state, model_output)


@partial(jax.pmap, static_broadcasted_argnums=(6), axis_name='i')
def train_step(params, state, opt_state, key, batch, loss_weights, config):
  """Performs a single training update."""
  lr = config.learning_rate
  grads, (scalars, state, outputs) = jax.grad(loss_fn, has_aux=True)(
      params, state, batch, loss_weights, key, config
  )

  grads = jax.lax.pmean(grads, axis_name='i')
  scalars = jax.lax.pmean(scalars, axis_name='i')

  updates, opt_state = optimizer(lr).update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)

  return params, state, opt_state, scalars, outputs, batch


@partial(jax.pmap, static_broadcasted_argnums=(5), axis_name='i')
def eval_step(params, state, key, batch, loss_weights, config):
  """Performs a single training update."""
  _, (scalars, _, model_output) = loss_fn(
      params, state, batch, loss_weights, key, config
  )
  return scalars, model_output
