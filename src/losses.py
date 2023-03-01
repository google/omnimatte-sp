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

import jax
import jax.numpy as jnp


def transform(x):
  """Map from [0, 1] -> [-1, 1]"""
  return x * 2 - 1


def untransform(x):
  """Map from [-1, 1] -> [0, 1]"""
  return x * .5 + .5


def compute_recon_loss(
    mask, valid_frames, orig_frame, bg, rgba_pred, valid_pixels=None
):
  """Computes reconstruction loss."""
  rgba_pred = untransform(rgba_pred)
  bg = untransform(bg)
  obj_masks = untransform(mask)
  valid_frames = untransform(valid_frames)
  temporal_loss_mask = jnp.clip(1.0 - valid_frames, a_min=0, a_max=1)

  # composite predicted layers
  recon = bg[:, :3]
  alpha_pred_recentered = rgba_pred[:, 3:4]  # B,1,L,T,H,W
  n_objs = mask.shape[1]
  for i in range(n_objs):
    recon = (
        recon * (1 - alpha_pred_recentered[:, :, i])
        + alpha_pred_recentered[:, :, i] * rgba_pred[:, :3, i]
    )
  rgba_pred = transform(rgba_pred)
  recon = transform(recon)
  obj_regions = (jnp.sum(obj_masks, axis=1, keepdims=True) > 1).astype(
      jnp.float32
  )
  spatial_weight_recon = 1.0 - 0.9 * obj_regions
  if valid_pixels:
    valid_pixels = untransform(valid_pixels)
    spatial_weight_recon *= valid_pixels
  recon_weights = temporal_loss_mask * spatial_weight_recon
  err_recon = jnp.sum(recon_weights * jnp.abs(recon - orig_frame)) / jnp.sum(
      recon_weights
  )
  return err_recon, recon


def compute_alpha_reg(mask, valid_frames, rgba_pred):
  n_objs = mask.shape[1]
  alpha = untransform(rgba_pred[:, 3:])
  alpha_composite = jnp.zeros_like(mask[:, :1])
  for i in range(n_objs):
    alpha_composite = (
        alpha_composite * (1 - alpha[:, :, i]) + alpha[:, :, i] * alpha[:, :, i]
    )
  valid_frames = untransform(valid_frames)
  temporal_loss_mask = jnp.clip(1. - valid_frames, a_min=0, a_max=1)
  # use transmittance to exclude occluded regions
  mask_recentered = untransform(mask)  # B,L,T,H,W
  occlusion = jnp.cumsum(jnp.flip(mask_recentered, axis=-4), axis=-4)
  occlusion = jnp.flip(occlusion, axis=-4)
  occlusion -= mask_recentered  # subtract self
  alpha_sparsity_weights = (
      1 - occlusion[:, jnp.newaxis]) * temporal_loss_mask[:, :, jnp.newaxis]
  pseudo_l0 = 2 * jax.nn.sigmoid(alpha_composite * 5.0) - 1.
  alpha_reg_l0 = jnp.mean(alpha_sparsity_weights * pseudo_l0)
  alpha_reg_l1 = jnp.mean(alpha_sparsity_weights * alpha_composite)
  return alpha_reg_l0, alpha_reg_l1


def compute_mask_loss(mask, valid_frames, rgba_pred):
  valid_frames = untransform(valid_frames)
  temporal_loss_mask = jnp.clip(1. - valid_frames, a_min=0, a_max=1)

  # balance positive and negative regions
  pos_mask = untransform(mask)
  neg_mask = 1 - pos_mask
  h, w = pos_mask.shape[-2:]
  npos = pos_mask.sum(-1, keepdims=True).sum(-2, keepdims=True) + 1
  nneg = neg_mask.sum(-1, keepdims=True).sum(-2, keepdims=True) + 1
  spatial_weight = h * w * .5 * (pos_mask / npos + neg_mask / nneg)
  pred_alpha = rgba_pred[:, 3]
  err_mask = jnp.mean(
      temporal_loss_mask * spatial_weight *
      jnp.abs(pred_alpha - mask))
  return err_mask
