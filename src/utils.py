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

import glob
import os
import pickle

import jax
import jax.numpy as jnp
from matplotlib import cm
import mediapy as media
import numpy as np
from PIL import Image
import tensorflow as tf


def read_image(filepath, width=None, height=None, pil_format='RGB'):
  im = Image.open(filepath).convert(pil_format)
  if width and height:
    im = im.resize((width, height), resample=Image.BILINEAR)
  return np.array(im)


def write_image(path, im, pil_format='RGB'):
  im = Image.fromarray(im, pil_format)
  im.save(path)


def process_output(x):
  """Converts network output to image array."""
  x = np.asarray(x) * .5 + .5
  x = 255 * np.transpose(x, [1, 2, 0])
  return x.astype(np.uint8)


def save_chkpt(chkpt_dir, params, opt_state, step):
  if not os.path.exists(chkpt_dir): os.makedirs(chkpt_dir)
  params_single = jax.tree_map(lambda x: x[0], params)
  opt_state_single = jax.tree_map(lambda x: x[0], opt_state)
  state = {
      'params': params_single,
      'opt_state': opt_state_single,
      'step': step}

  with open(os.path.join(chkpt_dir, 'arrays.npy'), 'wb') as f:
    for x in jax.tree_leaves(state):
      np.save(f, x, allow_pickle=False)

  tree_struct = jax.tree_map(lambda t: 0, state)
  with open(os.path.join(chkpt_dir, 'tree.pkl'), 'wb') as f:
    pickle.dump(tree_struct, f)


def restore_chkpt(chkpt_dir):
  with open(os.path.join(chkpt_dir, 'tree.pkl'), 'rb') as f:
    tree_struct = pickle.load(f)

  leaves, treedef = jax.tree_util.tree_flatten(tree_struct)
  with open(os.path.join(chkpt_dir, 'arrays.npy'), 'rb') as f:
    flat_state = [np.load(f) for _ in leaves]

  return jax.tree_util.tree_unflatten(treedef, flat_state)


def interpolate_pos_embs(params, target_params):
  """Reshapes positional embeddings if necessary."""
  # target_params has an extra device dimension
  param_names = ['pos_embs_x', 'pos_embs_y', 'pos_embs_t']
  for name in param_names:
    param = params['omnimatte_sp'][name]
    target_shape = target_params['omnimatte_sp'][name].shape
    if params['omnimatte_sp'][name].shape[0] != target_shape[1]:
      # interpolate
      params['omnimatte_sp'][name] = jax.image.resize(
          param[jnp.newaxis], (1,) + target_shape[1:], 'linear'
      )[0]
      params['omnimatte_sp']['mask_' + name] = jax.image.resize(
          param[jnp.newaxis], (1,) + target_shape[1:], 'linear'
      )[0]
  return params


def write_video(output_path, frames, fps=25):
  with media.VideoWriter(output_path, shape=frames[0].shape[:2], fps=fps) as w:
    for image in frames:
      w.add_image(image)


def make_grid_video(
    config,
    vid,
    results_dir,
    data_dir,
    image_width,
    image_height,
    n_obj,
    fps=12,
):
  if config.dataset_structure == 'kubric':
    vidpath = f'{results_dir}/vids/{vid}.mp4'
  else:
    vidpath = f'{results_dir}/vids/vid.mp4'
  if os.path.exists(vidpath):
    os.remove(vidpath)
  if config.dataset_structure == 'kubric':
    layer_paths = sorted(glob.glob(f'{results_dir}/rgba_pred/{vid}/mask0*'))
  else:
    layer_paths = sorted(glob.glob(f'{results_dir}/rgba_pred/01/*'))
    mask_names = os.listdir(f'{results_dir}/mask_input')
  fnames = [os.path.basename(path) for path in layer_paths]
  cmap = cm.get_cmap('gist_earth')
  # Write a video
  frames = []
  for fname in fnames:
    fbasename = f'{fname.split("_")[-1]}'
    cols = []
    if config.dataset_structure == 'kubric':
      im_path = f'{data_dir}/{vid}/rgba_{fname.split("_")[-1]}'
      im_path = f'{os.path.splitext(im_path)[0]}.jpg'
    else:
      im_path = f'{data_dir}/rgb/{fname.split("_")[-1]}'
    orig = read_image(im_path, image_width, image_height)
    recon = read_image(
        f'{results_dir}/recon/{vid}/{fbasename}', image_width, image_height
    )
    # pad with zeros
    cols.append(
        np.concatenate([orig, recon] + [np.zeros_like(orig)] * (n_obj - 2), 0)
    )
    # mask in
    masks = []
    for o in range(n_obj):
      if config.dataset_structure == 'kubric':
        mask_path = f'{results_dir}/mask_input/{vid}/mask{o}_{fbasename}'
      else:
        mask_path = f'{results_dir}/mask_input/{mask_names[o]}/{fname}'
      mask = read_image(mask_path, image_width, image_height)
      masks.append(mask)
    cols.append(np.concatenate(masks, 0))
    rgba_preds = []
    for o in range(n_obj):
      if config.dataset_structure == 'kubric':
        rgba_path = f'{results_dir}/rgba_pred/{vid}/mask{o}_{fbasename}'
      else:
        rgba_path = f'{results_dir}/rgba_pred/{mask_names[o]}/{fname}'
      rgba_pred = read_image(
          rgba_path, image_width, image_height, pil_format='RGBA'
      )
      rgba_preds.append(rgba_pred)
    rgba_pred = np.concatenate(rgba_preds, 0)
    alpha = rgba_pred[..., -1:].astype(np.float32) / 255.0
    alpha_vis = (255 * cmap(alpha[..., 0])).astype(np.uint8)[..., :3]
    cols.append(alpha_vis)
    # render rgba over white background
    rgb_pred = rgba_pred[..., :3].astype(np.float32)
    rgba_vis = alpha * rgb_pred + (1 - alpha) * 255 * np.ones_like(rgb_pred)
    cols.append(rgba_vis.astype(np.uint8))
    bg = read_image(
        f'{results_dir}/bg/{vid}/{fbasename}', image_width, image_height
    )
    recon_err = np.linalg.norm(
        recon.astype(np.float32) - orig.astype(np.float32), axis=-1
    )
    recon_err = np.clip(
        recon_err / 255.0, 0, 1
    )  # scale for visualization purposes
    recon_err = (cmap(recon_err) * 255).astype(np.uint8)[..., :3]
    # pad with zeros
    cols.append(
        np.concatenate([bg, recon_err] + [np.zeros_like(bg)] * (n_obj - 2), 0)
    )
    grid = np.concatenate(cols, 1)
    frames.append(grid)
  write_video(vidpath, frames, fps)


def make_grids(config, results_dir, n_objs, fps=12):
  """Visualizes results as video grids."""
  os.makedirs(f'{results_dir}/vids', exist_ok=True)

  if config.dataset_structure == 'kubric':
    viddirs = glob.glob(os.path.join(results_dir, 'rgba_pred', '*'))
    viddirs = [os.path.basename(v) for v in viddirs]
  else:
    viddirs = ['']

  for vid in viddirs:
    make_grid_video(
        config,
        vid,
        results_dir,
        config.datadir,
        config.input_width,
        config.input_height,
        n_objs,
        fps=fps
    )


def save_image_sequence(images, paths, prefix='', mode='RGB'):
  """Saves a sequence of `images`: C,T,H,W"""
  for j in range(images.shape[1]):  # time dimension
    im = process_output(images[:, j])
    p = f'{prefix}{paths[j]}'
    if mode == 'L' and len(im.shape) > 2:
      im = im[..., 0]
    if not os.path.exists(p):
      write_image(p, im, pil_format=mode)


def save_batch(images, paths, prefix='', mode='RGB'):
  """Saves a batch of image sequences. images: B,C,T,H,W"""
  for p in paths:
    for pp in p:
      os.makedirs(prefix + os.path.dirname(pp), exist_ok=True)
  for i in range(images.shape[0]):
    save_image_sequence(images[i], paths[i], prefix, mode)


def transfer_detail(rgba, recon, orig):
  """Transfers the residual detail to each layer, as described in 
  https://arxiv.org/pdf/2009.07833.pdf
  """
  residual = orig - recon
  n_layers = rgba.shape[2]
  rgba_d = rgba.copy()
  transmission_comp = np.zeros_like(recon[:, :1])
  for l in range(n_layers - 1, -1, -1):
    alpha = rgba[:, -1:, l] * .5 + .5
    transmission = 1. - transmission_comp
    transmission_comp = alpha + (1. - alpha) * transmission_comp
    rgba_d[:, :3, l] += transmission * residual
  rgba_d = np.clip(rgba_d, -1, 1)
  return rgba_d


def save_results(dset, outputs, mask_paths, save_dir, config):
  """Saves results as image frames."""
  # collapse device and batch dims
  mask_paths = mask_paths.reshape(-1, *mask_paths.shape[2:])
  mask_paths = [sample.decode().split(',') for sample in mask_paths]
  mask_paths = [
      ['/'.join(path.split('/')[-2:]) for path in sample]
      for sample in mask_paths
  ]
  batch = dset.current_batch
  # collapse device and batch dims
  batch = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), batch)
  outputs = jax.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), outputs)

  b_sz, _, t_sz = batch['mask'].shape[:3]
  layer0_paths = [sample[:t_sz] for sample in mask_paths]
  rgb_paths = [
      [dset.get_file_basename(path) for path in sample]
      for sample in layer0_paths
  ]  # remove the mask*_ prefix from filename
  vis_i = config.vis_frame
  batch['mask'] = batch['mask'][..., vis_i : vis_i + 1, :, :]
  batch['mask'] = batch['mask'].reshape(
      (b_sz, 1, -1) + batch['mask'].shape[-2:]
  )  # B,L*T, H, W
  shp = outputs['rgba'].shape
  # sliding window scheme, only save middle frame
  outputs = jax.tree_map(lambda x: x[..., vis_i : vis_i + 1, :, :], outputs)
  mask_paths = [x[vis_i::t_sz] for x in mask_paths]
  print('saving batch ', mask_paths[0])
  rgb_paths = [x[vis_i : vis_i + 1] for x in rgb_paths]
  if config.transfer_detail:
    outputs['rgba'] = transfer_detail(
        outputs['rgba'],
        outputs['recon'],
        batch['frame'][:, :, vis_i : vis_i + 1],
    )
  outputs['rgba'] = outputs['rgba'].reshape(
      shp[:2] + (-1,) + shp[-2:]
  )  # B,4,L*T,H,W
  save_batch(
      outputs['rgba'], mask_paths, prefix=f'{save_dir}/rgba_pred/', mode='RGBA'
  )
  save_batch(outputs['recon'], rgb_paths, f'{save_dir}/recon/')
  save_batch(
      batch['mask'], mask_paths, f'{save_dir}/mask_input/', mode='L'
  )
  save_batch(
      batch['bg'][:, :, vis_i : vis_i + 1], rgb_paths, f'{save_dir}/bg/'
  )


def log_images(sw, batch, outputs, step, config, prefix='', n_vis=8):
  """Visualizes inputs and outputs in Tensorboard."""
  n_objs = batch['mask'].shape[2]
  vis_frame = config.vis_frame
  # take one device's batch
  outputs = jax.tree_map(lambda x: x[0, :n_vis], outputs)
  batch = jax.tree_map(lambda x: x[0, :n_vis], batch)
  # concatenate layers vertically for visualization
  outputs['inp_mask'] = np.concatenate(
      [outputs['inp_mask'][:, i : i + 1] for i in range(n_objs)], -2
  )
  outputs['rgba'] = np.concatenate(
      [outputs['rgba'][:, :, i] for i in range(n_objs)], -2
  )

  def transform_for_vis(x):
    return np.einsum('bchw->bhwc', np.clip(x * 0.5 + 0.5, 0, 1))

  # visualize only device 0 and one target frame per sequence
  outputs = jax.tree_map(
      lambda x: transform_for_vis(x[:, :, vis_frame]), outputs
  )
  alpha = outputs['rgba'][..., 3:4]
  with sw.as_default():
    tf.summary.image(f'{prefix}2_mask_in', outputs['inp_mask'], step=step)
    tf.summary.image(f'{prefix}4_rgba_pred', outputs['rgba'], step=step)
    tf.summary.image(f'{prefix}3_alpha_pred', alpha, step=step)
    tf.summary.image(
        f'{prefix}5_bg',
        transform_for_vis(batch['bg'][:, :3, vis_frame]),
        step=step,
    )
    tf.summary.image(f'{prefix}1_recon', outputs['recon'], step=step)
    rgb_in = transform_for_vis(batch['frame'][:, :, vis_frame])
    tf.summary.image(f'{prefix}0_rgb_in', rgb_in, step=step)


def log_scalars(sw, scalars, step):
  """Logs loss terms in Tensorboard."""
  with sw.as_default():
    for loss_term in scalars:
      tf.summary.scalar(loss_term, scalars[loss_term], step=step)
