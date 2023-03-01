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

from abc import ABC, abstractmethod
import glob
import os

import jax
import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds


def read_crop_im(im, im_width, im_height, channels=3, order='CHW', crop=None):
  """`crop` should be [start_y, end_y, start_x, end_x] normalized [0, 1]."""
  # convert the compressed string to a 3D float tensor
  im = tf.io.decode_jpeg(im, channels=channels)
  if crop:
    height = float(tf.shape(im)[0])
    width = float(tf.shape(im)[1])
    starty = int(crop[0] * height)
    endy = int(crop[1] * height)
    startx = int(crop[2] * width)
    endx = int(crop[3] * width)
    im = im[starty:endy, startx:endx]
  im = tf.image.resize(im, [im_height, im_width], antialias=True)
  if order == 'CHW':
    im = tf.transpose(im, (2, 0, 1))
  return tf.cast(im, tf.float32) / 255.


def generate_random_crop(no_crop=.2):
  if tf.random.uniform([]) < no_crop:
    crop = (0., 1., 0., 1.)
  else:
    crop_ = tf.random.uniform(shape=[4], minval=0, maxval=.1)
    crop = (crop_[0] * 0.5, 1 - crop_[1], crop_[2], 1 - crop_[3])
  return crop


def generate_mask(mask_mode,
                  sequence_length,
                  num_masked,
                  random_flip=True):
  """Generates a binary sequence indicating whether each frame is visible to the model."""
  if num_masked == 0:
    valid = tf.convert_to_tensor([1.]*sequence_length)
  elif mask_mode == 'rand':
    frames, _, _ = tf.random.uniform_candidate_sampler(
        [list(range(0, sequence_length))],  # true_classes
        sequence_length,  # num_true
        num_masked,  # num_sampled,
        True,  # unique
        sequence_length  # range_max
    )
    valid = 1 - tf.reduce_sum(tf.one_hot(frames, sequence_length), axis=0)
  elif mask_mode == 'pred':
    valid_arr = [1.] * (sequence_length - num_masked) + [0.] * num_masked
    valid = tf.convert_to_tensor(valid_arr)
    if random_flip and tf.random.uniform([]) < 0.5:
      valid = tf.convert_to_tensor(valid_arr[::-1])
  elif mask_mode == 'inpaint':
    mask = [1.] * sequence_length
    start_i = sequence_length // 2 - num_masked // 2
    mask[start_i:start_i+num_masked] = [0.]*num_masked
    valid = tf.convert_to_tensor(mask)
  return valid


def read_order_from_file(filepath):
  with open(filepath) as f:
    order = f.readlines()
  order = [o.split(' ') for o in order]
  for i in range(len(order)):
    for j in range(len(order[i])):
      order[i][j] = order[i][j].rstrip()
  return order


def get_dataloader(
    config: ml_collections.ConfigDict, mode='train', split='all'
):
  if config.dataset_structure == 'kubric':
    print('loading kubric')
    return KubricDataLoader(config, mode, split)
  else:
    print('loading an omnimatte-formatted video')
    return OmnimatteDataLoader(config, mode, split)


def reshape_batch(batch):
  def reshape_tensor_batchdims(x):
    return x.reshape((jax.local_device_count(), -1,) + x.shape[1:])
  return jax.tree_map(reshape_tensor_batchdims, batch)


class BaseDataLoader(ABC):
  """Handles loading the dataset and reshaping the batches for multi-device training."""

  @abstractmethod
  def _get_viddirs(self, config):
    pass

  @abstractmethod
  def get_file_basename(self, path):
    """
    Args:
      path: path to mask file, e.g. `rootdir/vidname/mask0_0001.png`.
    Returns: filename with only the frame ID, e.g. `vidname/0001.png`
    """
    pass

  @abstractmethod
  def get_rgb_frames(self, viddir):
    pass

  @abstractmethod
  def get_maskpath(self, viddir, obj_id, fn):
    pass

  @abstractmethod
  def get_bgpath(self, viddir):
    pass

  @abstractmethod
  def get_nobjs(self, viddir):
    pass

  def __init__(self, config, mode, split):
    self.config = config
    self._load_dataset(config, mode, split)

  def _load_dataset(self, config, mode, split):
    dset = self.get_dataset(config, mode=mode, split=split)
    max_buffer_size = 5000
    buffer_size = min(dset.cardinality(), max_buffer_size)
    if buffer_size <= 0:
      buffer_size = max_buffer_size
    if mode == 'train':
      dataset = iter(
          tfds.as_numpy(
              dset.shuffle(buffer_size).repeat(-1).batch(
                  config.batch_size, drop_remainder=True).prefetch(-1)))
    else:
      dataset = iter(
          tfds.as_numpy(
              dset.batch(
                  config.batch_size, drop_remainder=True).prefetch(-1)))
    self.dataset = dataset
    self.current_batch = None
    self.current_mask_path = None

  def get_next_batch(self, reshape=True):
    batch = next(self.dataset, None)
    if batch and reshape:
      batch = reshape_batch(batch)
    self.current_batch = batch
    if batch:
      self.current_mask_path = batch.pop('mask_path')
    else:
      self.current_mask_path = None
    return self.current_batch

  def get_dataset(
      self, config: ml_collections.ConfigDict, mode='train', split='all'):
    dset = config.dataset
    gap = config.gap
    max_vids = config.max_vids
    stride = 1
    sequence_length = config.sequence_length
    mask_mode = config.mask_mode
    num_masked = config.num_masked
    augment_flip_mask = mode == 'train'
    iw = config.input_width
    ih = config.input_height
    viddirs = self._get_viddirs(config)

    has_valid_mask = os.path.exists(f'{viddirs[0]}/valid_mask')

    if len(viddirs) > 1 and split != 'all':
      nval = len(viddirs) // 10
      if split == 'train':
        viddirs = viddirs[:-nval]
      else:
        viddirs = viddirs[-nval:]
    viddirs = viddirs[:max_vids]
    print(f'videos found: {len(viddirs)}')
    n_objs = self.get_nobjs(viddirs[0])

    list_image_paths, list_mask_paths, list_bg_paths = [], [], []
    if has_valid_mask:
      list_valid_paths = []
    for viddir in viddirs:
      order = self.get_depth_order(viddir)
      frames = self.get_rgb_frames(viddir)
      # sliding window of N frames
      for i in range(0, len(frames) - (sequence_length - 1) * gap, stride):
        rgb_seq = frames[i:i + sequence_length * gap:gap]
        image_paths = ','.join(rgb_seq)
        mask_paths_arr = []
        for j in range(n_objs):
          for k, fn in enumerate(rgb_seq):
            obj_id = order[i+k*gap][j]
            mask_paths_arr.append(self.get_maskpath(viddir, obj_id, fn))
        mask_path = ','.join(mask_paths_arr)
        list_image_paths.append(image_paths)
        list_mask_paths.append(mask_path)
        list_bg_paths.append(self.get_bgpath(viddir))
        if has_valid_mask:
          list_valid_paths.append(image_paths.replace('/rgb/', '/valid_mask/'))

    paths_dict = {
        'image_path': tf.constant(list_image_paths),
        'mask_path': tf.constant(list_mask_paths),
        'bg_path': tf.constant(list_bg_paths)
    }
    if has_valid_mask:
      paths_dict['valid_path'] = tf.constant(list_valid_paths)

    list_ds = tf.data.Dataset.from_tensor_slices(paths_dict)
    image_count = list_ds.cardinality()

    if mode == 'train':
      list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)

    def process_path(paths):
      image, mask, valid_mask = [], [], []
      parts = {k: tf.strings.split(paths[k], ',') for k in paths}
      if mode == 'train':
        crop = generate_random_crop()
      else:
        crop = (0, 1, 0, 1)
      for i in range(sequence_length):
        image.append(
            read_crop_im(
                tf.io.read_file(parts['image_path'][i]), iw, ih, crop=crop
            )
        )
        if has_valid_mask:
          # valid regions after stabilization
          valid_mask_i = read_crop_im(
              tf.io.read_file(parts['valid_path'][i]), iw, ih, crop=crop
          )[:1]
          valid_mask.append(valid_mask_i)
      for i in range(sequence_length * n_objs):
        mask.append(
            read_crop_im(
                tf.io.read_file(parts['mask_path'][i]),
                iw,
                ih,
                channels=1,
                crop=crop,
            )
        )
      image = tf.stack(image, 1)  # 3,T,H,W
      mask = tf.reshape(tf.concat(mask, 0), (-1, *image.shape[1:]))  # L,T,H,W
      mask = tf.cast(
          tf.greater(mask, .6 * tf.ones_like(mask)),
          dtype=tf.float32)  # remove gray trimap region TODO: delete
      if has_valid_mask:
        valid_mask = tf.stack(valid_mask, 1)  # 1,T,H,W
      valid = generate_mask(
          mask_mode, sequence_length, num_masked, random_flip=augment_flip_mask
      )
      valid = tf.reshape(valid, [1, -1, 1, 1]) * tf.ones_like(image[:1])

      bg = read_crop_im(tf.io.read_file(paths['bg_path']), iw, ih, crop=crop)
      bg = tf.stack([bg] * sequence_length, 1)

      data = {
          'frame': image,  # 3,T,H,W
          'mask': mask,  # L,T,H,W
          'valid_frames': valid,  # 1,T,H,W
          'bg': bg  # 3,T,H,W
      }
      if has_valid_mask:
        data['valid_pixels'] = valid_mask  # 1,T,H,W

      # map to [-1, 1]
      data = jax.tree_map(lambda x: x * 2 - 1, data)
      data['mask_path'] = paths['mask_path']
      return data

    dset = list_ds.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    print('len dset: ', dset.cardinality())
    return dset


class KubricDataLoader(BaseDataLoader):

  def _get_viddirs(self, config):
    if config.stage == 'ttt':
      viddirs = [config.datadir]
    else:
      viddirs = glob.glob(f'{config.datadir}/*')
    return viddirs

  def get_file_basename(self, path):
    """
    Args:
      path: path to mask file, e.g. `rootdir/vidname/mask0_0001.png`.
    Returns: filename with only the frame ID, e.g. `vidname/0001.png`
    """
    return f'{os.path.dirname(path)}/{path.split("_")[-1]}'

  def get_rgb_frames(self, viddir):
    return sorted(glob.glob(f'{viddir}/rgba*'))

  def get_maskpath(self, viddir, obj_id, fn):
    basename = fn.split('_')[-1]
    return f'{viddir}/mask{obj_id}_{os.path.splitext(basename)[0]}.png'

  def get_bgpath(self, viddir):
    bg_name = 'bg' if self.config.use_gt_bg else 'bg_est'
    bg_path = f'{viddir}/{bg_name}.png'
    return bg_path

  def get_nobjs(self, viddir):
    return len(glob.glob(f'{viddir}/mask*_00000.png'))

  def get_depth_order(self, viddir):
    return read_order_from_file(f'{viddir}/order.txt')


class OmnimatteDataLoader(BaseDataLoader):
  def _get_viddirs(self, config):
    return [config.datadir]

  def get_file_basename(self, path):
    return os.path.basename(path)

  def get_rgb_frames(self, viddir):
    return sorted(glob.glob(f'{viddir}/rgb/*'))

  def get_maskpath(self, viddir, obj_id, fn):
    return f'{viddir}/mask/{obj_id}/{fn.split("/")[-1]}'

  def get_bgpath(self, viddir):
    bg_path = f'{self.config.datadir}/bg_est.png'
    return bg_path

  def get_nobjs(self, viddir):
    return len(os.listdir(f'{viddir}/mask'))

  def get_depth_order(self, viddir):
    if os.path.exists(f'{viddir}/order.txt'):
      order = read_order_from_file(f'{viddir}/order.txt')
    else:
      # create order
      obj_dirs = sorted(os.listdir(f'{viddir}/mask/'))
      nframes = len(os.listdir(f'{viddir}/rgb'))
      order = [obj_dirs] * nframes
    return order
