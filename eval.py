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

"""Compute evaluation metrics on results."""

import glob
import json
from multiprocessing import Pool
import os

from absl import app
from absl import flags
import numpy as np
from src import utils

flags.DEFINE_string('resdir', None, 'root directory of results')
flags.DEFINE_string('datadir', None, 'root directory of data')
flags.DEFINE_string('save_json_path', None, 'json file to save results')
flags.DEFINE_integer('max_vids', 10000, '')

FLAGS = flags.FLAGS


def read_parallel(im_paths, num_workers=12, size=None, pil_format='RGB'):
  num_workers = os.cpu_count()
  if size is None:
    size = (None, None)
  with Pool(num_workers) as p:
    ims = p.starmap(
        utils.read_image,
        zip(
            im_paths,
            *zip(*([size] * len(im_paths))),
            [pil_format] * len(im_paths),
        ),
    )
  return ims


def comp_iou(pred, gt):
  assert pred.max() <= 1
  assert pred.min() >= 0
  assert gt.max() <= 1
  assert gt.min() >= 0
  intersection = pred * gt
  union = np.clip(pred + gt, 0, 1)
  return intersection.sum() / union.sum()


def get_nobjs(datadir):
  nmasks = len(glob.glob(f'{datadir}/mask*'))
  return nmasks // 24


def eval_single(rgbadir, datadir):
  """Compute IoU for a single video."""
  # read RGBA layers from results dir, render over BG, compare with rendered
  bg_path = f'{datadir}/bg_128.png'
  bg = utils.read_image(bg_path, width=128, height=128)[np.newaxis]
  ious = {}
  n_objs = get_nobjs(datadir)
  for obj in range(n_objs):
    rgba_pred_paths = sorted(glob.glob(f'{rgbadir}/mask{obj}*'))
    rgba_preds = np.stack(
        read_parallel(rgba_pred_paths, pil_format='RGBA'), 0
        ).astype(np.float32)
    alpha_preds = rgba_preds[..., -1:] / 255.0
    rgb_preds = rgba_preds[..., :3]
    # composite over bg
    preds = alpha_preds * rgb_preds + (1 - alpha_preds) * bg

    diff = np.sum((preds - bg) ** 2, -1)
    thresh = 0.25 * 255
    bin_diff = (diff > thresh).astype(np.float32)
    gt_paths = [
        f'{datadir}/gt/{os.path.basename(fn)}' for fn in rgba_pred_paths
    ]
    gts = np.stack(
        read_parallel(gt_paths, size=(128, 128), pil_format='L'), 0
    ).astype(np.float32)
    bin_diff_gt = gts / 255.

    iou = comp_iou(bin_diff, bin_diff_gt)
    ious[obj] = float(iou)

  return ious


def main(_):
  datadir = FLAGS.datadir
  vidnames = sorted(os.listdir(f'{FLAGS.resdir}/rgba_pred'))[:FLAGS.max_vids]
  ious = {}
  for vidname in vidnames:
    rgbadir = f'{FLAGS.resdir}/rgba_pred/{vidname}'
    print(f'computing iou for {rgbadir}')
    iou = eval_single(rgbadir, f'{datadir}/{vidname}')
    ious[rgbadir] = iou

  ious['mean'] = float(np.mean([np.mean(list(ious[x].values())) for x in ious]))

  with open(FLAGS.save_json_path, 'w') as outfile:
    outfile.write(json.dumps(ious, indent=2, sort_keys=True))

  print(f'saved to {FLAGS.save_json_path}')


if __name__ == '__main__':
  app.run(main)
