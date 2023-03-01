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

import ml_collections


def get_config():
  """Config for inference."""
  config = ml_collections.ConfigDict()

  config.mode = 'eval'

  config.load_weights_path = ''

  # dataset options
  config.input_width = 224
  config.input_height = 128
  config.dataset = ''
  config.datadir = ''
  config.dataset_structure = 'kubric'
  config.gap = 3
  config.max_vids = 10000
  config.sequence_length = 5
  config.use_gt_bg = True
  config.mask_mode = 'pred'
  config.num_masked = 3

  # train options, unused
  config.batch_size = 64
  config.stage = 'pretrain'
  config.train_log_every = 100
  config.num_steps = 60000
  config.save_every = 20000
  config.eval_every = 60001
  config.seed_model = 0
  config.seed_data = 0
  config.learning_rate = .001
  config.lambda_alpha_l0 = .05
  config.lambda_alpha_l1 = .05
  config.lambda_mask = 1.

  config.do_tensorboard = True
  config.vis_frame = -1 # if -1, defaults to middle frame
  config.transfer_detail = False
  return config


def get_hyper(h):
  return h.product([], name='config')
