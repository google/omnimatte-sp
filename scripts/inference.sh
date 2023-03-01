#!/bin/bash
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

#
# Run inference.
#
EXPNAME=pretrain_shadows-v1
WEIGHTS_PATH=pretrained_weights/$EXPNAME/model_weights/0
SAVEDIR=inference
DATADIR=data/kubric-shadows-v1-2obj-test
python main.py \
  --config=configs/inference.py \
  --config.load_weights_path=$WEIGHTS_PATH \
  --save_dir=$SAVEDIR  \
  --experiment_name=$EXPNAME \
  --config.datadir=$DATADIR \
  --config.dataset_structure=kubric \
  --config.input_height=128 \
  --config.input_width=128

DATADIR=data/kubric-shadows-v1-4obj-test
python main.py \
  --config=configs/inference.py \
  --config.load_weights_path=$WEIGHTS_PATH \
  --save_dir=$SAVEDIR  \
  --experiment_name=$EXPNAME \
  --config.datadir=$DATADIR \
  --config.dataset_structure=kubric \
  --config.input_height=128 \
  --config.input_width=128
