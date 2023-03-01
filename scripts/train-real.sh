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
# Reproduce the real video experiments.
# Pretrain on Synthetic V2 data and then test-time train on a single video.
#
SEED=0
DATASET=shadows-v2
TRAIN_DATADIR=data/kubric-$DATASET-train
TTT_DATADIR=data/dogwalk
SAVEDIR=checkpoints/
BATCHSIZE=64
MASK_MODE=rand
EXPNAME=pretrain_${DATASET}_${MASK_MODE}

echo "experiment name: ${EXPNAME}"
# Pretrain on Synthetic V2 data
python main.py \
  --config=configs/pretrain.py \
  --save_dir=$SAVEDIR  \
  --config.mode=train_eval \
  --experiment_name=$EXPNAME \
  --config.batch_size=$BATCHSIZE \
  --config.mask_mode=$MASK_MODE \
  --config.num_steps=60000 \
  --config.datadir=$TRAIN_DATADIR \
  --config.input_height=128 \
  --config.input_width=224 \
  --config.seed_model=$SEED \
  --config.seed_data=$SEED

WEIGHTS_PATH=${SAVEDIR}/${EXPNAME}_lr0.0010_b${BATCHSIZE}_sm${SEED}_sd${SEED}/model_weights/0/
# Test-time train on a real video
python main.py \
  --config=configs/ttt.py \
  --save_dir=$SAVEDIR  \
  --config.mode=train_eval \
  --experiment_name=ttt_${DATASET}_${MASK_MODE} \
  --config.load_weights_path=$WEIGHTS_PATH \
  --config.mask_mode=$MASK_MODE \
  --config.datadir=$TTT_DATADIR \
  --config.dataset_structure=omnimatte \
  --config.input_height=128 \
  --config.input_width=224 \
  --config.num_steps=1000 \
  --config.seed_model=$SEED \
  --config.seed_data=$SEED \
  --config.transfer_detail

