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

"""Run training and inference."""

import os
import shutil

from absl import app
from absl import flags
from absl import logging
import jax
import jax.numpy as jnp
import ml_collections
from ml_collections import config_flags
import numpy as np
from src import dataset
from src import omnimatte_sp
from src import utils
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

config = config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('experiment_name', 'lnr_example', '')
flags.DEFINE_string('save_dir', None,
                    'Path to directory to save final predictions.')

FLAGS = flags.FLAGS


def inference(config: ml_collections.ConfigDict, exp_dir: str):
  if 'test' not in config.dataset:
    config.max_vids = 50
  if 'eval' in config.mode and config.dataset_structure != 'kubric':
    config.batch_size = jax.local_device_count()
  config.use_gt_bg = True
  host_id = jax.host_id()
  # Initialize the model on all devices.
  rng = jax.random.PRNGKey(config.seed_model)

  # Load datasets
  if config.stage == 'pretrain' and 'train' in config.mode:
    train_set = dataset.get_dataloader(config, mode='eval', split='train')
    val_set = dataset.get_dataloader(config, mode='eval', split='val')
    dataloaders = [('train', train_set), ('val', val_set)]
  else:
    dataloader = dataset.get_dataloader(config, mode='eval', split='all')
    dataloaders = [('test', dataloader)]
  batch = dataloaders[0][1].get_next_batch()
  n_objs = batch['mask'].shape[2]  # D,B,L,T,H,W

  config_frozen = ml_collections.FrozenConfigDict(config)
  params, state, _ = omnimatte_sp.make_initial_state(
      _broadcast_to_devices(rng), batch, config_frozen
  )
  print('Model initialized!')
  if 'train' in config.mode:
    chkpt_path = f'{exp_dir}/model_weights/{host_id}'
  else:
    chkpt_path = config.load_weights_path

  try:
    chkpt = utils.restore_chkpt(chkpt_path)
    chkpt_params = utils.interpolate_pos_embs(chkpt['params'], params)
    params = _broadcast_to_devices(chkpt_params)
    print('Model restored!')
  except Exception:
    print('Model could not be restored')

  loss_weights = {
      'lambda_alpha_l0': config.lambda_alpha_l0,
      'lambda_alpha_l1': config.lambda_alpha_l1,
      'lambda_mask': config.lambda_mask,
  }
  loss_weights = jax.tree_map(
      lambda x: x * jnp.ones([jax.local_device_count()]), loss_weights
  )
  eval_fn = omnimatte_sp.eval_step
  for subset, dset in dataloaders:
    vis_dir = f'{exp_dir}/results/{config.dataset}_{subset}_{config.mask_mode}{config.num_masked}_vis{config.vis_frame}_{chkpt["step"]}'
    if host_id == 0 and os.path.exists(vis_dir):
      shutil.rmtree(vis_dir)
    if dset.current_batch is None:
      batch = dset.get_next_batch()
    while batch is not None:
      mask_path = dset.current_mask_path
      rng, key = jax.random.split(rng)
      key = _broadcast_to_devices(key)
      _, outputs = eval_fn(
          params, state, key, batch, loss_weights, config_frozen
      )
      outputs = jax.device_get(outputs)
      utils.save_results(dset, outputs, mask_path, vis_dir, config)
      batch = dset.get_next_batch()
    utils.make_grids(config, vis_dir, n_objs)


def _broadcast_to_devices(x):
  return jax.tree_map(
      lambda y: jnp.broadcast_to(y, (jax.local_device_count(),) + y.shape), x
  )


def train(config: ml_collections.ConfigDict, exp_dir: str):
  print('starting training')

  # Set hyper-parameters.
  eval_every = config.eval_every  # save val results
  save_every = config.save_every  # save model chkpt

  train_dataset = dataset.get_dataloader(config, mode='train', split='train')
  val_dataset = dataset.get_dataloader(config, mode='train', split='val')

  host_id = jax.host_id()
  is_main_host = host_id == 0

  # Initialize the model on all devices.
  rng = jax.random.PRNGKey(config.seed_model)
  batch = train_dataset.get_next_batch()

  config_frozen = ml_collections.FrozenConfigDict(config)

  params, state, opt_state = omnimatte_sp.make_initial_state(
      _broadcast_to_devices(rng), batch, config_frozen)
  chkpt_path = f'{exp_dir}/model_weights/{host_id}'
  step = 0
  logging.info('Model initialized!')
  if config.load_weights_path:
    old_chkpt = utils.restore_chkpt(config.load_weights_path)
    old_chkpt_params = utils.interpolate_pos_embs(old_chkpt['params'], params)
    params = _broadcast_to_devices(old_chkpt_params)
    print('loaded old model weights')
  elif os.path.exists(f'{chkpt_path}/tree.pkl'):  # continue training from last checkpoint
    try:
      chkpt = utils.restore_chkpt(chkpt_path)
      chkpt_params = utils.interpolate_pos_embs(chkpt['params'], params)
      params = _broadcast_to_devices(chkpt_params)
      opt_state = _broadcast_to_devices(chkpt['opt_state'])
      step = chkpt['step']
      logging.info('Model restored!')
    except Exception:
      print('Model could not be restored from previous checkpoint.')

  log_every = config.train_log_every
  loss_weights = {
      'lambda_alpha_l0': config.lambda_alpha_l0,
      'lambda_alpha_l1': config.lambda_alpha_l1,
      'lambda_mask': config.lambda_mask,
  }
  loss_weights = jax.tree_map(
      lambda x: x * jnp.ones([jax.local_device_count()]), loss_weights
  )
  lambda_mask_sched = [
      min(int(0.05 * config.num_steps), 1000),
      min(int(0.1 * config.num_steps), 2000),
  ]
  if step > lambda_mask_sched[0]:
    loss_weights['lambda_mask'] *= .1
  if step > lambda_mask_sched[1]:
    loss_weights['lambda_mask'] *= .1

  train_fn = omnimatte_sp.train_step
  eval_fn = omnimatte_sp.eval_step

  # Set up JAXboard and checkpointing.
  if is_main_host and config.do_tensorboard:
    sw_dir = os.path.join(FLAGS.save_dir, 'vis', os.path.basename(exp_dir))
    sw_t = tf.summary.create_file_writer(f'{sw_dir}_train')
    sw_v = tf.summary.create_file_writer(f'{sw_dir}_val')
  val_every = 5
  while step < config.num_steps:
    step += 1
    rng, key = jax.random.split(rng)
    key = _broadcast_to_devices(key)
    batch = train_dataset.get_next_batch()
    params, state, opt_state, train_results, outputs, batch = train_fn(
        params, state, opt_state, key, batch, loss_weights, config_frozen)

    train_results = jax.tree_map(lambda x: x[0], train_results)
    train_results = jax.device_get(train_results)
    assert not jnp.isnan(train_results['loss'])
    if step % 20 == 0:
      log = f'[{step}/{config.num_steps}] '
      for l in train_results:
        log += f'{l}={train_results[l]:0.6f} '
      print(log)
    # update loss schedule
    if step == lambda_mask_sched[0] or step == lambda_mask_sched[1]:
      loss_weights['lambda_mask'] *= .1

    if is_main_host and config.do_tensorboard:
      utils.log_scalars(sw_t, train_results, step)
      if step % log_every == 1:
        utils.log_images(sw_t, batch, outputs, step, config)
        sw_t.flush()

    # validation
    if config.do_tensorboard and step % val_every == 1:
      batch = val_dataset.get_next_batch()
      val_results, outputs = eval_fn(
          params, state, key, batch, loss_weights, config_frozen)
      if is_main_host:
        val_results = jax.tree_map(lambda x: x[0], val_results)
        val_results = jax.device_get(val_results)
        utils.log_scalars(sw_v, val_results, step)
        if step % log_every == 1:
          utils.log_images(sw_v, batch, outputs, step, config)
          sw_v.flush()

    if step % save_every == 0:
      print(f'saving at step {step}')
      utils.save_chkpt(chkpt_path, params, opt_state, step)

    if is_main_host and step % eval_every == 0:
      print('running inference')
      inference(config, exp_dir)

  # Save final layers, model weights, and losses.
  if is_main_host and config.do_tensorboard:
    sw_t.flush()
    sw_t.close()
    sw_v.flush()
    sw_v.close()

  utils.save_chkpt(chkpt_path, params, opt_state, step)

  print('finished training')


def main(_):
  logging.info(
      'Device count: %d, host count: %d, local device count: %d',
      jax.device_count(),
      jax.host_count(),
      jax.local_device_count(),
  )
  config = FLAGS.config
  assert (
      config.batch_size % jax.local_device_count() == 0
  ), 'Batch size should be a multiple of device count'
  host_id = jax.host_id()
  tf.random.set_seed(config.seed_data + host_id)
  np.random.seed(config.seed_data + host_id)
  config.dataset = os.path.basename(config.datadir)
  exp_dirname = FLAGS.experiment_name
  if 'train' in config.mode:
    exp_dirname += (
        f'_lr{config.learning_rate:.4f}'
        f'_b{config.batch_size:d}'
        f'_sm{config.seed_model}'
        f'_sd{config.seed_data}'
    )
  exp_dir = os.path.join(FLAGS.save_dir, exp_dirname)
  if config.vis_frame == -1:
    if config.mask_mode == 'pred' and config.num_masked > 0:
      # visualize the first predicted frame
      config.vis_frame = config.sequence_length - config.num_masked
    else:
      # visualize the middle frame
      config.vis_frame = config.sequence_length // 2
  if 'train' in config.mode:
    train(config, exp_dir)
  if 'eval' in config.mode and host_id == 0:
    if config.mask_mode == 'rand':
      config.mask_mode = 'pred'
    inference(config, exp_dir)

  if jax.host_count() > 1:
    # Make sure all hosts stay up until the end of main.
    x = jnp.ones([jax.local_device_count()])
    x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    assert x[0] == jax.device_count()
  print('Done. Experiment directory: ', exp_dir)


if __name__ == '__main__':
  app.run(main)
