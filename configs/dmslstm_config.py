# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Config template to train DMSLSTM."""
# based on http://github.com/tensorflow/tpu/blob/master/models/official/resnet/configs/resnet_config.py

# pylint: disable=line-too-long

# rename the main dictionary in configuration file to avoid shadowing
# the argument parameters in train_and_evaluate function
DMSLSTM_CFG = {
    # use the SLDB_DIRECTORY as value
    'path': 'CPE04115_H_kw_20201021084001_008001_008024_008168_024',
    # ToDo: get the following five values from TIME_SERIES_DIRECTORY/ts.json
    'device': 'CPE04115',
    'variable': 'kw',
    'resolution': 'H',
    'start_timestamp': '2016-01-01 00:00:00',
    'end_timestamp': '2018-07-31_23:00:00',
    # ToDo: automatically get the following three values from sldb.json
    #   hard-wire them in the meantime
    'embedding': {
        'hourly': 8,
        'daily': 8,
        'weekly': 8
    },
    'tau': {
        'hourly': 1,
        'daily': 24,
        'weekly': 168
    },
    # it is very likely the value of no_targets will be the same for all time resolutions
    'no_targets': 24,

    # ToDo: automatically get the following two values from sldbs/SLDB_DIRECTORY/sldb.json
    # this on from ['stats']['train']['trimmed_to_count']
    'total_train_rows': 16736,
    # this on from ['stats']['eval']['trimmed_to_count']
    'total_eval_rows': 896,
    # this on from ['stats']['test']['trimmed_to_count']
    'total_test_rows': 896,

    'hourly': {
        'structure': [64, 128],
        'dropout': 0.2,
        'unroll': False,
        'implementation_mode': 1
    },
    'daily': {
        'structure': [64, 128],
        'dropout': 0.2,
        'unroll': False,
        'implementation_mode': 1
    },
    'weekly': {
        'structure': [64, 128],
        'dropout': 0.2,
        'unroll': False,
        'implementation_mode': 1
    },
    'use_timestamps': False,
    'dense': {
        # the last value in the following list must be equal to sldb_parameters['no_targets']
        'structure': [512, 128, 24],
        # be sure the following list has the same length as the one above
        # 'activation': 'relu',
        # 'output_activation': 'sigmoid'
        'activation': ['relu', 'relu', 'sigmoid']
    },

    # remove model_dir from configuration dictionary to enforce using it via Abseil Flags
    # 'model_dir': 'gs://cbidmltsf/models/some_model_directory',
    # ToDo: remove train_data_path and eval_data_path from configuration dictionary
    #  to enforce using it via Abseil Flags.
    #  Aggregate them into a single data_dir variable
    # remove data_dir from configuration dictionary to enforce using it via Abseil Flags
    # 'data_dir': 'gs://cbidmltsf/sldbs/some_sldb_directory'
    'base_learning_rate': 0.8,
    # in order to get to get lr=[0.1, 0.01, 0.001, 0.0001] at batch_size=32
    # remove train_steps from configuration dictionary to enforce using it via Abseil Flags
    # 'train_steps': 400,
    # remove train_batch_size from configuration dictionary to enforce using it via Abseil Flags
    # 'train_batch_size': 32,
    'eval_interval': 300,
    # remove iterations_per_loop from configuration dictionary to enforce using it via Abseil Flags
    # 'iterations_per_loop': 200,
    'keep_checkpoint_max': 3,
    'start_delay_secs': 600,
    'throttle_secs': 600,
    'job-dir': 'junk',
    'project': 'spheric-rhythm-234515',
    'num_cores': 8,
    # remove use_tpu from configuration dictionary to enforce using it via Abseil Flags.
    'use_tpu': False,
    'tpu': 'tpu-v3-8-tensorflow-1-15-preemptible',
    'tpu_zone': 'us-central1-a',
    'skip_host_call': False,
    # ToDo: make learning rate schedule parameters flag-'able'
    #   use the ones from official ResNet in the meantime
    'lrs_steps': [5, 30, 60, 80],
    'lrs_weights': [1.0, 0.1, 0.01, 0.001],
    # discard num_cores remainder from total_train_rows
    'num_train_rows': 16736,
    # discard num_cores remainder from total_eval_rows
    'num_eval_rows': 896,
    # set batch size for evaluation
    'eval_batch_size': 896,
    'mode': 'train_and_eval',
    # logging parameters
    'save_summary_steps': 25,
    'log_step_count_steps': 250
}

DMSLSTM_RESTRICTIONS = [
]

# pylint: enable=line-too-long
