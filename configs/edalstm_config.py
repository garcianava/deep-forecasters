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
# ToDo: avoid redundancy for sldb parameters declaration,
#       as they are present both in sldbs/sldb_id/sldb.json
#       and here!
sldb_parameters = {
    # use the SLDB_DIRECTORY as value
    # 'path': 'CPE04115_H_kw_20201021084001_008001_008024_008168_024',
    'path': 'CPE04115_H_kw_20201021084001_256001_024',
    # the following five values come from TIME_SERIES_DIRECTORY/ts.json
    'device': 'CPE04115',
    'variable': 'kw',
    'resolution': 'H',
    'start_timestamp': '2016-01-01 00:00:00',
    'end_timestamp': '2018-07-31_23:00:00',
    # ToDo: automatically get the following three values from sldb.json
    #       hard-wire them in the meantime
    'embedding': {
        'hourly': 256
    },
    'tau': {
        'hourly': 1
    },
    # it is very likely the value of no_targets will be the same for all time resolutions
    'no_targets': 24,

    # ToDo: automatically get the following two values from sldbs/SLDB_DIRECTORY/sldb.json
    # this on from ['stats']['train']['trimmed_to_count']
    'total_train_rows': 17824,
    # this on from ['stats']['eval']['trimmed_to_count']
    'total_eval_rows': 1984,
    # this on from ['stats']['test']['trimmed_to_count']
    'total_test_rows': 1984
}

architecture_parameters = {
    'encoder': {
        'no_hidden': 256,
        'activation': 'elu',
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
    },
    'decoder': {
        'no_hidden': 256,
        'activation': 'elu',
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
    },
    'use_batch_normalization': True,
    # last hidden state from LSTM branches are passed to MLPs for auto-balance
    'dense': {
        # the last value in the following list must be equal to sldb_parameters['no_targets']
        # 'no_units': 1,
        # be sure the following list has the same length as the one above
        # 'activation': 'sigmoid',
        # ToDo: implement a multi-layer perceptron structure by using lists
        'structure': [128, 32, 8, 1],
        'activation': ['relu', 'relu', 'relu', 'sigmoid']
    }
}

training_parameters = {
    # remove model_dir from configuration dictionary to enforce using it via Abseil Flags
    # 'model_dir': 'gs://cbidmltsf/models/some_model_directory',
    # ToDo: remove train_data_path and eval_data_path from configuration dictionary
    #  to enforce using it via Abseil Flags.
    #  Aggregate them into a single data_dir variable
    # remove data_dir from configuration dictionary to enforce using it via Abseil Flags
    # remove data_dir from configuration dictionary to enforce using it via Abseil Flags
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
    # 'use_tpu': False,
    'tpu': 'tpu-v3-8-tensorflow-1-15-preemptible',
    'tpu_zone': 'us-central1-a',
    'skip_host_call': False,
    # ToDo: make learning rate schedule parameters flag-'able'
    #   use the ones from official ResNet in the meantime
    # 'lrs_total_epochs': 20,
    # 'lrs_steps': [1, 6, 12, 18],
    # 'lrs_weights': [1.0, 0.1, 0.01, 0.001]
    'lrs': [(1.0, 2),
            (0.5, 10),
            (0.2, 15),
            (0.1, 20)]
}

# num_train_rows is calculated given the number of cores
# ToDo: pass this code snippet to train.py
#  and do not persist the adjusted num_train_rows
training_parameters['num_train_rows'] = \
    sldb_parameters['total_train_rows'] -\
    sldb_parameters['total_train_rows'] % training_parameters['num_cores']

# num_eval_rows is calculated given the number of cores
# ToDo: pass this code snippet to train.py
#  and do not persist the adjusted num_eval_rows
training_parameters['num_eval_rows'] = \
    sldb_parameters['total_eval_rows'] -\
    sldb_parameters['total_eval_rows'] % training_parameters['num_cores']

# eval_batch_size is set to num_eval_rows (which was calculated given the number of cores)
# ToDo: pass this code snippet to train.py
#  and do not persist the adjusted eval_batch_size
training_parameters['eval_batch_size'] = training_parameters['num_eval_rows']

# in the official ResNet50 Cloud TPU demo, there is only one configuration dictionary, like this one:
# DMSLSTM_CFG = {
#     'sldb_parameters': {
#     },
#     'architecture_parameters': {
#     },
#     'training_parameters': {
#     }
# }

EDALSTM_RESTRICTIONS = [
]
# pylint: enable=line-too-long
