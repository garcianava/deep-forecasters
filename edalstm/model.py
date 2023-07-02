
import tensorflow.compat.v1 as tf
# import tensorflow as tf

import numpy as np

# ToDo: remove dependency to __init__.py
#  bring structure, dropout, unroll, and implementation mode for all LSTM branches from configuration file
#  bring use_timestamps flag from configuration file
#  bring structure, activation function and output activation function from configuration file
# from dplstm import architecture_parameters
from configs.edalstm_config import architecture_parameters


# LSTM-based encoder - decoder without attention mechanism
# hourly-resolution only
class EDALSTM(object):
    # pass features (set of tensors) as inputs at call operator
    def __call__(self, features):
        # supervised learning database as input for the LSTM layers

        encoder_stack_h, encoder_last_h, encoder_last_c = tf.keras.layers.LSTM(
            units=architecture_parameters['encoder']['no_hidden'],
            activation=architecture_parameters['encoder']['activation'],
            dropout=architecture_parameters['encoder']['dropout'],
            recurrent_dropout=architecture_parameters['encoder']['recurrent_dropout'],
            return_sequences=True,
            return_state=True,
            name='encoder')(features['hourly'])
        # encoder_stack_h is the output from return_sequences=True
        # encoder_stack_h has shape (?, no_input_timesteps, no_hidden)
        # for this execution is (?, 256, 256)
        # encoder_last_h and encoder_last_c are the output from return_state=True
        # encoder_last_h and encoder_last_c have shape (?, no_hidden)
        # for this execution is (?, 256)

        if architecture_parameters['use_batch_normalization']:
            encoder_last_h = tf.keras.layers.BatchNormalization(momentum=0.6)(encoder_last_h)
            encoder_last_c = tf.keras.layers.BatchNormalization(momentum=0.6)(encoder_last_c)

        # ToDo: get this parameter from sldb_parameters['no_targets']
        decoder_input = tf.keras.layers.RepeatVector(24)(encoder_last_h)
        # decoder_input has shape (?, no_targets, no_hidden)
        # for this execution is (?, 24, 256)

        # calculate alignment score from stacked hidden state
        decoder_stack_h = tf.keras.layers.LSTM(
            units=architecture_parameters['decoder']['no_hidden'],
            activation=architecture_parameters['decoder']['activation'],
            # activation='elu',
            dropout=architecture_parameters['decoder']['dropout'],
            recurrent_dropout=architecture_parameters['decoder']['recurrent_dropout'],
            return_state=False,
            return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        # decoder_stack has shape (?, no_targets, no_hidden)
        # for this execution is (?, 24, 256)

        attention = tf.keras.layers.dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        # ToDo: test with an activation different from softmax
        attention = tf.keras.layers.Activation('softmax')(attention)
        # attention has shape (?, no_targets, no_input_timesteps)
        # for this execution is (?, 24, 256)

        context = tf.keras.layers.dot([attention, encoder_stack_h], axes=[2, 1])

        if architecture_parameters['use_batch_normalization']:
            context = tf.keras.layers.BatchNormalization(momentum=0.6)(context)

        # context vector has shape (?, no_targets, no_hidden)
        # for this execution is (?, 24, 256)

        decoder_combined_context = tf.keras.layers.concatenate([context, decoder_stack_h])
        # decoder_combined_context has shape (?, no_targets, 2*no_hidden)
        # for this execution is (?, 24, 512)

        # dense_layer = tf.keras.layers.Dense(
        # units=architecture_parameters['dense']['no_units'],
        # activation=architecture_parameters['dense']['activation'],
        # name='dense_layer')

        # use_dense_layer_list = False
        # if use_dense_layer_list:
        #     # build dense output layer as a list
        #     dense_layer = list()
        #     # the first dense unit in the dense structure has no previous dense unit
        #     # to be applied on, therefore
        #     level = 0
        #     block_name = 'dense_{}'.format(level)
        #     dense_layer.append(
        #         tf.keras.layers.Dense(
        #             units=architecture_parameters['dense']['structure'][level],
        #             activation=architecture_parameters['dense']['activation'][level],
        #             name=block_name
        #         )
        #     )
        #     # if there is more than one item in the dense output structure
        #     if len(architecture_parameters['dense']['structure']) > 1:
        #         # iterate on the remaining items of the dense structure
        #         for level in np.arange(1, len(architecture_parameters['dense']['structure'])):
        #             block_name = 'dense_{}'.format(level)
        #             dense_layer.append(
        #                 tf.keras.layers.Dense(
        #                     units=architecture_parameters['dense']['structure'][level],
        #                     activation=architecture_parameters['dense']['activation'][level],
        #                     name=block_name
        #                 )(dense_layer[-1])
        #             )

        # build a TimeDistributed Dense flow to produce a multi-layer output
        dense_layer_level_0 = tf.keras.layers.Dense(
            units=128,
            activation='relu',
            name='dense_layer_level_0')

        dense_layer_level_1 = tf.keras.layers.Dense(
            units=32,
            activation='relu',
            name='dense_layer_level_1')

        dense_layer_level_2 = tf.keras.layers.Dense(
            units=8,
            activation='relu',
            name='dense_layer_level_2')

        dense_layer_level_3 = tf.keras.layers.Dense(
            units=1,
            activation='sigmoid',
            name='dense_layer_level_3')

        level_0_output = tf.keras.layers.TimeDistributed(dense_layer_level_0)(decoder_combined_context)
        # level_0_output has shape (?, no_targets, no_units_dense_layer_level_0)
        # for this execution is (?, 24, 128)

        level_1_output = tf.keras.layers.TimeDistributed(dense_layer_level_1)(level_0_output)
        # level_1_output has shape (?, no_targets, no_units_dense_layer_level_1)
        # for this execution is (?, 24, 32)

        level_2_output = tf.keras.layers.TimeDistributed(dense_layer_level_2)(level_1_output)
        # level_2_output has shape (?, no_targets, no_units_dense_layer_level_2)
        # for this execution is (?, 24, 8)

        level_3_output = tf.keras.layers.TimeDistributed(dense_layer_level_3)(level_2_output)
        # level_3_output has shape (?, no_targets, no_units_dense_layer_level_3)
        # for this execution is (?, 24, 1), no need to reshape

        forecast = level_3_output

        return forecast
