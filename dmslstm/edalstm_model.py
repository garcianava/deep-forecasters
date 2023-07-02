import tensorflow.compat.v1 as tf


# LSTM-based encoder - decoder without attention mechanism
# hourly-resolution only
class EDALSTM(object):
    # pass features (set of tensors) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the LSTM layers

        encoder_stack_h, encoder_last_h, encoder_last_c = tf.keras.layers.LSTM(
            units=model_params['encoder']['no_hidden'],
            activation=model_params['encoder']['activation'],
            dropout=model_params['encoder']['dropout'],
            recurrent_dropout=model_params['encoder']['recurrent_dropout'],
            return_sequences=True,
            return_state=True,
            name='encoder')(features['hourly'])
        # encoder_stack_h is the output from return_sequences=True
        # encoder_stack_h has shape (?, no_input_timesteps, no_hidden)
        # for this execution is (?, 256, 256)
        # encoder_last_h and encoder_last_c are the output from return_state=True
        # encoder_last_h and encoder_last_c have shape (?, no_hidden)
        # for this execution is (?, 256)

        if model_params['use_batch_normalization']:
            # ToDo: include momentum parameter in configuration dictionary
            encoder_last_h = tf.keras.layers.BatchNormalization(momentum=0.6)(encoder_last_h)
            encoder_last_c = tf.keras.layers.BatchNormalization(momentum=0.6)(encoder_last_c)

        # ToDo: get this parameter from configuration dictionary ['no_targets']
        decoder_input = tf.keras.layers.RepeatVector(24)(encoder_last_h)
        # decoder_input has shape (?, no_targets, no_hidden)
        # for this execution is (?, 24, 256)

        # calculate alignment score from stacked hidden state
        decoder_stack_h = tf.keras.layers.LSTM(
            units=model_params['decoder']['no_hidden'],
            activation=model_params['decoder']['activation'],
            # activation='elu',
            dropout=model_params['decoder']['dropout'],
            recurrent_dropout=model_params['decoder']['recurrent_dropout'],
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

        if model_params['use_batch_normalization']:
            context = tf.keras.layers.BatchNormalization(momentum=0.6)(context)

        # context vector has shape (?, no_targets, no_hidden)
        # for this execution is (?, 24, 256)

        decoder_combined_context = tf.keras.layers.concatenate([context, decoder_stack_h])
        # decoder_combined_context has shape (?, no_targets, 2*no_hidden)
        # for this execution is (?, 24, 512)

        # ToDo: use a dictionary to manage dense layer levels
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

        # ToDo: use a dictionary to manage dense layer outputs by level
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
