import numpy as np
import tensorflow.compat.v1 as tf


class DMSLSTM(object):
    # pass features (set of tensors) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the LSTM layers

        # collect one-hot timestamps for week/day and day/hour to merge them as features
        # oh:wd: encoded day of the week
        flat_one_hot_week_day = tf.keras.layers.Flatten()(features['oh_wd'])
        # oh_dh: encoded hour of the day
        flat_one_hot_day_hour = tf.keras.layers.Flatten()(features['oh_dh'])
        one_hot_merge = tf.keras.layers.concatenate([flat_one_hot_week_day, flat_one_hot_day_hour])

        # a dictionary to store the resolution-based LSTM stacks
        lstm = dict()
        # a dictionary to store the dropout layers for resolution-based LSTM stacks
        dropout = dict()

        # build the LSTM stacks for three time resolutions
        for resolution in ['hourly', 'daily', 'weekly']:
            lstm[resolution] = list()
            lstm[resolution].append(features[resolution])
            for level in np.arange(len(model_params[resolution]['structure'])):
                is_this_last_level = level == len(model_params[resolution]['structure']) - 1
                block_name = 'lstm_{}_{}'.format(resolution, level)
                lstm[resolution].append(
                    tf.keras.layers.LSTM(
                        units=model_params[resolution]['structure'][level],
                        # if multi-level, then LSTM blocks between first
                        # and the one before last must return sequences
                        return_sequences=not is_this_last_level,
                        dropout=model_params[resolution]['dropout'],
                        unroll=model_params[resolution]['unroll'],
                        implementation=model_params[resolution]['implementation_mode'],
                        name=block_name)(lstm[resolution][-1])
                )
            # ToDo: verify what type of dropout regularization can be applied inside the LSTM block
            # point the dropout to the last LSTM cell on the stack
            dropout[resolution] = tf.keras.layers.Dropout(
                model_params[resolution]['dropout'])(lstm[resolution][-1])

        # merge results of DMSLSTM branches
        if model_params['use_timestamps']:
            lstm_merge = tf.keras.layers.concatenate([dropout['hourly'],
                                                      dropout['daily'],
                                                      dropout['weekly'],
                                                      one_hot_merge])
        else:
            lstm_merge = tf.keras.layers.concatenate([dropout['hourly'],
                                                      dropout['daily'],
                                                      dropout['weekly']])

        # build dense layer as a list
        dense_layer = list()
        # input to dense layer is the vector LSTM results + timestamps
        dense_layer.append(lstm_merge)
        # iterate on dense structure to build dense levels
        for level in np.arange(len(model_params['dense']['structure'])):
            block_name = 'dense_{}'.format(level)
            dense_layer.append(
                tf.keras.layers.Dense(
                    units=model_params['dense']['structure'][level],
                    activation=model_params['dense']['activation'][level],
                    name=block_name
                )(dense_layer[-1])
            )

        # finally, make the forecast equal to the last layer of 'Dense'
        # (which is referenced by the most recent record in the dense_layer list
        forecast = dense_layer[-1]
        # reshape the forecast tensor to be consistent with target in the parsed datasets,
        # target dimension for multi-step forecasting is equal
        # to the last value in params['dense']['structure']
        # (and this value remains in params['dense']['structure'][level]
        # after 'dense' layer construction)
        # however, let's use a direct reference, just to be sure
        no_targets = model_params['dense']['structure'][-1]
        forecast = tf.keras.layers.Reshape((no_targets, 1))(forecast)

        return forecast


# Stacked LSTM-based encoder - decoder with attention mechanism
# hourly-resolution only
# based on Jeremy Wortz' code from
# https://stackoverflow.com/questions/50915634/multilayer-seq2seq-model-with-lstm-in-keras
class EDSLSTM(object):
    # pass features (set of tensors) as inputs at call operator
    def __call__(self, features, model_params):
        # supervised learning database as input for the LSTM layers
        # start with [256, 256, 256] LSTM stacks for encoder and decoder, generalize later

        # build the encoder object
        # generalize the construction of the encoder layers

        # a list of indexes for the encoder layers
        # use any encoder feature list to get the number of layers
        # for our final case, with 3 LSTM layers, indexes = [0, 1, 2]
        indexes = list(np.arange(len(model_params['encoder']['no_hidden'])))

        # load all the encoder feature lists
        no_hiddens = model_params['encoder']['no_hidden']
        activations = model_params['encoder']['activation']
        dropouts = model_params['encoder']['dropout']
        recurrent_dropouts = model_params['encoder']['recurrent_dropout']

        # a dict to store all layers of the encoder
        encoder = dict()

        for index, no_hidden, activation, dropout, recurrent_dropout in zip(indexes,
                                                                            no_hiddens,
                                                                            activations,
                                                                            dropouts,
                                                                            recurrent_dropouts):
            '''
            zip returns an iterable like this one
            0 256 elu 0.2 0.2
            1 256 elu 0.2 0.2
            2 256 elu 0.2 0.2
            '''

            level_key = 'level_{}'.format(index)
            encoder[level_key] = tf.keras.layers.LSTM(
                units=no_hidden,
                activation=activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                # LSTMs in all levels return sequences
                return_sequences=True,
                # LSTM in the uppermost level returns also the state
                return_state=index == indexes[-1],
                name='encoder_{}'.format(level_key))

        # now build a list to generalize the encoder output management
        encoder_output = list()

        # initialize the encoder output list with the features passed as input to the model
        encoder_output.append(features['hourly'])

        # build a list of level keys (on indexes) to iterate on
        level_keys = ['level_{}'.format(index) for index in indexes]

        # now process the encoder stack of LSTMs as a cascade:
        # list[n] is the input for list[n+1]
        for level_key in level_keys:
            encoder_output.append(encoder[level_key](encoder_output[-1]))

            '''
            encoder['level_0'] LSTM receives features['hourly']
                               and returns encoder_stack_h0 (return_sequences=True)
            encoder['level_1'] LSTM receives encoder_stack_h0
                               and returns encoder_stack_h1 (return_sequences=True)
            encoder['level_n-1'] LSTM receives encoder_stack_hn-2
                               and returns encoder_stack_hn-1 (return_sequences=True)
            encoder['level_n'] LSTM receives encoder_stack_hn-1 and returns the tuple
                               (encoder_stack_hn, encoder_last_hn, encoder_last_cn)
                               (return_sequences=True, return_state=True)
            '''


        # annotate all tensor dimensions to verify consistency
        # encoder_stack_h0, 1, 2 has shape (?, features['hourly'].timesteps, no_hidden) (?, 64, 256)
        # encoder_stack_hn has shape (?, encoder_stack_h1.timesteps, no_hidden) (?, 64, 256)
        # encoder_last_hn has shape (?, no_hidden) (?, 256)
        # encoder_last_cn has shape (?, no_hidden) (?, 256)

        # where do the relevant encoder stack outputs lie now?
        # hidden state stack of the lowermost layer
        # encoder_stack_h0 = encoder_output[1]
        # hidden state stack of the uppermost layer
        encoder_stack_hn = encoder_output[-1][0]
        # last hidden state of the uppermost layer
        encoder_last_hn = encoder_output[-1][1]
        # last cell state of the uppermost layer
        encoder_last_cn = encoder_output[-1][2]

        if model_params['use_batch_normalization']:
            encoder_last_hn = tf.keras.layers.BatchNormalization(
                momentum=model_params['encoder']['momentum_h'])(encoder_last_hn)
            encoder_last_cn = tf.keras.layers.BatchNormalization(
                momentum=model_params['encoder']['momentum_c'])(encoder_last_cn)

        # last hidden state in encoder stack is passed as decoder input
        decoder_input = tf.keras.layers.RepeatVector(
            model_params['no_targets'])(encoder_last_hn)
        # decoder_input has shape (?, repeat, encoder_last_h2.no_hidden)
        # for this execution is (?, 24, 256)

        # generalize now the decoder object construction
        # a list of indexes for the decoder layers
        # use any decoder feature list to get the number of layers
        indexes = list(np.arange(len(model_params['decoder']['no_hidden'])))

        # load all the decoder feature lists
        no_hiddens = model_params['decoder']['no_hidden']
        activations = model_params['decoder']['activation']
        dropouts = model_params['decoder']['dropout']
        recurrent_dropouts = model_params['decoder']['recurrent_dropout']

        # a dict to store all layers of the decoder
        decoder = dict()
        for index, no_hidden, activation, dropout, recurrent_dropout in zip(indexes,
                                                                            no_hiddens,
                                                                            activations,
                                                                            dropouts,
                                                                            recurrent_dropouts):
            '''
            zip returns an iterable like this one
            0 256 elu 0.2 0.2
            1 256 elu 0.2 0.2
            2 256 elu 0.2 0.2
            '''

            level_key = 'level_{}'.format(index)
            decoder[level_key] = tf.keras.layers.LSTM(
                units=no_hidden,
                activation=activation,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                # LSTMs in all levels return sequences
                return_sequences=True,
                # LSTMs in all levels do not return state
                return_state=False,
                name='decoder_{}'.format(level_key))

        # now build a list to generalize the decoder output management
        decoder_output = list()

        # initialize the decoder output list with the TimeDistributed decoder input
        decoder_output.append(decoder_input)

        # build a list of level keys (on indexes) to iterate on
        level_keys = ['level_{}'.format(index) for index in indexes]

        # now process the decoder stack of LSTMs as a cascade:
        # list[n] is the input for list[n+1]
        for level_key in level_keys:
            # last hidden state and last cell state in encoder stack
            # are passed as initial state for first decoder layer
            # ToDo: test parallel initialization of the decoder layers
            #       with last_h and last_c from the encoder layers
            initial_state = [encoder_last_hn, encoder_last_cn] if level_key == 'level_0' else None
            # ensure initial_state is passed as a call argument
            decoder_output.append(decoder[level_key](decoder_output[-1], initial_state=initial_state))


        # decoder_stack_h0, 1,... up to n have shape
        # (?, decoder_input.timesteps, no_hidden) (?, 24, 256)

        # where do the relevant decoder stack outputs lie now?
        # hidden state stack of the uppermost layer
        decoder_stack_h0 = decoder_output[1]
        decoder_stack_hn = decoder_output[-1]

        # ToDo: change model architecture according to GNMT stacking
        #       (Google's Neural Machine Translation, 2016):
        #       alignment score from decoder_stack_h0 and encoder_stack_hn
        #       use residual connections in LSTM layers above second level
        #       additional: initialize all decoder layers with
        #       encoder_last_h and encoder_last_c from
        #       the corresponding layers in the encoder, later...
        #
        # build alignment from uppermost LSTM in decoder
        # and lowermost LSTM in encoder
        # alignment = tf.keras.layers.dot([decoder_stack_hn, encoder_stack_h0], axes=[2, 2])

        # change to GNMT stacking, then
        # build alignment from lowermost LSTM in decoder
        # and uppermost LSTM in encoder
        alignment = tf.keras.layers.dot([decoder_stack_h0, encoder_stack_hn], axes=[2, 2])
        alignment = tf.keras.layers.Activation('softmax')(alignment)
        # alignment has shape [24, 256]dot[64, 256], axes=[2, 2]
        # for this execution is (?, 24, 64)

        # build context from alignment and the uppermost LSTM in encoder
        context = tf.keras.layers.dot([alignment, encoder_stack_hn], axes=[2, 1])
        # context has shape [24, 64]dot[64, 256], axes=[2, 1]
        # for this execution is (?, 24, 256)

        if model_params['use_batch_normalization']:
            context = tf.keras.layers.BatchNormalization(
                momentum=model_params['context_momentum'])(context)

        # ToDo: try different approaches to build decoder_combined_context
        decoder_combined_context = tf.keras.layers.concatenate([context, decoder_stack_hn])
        # decoder_combined_context has shape [24, 256]concatenate[24, 256]
        # for this execution is (?, 24, 512)

        # build a TimeDistributed Dense flow to produce a multi-layer output

        # get dense layer structure and activations (two lists)
        structure = model_params['dense']['structure']
        activation = model_params['dense']['activation']
        # get indexes for structure levels (as a list)
        indexes = list(np.arange(len(structure)))

        # a dictionary to store the dense layer levels
        dense = dict()
        # iterate via zip on indexes, cells, and activations
        for index, no_units, activation in zip(indexes, structure, activation):
            level_key = 'level_{}'.format(index)
            dense[level_key] = tf.keras.layers.Dense(
                units=no_units,
                activation=activation,
                name='dense_layer_{}'.format(level_key)
            )

        # generalize the dense layer outputs using a list
        output = list()
        # initialize the output list with decoder_combined_context
        output.append(decoder_combined_context)

        # build a list of level keys (on indexes) to iterate on
        level_keys = ['level_{}'.format(index) for index in indexes]

        for level_key in level_keys:
            output.append(tf.keras.layers.TimeDistributed(
                dense[level_key])(output[-1]))

        # at the end of the building loop, the uppermost level of the dense layer
        # is located in the final position of the output list
        forecast = output[-1]

        return forecast
