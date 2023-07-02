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

