# operating system utilities
# import os
# import sys

# date-time management utilities
# import datetime
# import time

# advanced numerical operations
import numpy as np

# Python data analysis
# import pandas as pd

# Parquet support for Pandas
# import pyarrow

import tensorflow as tf

# normalize data sets to improve LSTM network performance
# from sklearn.preprocessing import MinMaxScaler

# persistence for the scaler
from sklearn.externals import joblib

# Anaconda Interactive Visualization
from bokeh.plotting import figure, show
from bokeh.plotting import output_file, save
# from bokeh.io import output_notebook
# from bokeh.models import ColumnDataSource
# from bokeh.models import Span

_EQUIPMENT = 'CPE04105'


# retrieve data sets from data arrays path
# it is assumed they have already been persisted to disk

Xadj_train = np.load('../data/arrays/Xadj_train.npy', allow_pickle=True)
Xday_train = np.load('../data/arrays/Xday_train.npy', allow_pickle=True)
Xweek_train = np.load('../data/arrays/Xweek_train.npy', allow_pickle=True)
Xtsweekday_train = np.load('../data/arrays/Xtsweekday_train.npy', allow_pickle=True)
Xtshour_train = np.load('../data/arrays/Xtshour_train.npy', allow_pickle=True)
y_train = np.load('../data/arrays/y_train.npy', allow_pickle=True)
yts_train = np.load('../data/arrays/yts_train.npy', allow_pickle=True)

Xadj_val = np.load('../data/arrays/Xadj_val.npy', allow_pickle=True)
Xday_val = np.load('../data/arrays/Xday_val.npy', allow_pickle=True)
Xweek_val = np.load('../data/arrays/Xweek_val.npy', allow_pickle=True)
Xtsweekday_val = np.load('../data/arrays/Xtsweekday_val.npy', allow_pickle=True)
Xtshour_val = np.load('../data/arrays/Xtshour_val.npy', allow_pickle=True)
y_val = np.load('../data/arrays/y_val.npy', allow_pickle=True)
yts_val = np.load('../data/arrays/yts_val.npy', allow_pickle=True)

Xadj_test = np.load('../data/arrays/Xadj_test.npy', allow_pickle=True)
Xday_test = np.load('../data/arrays/Xday_test.npy', allow_pickle=True)
Xweek_test = np.load('../data/arrays/Xweek_test.npy', allow_pickle=True)
Xtsweekday_test = np.load('../data/arrays/Xtsweekday_test.npy', allow_pickle=True)
Xtshour_test = np.load('../data/arrays/Xtshour_test.npy', allow_pickle=True)
y_test = np.load('../data/arrays/y_test.npy', allow_pickle=True)
yts_test = np.load('../data/arrays/yts_test.npy', allow_pickle=True)

# reload scaler fitted model here
scaler = joblib.load('../data/scalers/ci_LSTM_scaler.save')

# get targets for only the first model (first-step-ahead)
n = 0
ytarget_train = y_train[:, n]
ytarget_val = y_val[:, n]

ytarget_test = y_test[:, n]

# reshaping is required for input_fn's
ytarget_train = ytarget_train.reshape(ytarget_train.shape[0], 1)
ytarget_val = ytarget_val.reshape(ytarget_val.shape[0], 1)
ytarget_test = ytarget_test.reshape(ytarget_test.shape[0], 1)

print('Shapes for target tensors are {0}, {1}, {2}'.format(ytarget_train.shape,
                                                           ytarget_val.shape,
                                                           ytarget_test.shape))

# somehow this data sets are not consistent with training and evaluation features
# ToDo: define if data is reshaped before importing or in ETL
Xadj_test = Xadj_test.reshape(Xadj_test.shape[0], Xadj_test.shape[1], 1)
Xday_test = Xday_test.reshape(Xday_test.shape[0], Xday_test.shape[1], 1)
Xweek_test = Xweek_test.reshape(Xweek_test.shape[0], Xweek_test.shape[1], 1)

# verify Xadj_train, ytarget_train shapes, as they come from disk
# ToDo: change prefixes
# Xadj to Xadjhour
# Xday to Xadjday
# Xweek to Xadjweek
# ToDo: verify if prefixed variables can be changed to dictionary keys

print('Shapes for adjacent hour data sets are {0}, {1}, {2}'.format(Xadj_train.shape,
                                                                    Xadj_val.shape,
                                                                    Xadj_test.shape))

print('Shapes for adjacent day data sets are {0}, {1}, {2}'.format(Xday_train.shape,
                                                                   Xday_val.shape,
                                                                   Xday_test.shape))

print('Shapes for adjacent week data sets are {0}, {1}, {2}'.format(Xweek_train.shape,
                                                                    Xweek_val.shape,
                                                                    Xweek_test.shape))

# ToDo: get a timestamp-based identifier and use it to mark the model directory
_MODEL_DIR = 'cudnn_lstm_100_epochs'
_LEARNING_RATE = 0.001
_NUM_EPOCHS = 100
_BATCH_SIZE = 32

_NO_HIDDEN_UNITS_HOUR = 128
_NO_HIDDEN_UNITS_DAY = 64
# _M_HOUR = 24
_LSTM_DROPOUT = 0.2
_LSTM_UNROLL = False
# TensorFlow LSTM implementation mode:
# Mode 1 will structure its operations as a larger number of smaller dot products and additions,
# whereas mode 2 will batch them into fewer, larger operations.
_LSTM_IMPLEMENTATION_MODE = 1

_OUTPUT_ACTIVATION = 'sigmoid'


def train_input_fn():
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    'adjacent_hours': tf.cast(Xadj_train, tf.float32),
                    'adjacent_days': tf.cast(Xday_train, tf.float32),
                    'adjacent_weeks': tf.cast(Xweek_train, tf.float32),
                },
                tf.cast(ytarget_train, tf.float32)
            )
        )
    )
    dataset = dataset.shuffle(buffer_size=9000).repeat(count=1).batch(batch_size=_BATCH_SIZE)
    return dataset


def eval_input_fn():
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    'adjacent_hours': tf.cast(Xadj_val, tf.float32),
                    'adjacent_days': tf.cast(Xday_val, tf.float32),
                    'adjacent_weeks': tf.cast(Xweek_val, tf.float32),
                },
                tf.cast(ytarget_val, tf.float32)
            )
        )
    )
    dataset = dataset.repeat(count=1).batch(batch_size=_BATCH_SIZE)
    return dataset


def test_input_fn():
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
                {
                    'adjacent_hours': tf.cast(Xadj_test, tf.float32),
                    'adjacent_days': tf.cast(Xday_test, tf.float32),
                    'adjacent_weeks': tf.cast(Xweek_test, tf.float32),
                },
                tf.cast(ytarget_test, tf.float32)
            )
        )
    )
    dataset = dataset.repeat(count=1).batch(batch_size=_BATCH_SIZE)
    return dataset


class Model(object):
    # pass features (set of tensors) as inputs when calling model
    def __call__(self, features):
        # supervised learning database as input for the LSTM layers
        db_hours = features['adjacent_hours']
        db_days = features['adjacent_days']
        # db_weeks = features['adjacent_weeks']

        lstm_hours_1 = tf.keras.layers.CuDNNLSTM(units=_NO_HIDDEN_UNITS_HOUR,
                                                 return_sequences=True,
                                                 name='lstm_hours_1')(db_hours)

        lstm_hours_2 = tf.keras.layers.CuDNNLSTM(units=_NO_HIDDEN_UNITS_HOUR,
                                                 return_sequences=False,
                                                 name='lstm_hours_2')(lstm_hours_1)

        # ToDo: verify what type of dropout regularization can be applied inside the LSTM block
        dropout_hours = tf.keras.layers.Dropout(0.2)(lstm_hours_2)

        lstm_days_1 = tf.keras.layers.CuDNNLSTM(units=_NO_HIDDEN_UNITS_DAY,
                                                return_sequences=False,
                                                name='lstm_days_1')(db_days)

        dropout_days = tf.keras.layers.Dropout(0.2)(lstm_days_1)

        lstm_merge = tf.keras.layers.concatenate([dropout_hours, dropout_days])

        dense_1 = tf.keras.layers.Dense(units=64, activation='relu', name='output_1')(lstm_merge)

        dense_2 = tf.keras.layers.Dense(units=32, activation='relu', name='output_2')(dense_1)

        dense_3 = tf.keras.layers.Dense(units=8, activation='relu', name='output_3')(dense_2)

        forecast = tf.keras.layers.Dense(units=1, activation=_OUTPUT_ACTIVATION, name='output_4')(dense_3)

        return forecast


def model_fn(features, labels, mode):
    # instantiate Model in model
    model = Model()
    global_step = tf.train.get_global_step()

    # call to model
    forecast = model(features)

    # PREDICT
    predictions = {
        "forecast": forecast
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    with tf.name_scope('loss'):
        mean_squared_error = tf.losses.mean_squared_error(
            labels=labels, predictions=forecast, scope='loss')
        tf.summary.scalar('loss', mean_squared_error)

    with tf.name_scope('val_loss'):
        val_loss = tf.metrics.mean_squared_error(
            labels=labels, predictions=forecast, name='mse')
        tf.summary.scalar('val_loss', val_loss[1])

    # EVAL
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mean_squared_error,
            eval_metric_ops={'val_loss': val_loss},
            evaluation_hooks=None)

    # define optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=_LEARNING_RATE)
    train_op = optimizer.minimize(
        mean_squared_error, global_step=global_step)

    # Create a hook to print acc, loss & global step every 100 iter
    train_hook_list = []
    train_tensors_log = {'val_loss': val_loss[1],
                         'loss': mean_squared_error,
                         'global_step': global_step}
    train_hook_list.append(tf.train.LoggingTensorHook(
        tensors=train_tensors_log, every_n_iter=100))

    # TRAIN
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=mean_squared_error,
            train_op=train_op,
            training_hooks=train_hook_list)


# ToDo: evaluate the convenience of packaging execution in a tf.app
# in the meantime, run the estimator outside tf.app package

tf.logging.set_verbosity(tf.logging.INFO)

# create a estimator with model_fn
lstm_forecaster = tf.estimator.Estimator(model_fn=model_fn,
                                         model_dir=_MODEL_DIR)
# train and evaluate the model after each epoch
for _ in range(_NUM_EPOCHS):
    lstm_forecaster.train(input_fn=train_input_fn)
    metrics = lstm_forecaster.evaluate(input_fn=eval_input_fn)

# build predictions using the trained model
pred = list(lstm_forecaster.predict(input_fn = test_input_fn))
pred = [p['forecast'][0] for p in pred]
pred = np.asarray(pred)
pred_ci = scaler.inverse_transform(pred.reshape(-1, 1))
pred_ci = np.squeeze(pred_ci)
actual_ci = scaler.inverse_transform(ytarget_test)
actual_ci = np.squeeze(actual_ci)

ci_predictions_fig = figure(title='Predicted Current Imbalance for ' + _EQUIPMENT,
                            background_fill_color='#E8DDCB',
                            plot_width=1800, plot_height=450, x_axis_type='datetime')

ci_predictions_fig.line(yts_test[:,n],
                        actual_ci, line_color='red',
                        line_width=1, alpha=0.7, legend='Actual')

ci_predictions_fig.line(yts_test[:,n],
                        pred_ci, line_color='blue',
                        line_width=1, alpha=0.7, legend='Predicted')

ci_predictions_fig.legend.location = "top_right"
ci_predictions_fig.legend.background_fill_color = "darkgrey"

ci_predictions_fig.xaxis.axis_label = 'Timestamp'
ci_predictions_fig.yaxis.axis_label = 'Current Imbalance [%]'

output_file('../plots/' + '{:03d}'.format(n) + '.html', title=_EQUIPMENT)
save(ci_predictions_fig)
