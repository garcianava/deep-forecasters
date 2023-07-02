from configs.edalstm_config import sldb_parameters
from configs.edalstm_config import training_parameters

import tensorflow.compat.v1 as tf

read_features = {
    'hourly': tf.io.VarLenFeature(dtype=tf.float32),
    'target': tf.io.VarLenFeature(dtype=tf.float32),
    'oh_wd': tf.io.VarLenFeature(dtype=tf.float32),
    'oh_dh': tf.io.VarLenFeature(dtype=tf.float32),
    'timestamp': tf.io.VarLenFeature(dtype=tf.string)
}


def _parse_dataset_function(example_proto, objective_shapes, parse_timestamp):
    # parse the input tf.Example proto using the dictionary above
    row = tf.io.parse_single_example(example_proto, read_features)
    # pass objective shape as a list of lists [hourly_shape, daily_shape, weekly_shape]
    hourly = tf.reshape(row['hourly'].values, objective_shapes['hourly'])
    target = tf.reshape(row['target'].values, objective_shapes['target'])
    oh_wd = tf.reshape(row['oh_wd'].values, objective_shapes['oh_wd'])
    oh_dh = tf.reshape(row['oh_dh'].values, objective_shapes['oh_dh'])
    # do not parse the timestamp to TPUEstimator, as it does not support string types!
    # ToDo: code timestamps into features, as numbers
    #  so they can be parsed to training
    timestamp = tf.reshape(row['timestamp'].values, objective_shapes['timestamp'])
    # the parsed dataset must have the shape {features}, target!!!
    # so:
    feature_dict = {
        'hourly': hourly,
        'oh_wd': oh_wd,
        'oh_dh': oh_dh,
    }
    # Do not parse the timestamp for training!!! Strings are not supported in TPUs!!!,
    # or parse it as a number
    if parse_timestamp:
        feature_dict['timestamp'] = timestamp

    # _parse_dataset_function returns:
    # features as a dictionary, and
    # target as a float scalar
    # ToDo: review notebook for making sldbs, it must persist target as a vector
    # return feature_dict, target[0]
    return feature_dict, target


# pass the batch_size as argument of _input_fn
def make_input_fn(tfrecord_path, mode):

    def _input_fn(params):

        batch_size = params['batch_size']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # get the dataset from a TFRecord file
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        if is_training:
            # shuffle the whole training dataset
            dataset = dataset.shuffle(training_parameters['num_train_rows'])
            # repeat the dataset indefinitely
            dataset = dataset.repeat(count=None)

        # ToDo: pass objective_shapes dictionary instead of the hard-wired list
        # store the objective shapes for reshaping tensors in a dictionary
        _TRAIN_OBJECTIVE_SHAPES = {
            'hourly': [sldb_parameters['embedding']['hourly'], 1],
            'target': [sldb_parameters['no_targets'], 1],
            'oh_wd': [7, 1],  # Monday to Sunday
            'oh_dh': [24, 1],  # midnight to 23:00
            # ToDo: verify only the initial timestamp is passed to the model
            'timestamp': [sldb_parameters['no_targets'], 1]
        }

        dataset = dataset.map(lambda row: _parse_dataset_function(example_proto=row,
                                                                  objective_shapes=_TRAIN_OBJECTIVE_SHAPES,
                                                                  parse_timestamp=False),
                              num_parallel_calls=training_parameters['num_cores'])

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        # changed from Distributed TensorFlow CPU/GPU to single-device TPU
        # ToDo: return to Distributed TensorFlow TPU after successful implementation

        # ToDo: verify application of transposing, later...
        # ToDo: verify application of tf.contrib.data.parallel_interleave, later...

        # for TPU, prefetch data while training
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    return _input_fn


def serving_input_fn():
    # ToDo: for some reason, serving objective shapes are tensors with just one row,
    #       find out why!
    _SERVING_OBJECTIVE_SHAPES = {
        'hourly': [1, sldb_parameters['embedding']['hourly'], 1],
        'target': [1, sldb_parameters['no_targets'], 1],
        'oh_wd': [1, 7, 1],  # Monday to Sunday
        'oh_dh': [1, 24, 1],  # midnight to 23:00
        # ToDo: verify only the initial timestamp is passed to the model
        'timestamp': [1, sldb_parameters['no_targets'], 1]
    }

    # TPU are not optimized for serving, so it is assumed the predictions server is CPU or GPU-based
    # inputs is equivalent to example protocol buffers
    feature_placeholders = {'example_bytes': tf.placeholder(tf.string, shape=())}

    # the serving input function does not require the label
    # the following underscore discards the target (not required for serving predictions)
    features, _ = _parse_dataset_function(example_proto=feature_placeholders['example_bytes'],
                                          objective_shapes=_SERVING_OBJECTIVE_SHAPES,
                                          # parse timestamps to use them in plotting results
                                          parse_timestamp=True)

    # re-shape to original model spec
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)