import os
import time
import json
import numpy as np

from absl import flags
from absl import app

# ToDo: pass this code to the setup.py file of the final module!!!
# temporarily solved: added cbidmltsf dir to PYTHONPATH by exporting in .bashrc!!!

# uncomment the following two lines if using Abseil Flags to pass hyperparameters
# from configs import common_hparams_flags
# from configs import common_tpu_params

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

from google.cloud import storage

from forecasters.models import EDALSTM

# uncomment the following two lines if using Abseil Flags to pass hyperparameters
# common_hparams_flags.define_common_hparams_flags()
# common_tpu_params.define_common_tpu_flags()

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'config_json_file', default=None,
    help=('A JSON file which specifies parameters: '
          'general (at root level dictionary), '
          'sldb (supervised-learning database), '
          'architecture (model network), and '
          'training.'))

flags.DEFINE_integer(
    'execution', default=None,
    help=('The execution number for this experiment.'
          'From 00, up to 99.'))


# ToDo: break train_and_evaluate function into the three execution modes
#  included in the official ResNet model: 'train', 'evaluate', and 'train_and_eval'.
#  When training in TPU, evaluation stage is not necessarily required, and
#  the model can be immediately tested after training is complete
# flags.DEFINE_string('mode', default='train_and_eval',
#                     help='One of train_and_eval", "train", "eval"}.')

# ToDo: it seems invoking the host_call during CPU-based training does not alter performance,
#  then compare invoking or not the host call just for TPU-based training


edalstm_features = {
    'hourly': tf.io.VarLenFeature(dtype=tf.float32),
    'target': tf.io.VarLenFeature(dtype=tf.float32),
    # ToDo: for one-hot encoders, replace float32 with intXX
    'oh_wd': tf.io.VarLenFeature(dtype=tf.float32),
    'oh_dh': tf.io.VarLenFeature(dtype=tf.float32),
    'timestamp': tf.io.VarLenFeature(dtype=tf.string)
}


def _parse_dataset_function(example_proto, read_features, objective_shapes, parse_timestamp):
    # parse the input tf.Example proto using the dictionary above
    row = tf.io.parse_single_example(serialized=example_proto,
                                     features=read_features)
    # pass objective shape as a list of lists [hourly_shape, daily_shape, weekly_shape]
    hourly = tf.reshape(row['hourly'].values, objective_shapes['hourly'])
    target = tf.reshape(row['target'].values, objective_shapes['target'])
    oh_wd = tf.reshape(row['oh_wd'].values, objective_shapes['oh_wd'])
    oh_dh = tf.reshape(row['oh_dh'].values, objective_shapes['oh_dh'])
    # do not parse the timestamp to TPUEstimator, as it does not support string types!
    # ToDo: code timestamps into features, as numbers
    #  so they can be parsed to training, if required
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
    # target as a float vector
    return feature_dict, target


# make_input_fh based on:
# https://medium.com/tensorflow/how-to-write-a-custom-estimator-model-for-the-cloud-tpu-7d8bd9068c26
# pass num_cores as argument to remove dependency
# from parameters['training'] in configuration file
def make_input_fn(tfrecord_path, num_train_rows, tensor_shapes, mode, num_cores):

    # from tsf_estimator definition (params = parameters)
    # important: TPUEstimator requires the argument params and cannot be renamed!
    def _input_fn(params):
        # remember, there is no parameter called batch_size in the parameters set,
        # whether the parameter set is built from command line flags or from a parameters dictionary.
        # The parameters effectively used are train_batch_size and eval_batch_size.
        # The TPUEstimator is responsible to switch between them as required!
        batch_size = params['batch_size']
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        # get the dataset from a TFRecord file
        dataset = tf.data.TFRecordDataset(tfrecord_path)

        if is_training:
            # shuffle the complete training dataset
            dataset = dataset.shuffle(num_train_rows)
            # repeat the dataset indefinitely
            dataset = dataset.repeat(count=None)

        dataset = dataset.map(
            lambda row: _parse_dataset_function(example_proto=row,
                                                read_features=edalstm_features,
                                                objective_shapes=tensor_shapes,
                                                parse_timestamp=False),
            num_parallel_calls=num_cores)

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

        # changed from Distributed TensorFlow CPU/GPU to single-device TPU
        # ToDo: return to Distributed TensorFlow TPU after successful implementation
        # ToDo: verify application of transposing, later...
        # ToDo: verify application of tf.contrib.data.parallel_interleave, later...

        # for TPU, prefetch data while training
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset

    return _input_fn


def set_make_input_fn_shapes(config_dict):
    train_eval_objective_shapes = {
        'hourly': [config_dict['embedding']['hourly'], 1],
        'target': [config_dict['no_targets'], 1],
        'oh_wd': [7, 1],  # Monday to Sunday
        'oh_dh': [24, 1],  # midnight to 23:00
        # ToDo: verify only the initial timestamp is passed to the model
        'timestamp': [config_dict['no_targets'], 1]
    }
    return train_eval_objective_shapes


def set_serving_input_fn_shapes(config_dict):
    serving_objective_shapes = {
        'hourly': [1, config_dict['embedding']['hourly'], 1],
        'target': [1, config_dict['no_targets'], 1],
        'oh_wd': [1, 7, 1],  # Monday to Sunday
        'oh_dh': [1, 24, 1],  # midnight to 23:00
        # ToDo: remember timestamp feature stores no_targets timestamps
        #   (for all components of the target vector),
        #   but only the initial timestamp is passed for serving
        'timestamp': [1, config_dict['no_targets'], 1]
    }
    return serving_objective_shapes

# # from GitHub TensorFlow Cloud TPU Resnet (official)
# def get_lr_schedule(train_steps, num_train_rows, train_batch_size):
#     """learning rate schedule."""
#     steps_per_epoch = np.floor(num_train_rows / train_batch_size)
#     train_epochs = train_steps / steps_per_epoch
#     return [  # (multiplier, epoch to start) tuples
#         (1.0, np.floor(5 / 90 * train_epochs)),
#         (0.1, np.floor(30 / 90 * train_epochs)),
#         (0.01, np.floor(60 / 90 * train_epochs)),
#         (0.001, np.floor(80 / 90 * train_epochs))
#     ]


# a new function that builds the learning rate schedule (a tuples list)
# from parameter values in the config_json_file
def get_lr_schedule_from_config_dict(config_dict):
    """learning rate schedule."""
    steps_per_epoch = np.floor(config_dict['num_train_rows'] / config_dict['train_batch_size'])
    train_epochs = config_dict['train_steps'] / steps_per_epoch
    lrs_tuples_list = [(item[0], np.floor(item[1] / config_dict['lrs_max_epochs'] * train_epochs))
                       for item in zip(config_dict['lrs_weights'], config_dict['lrs_steps'])]

    return lrs_tuples_list


# ToDo: replace params with lrs_params
def learning_rate_schedule(params, current_epoch):
    """Handles linear scaling rule, gradual warmup, and LR decay.
    The learning rate starts at 0, then it increases linearly per step.
    After 5 epochs we reach the base learning rate (scaled to account
      for batch size).
    After 30, 60 and 80 epochs the learning rate is divided by 10.
    After 90 epochs training stops and the LR is set to 0. This ensures
      that we train for exactly 90 epochs for reproducibility.
    Args:
      params: Python dict containing parameters for this run.
      current_epoch: `Tensor` for current epoch.
    Returns:
      A scaled `Tensor` for current learning rate.
    """
    # ToDo: change parameters['training']['learning_rate']
    #  to parameters['training']['base_learning_rate']
    # ToDo: verify params dictionary scope (it is outside the model function)
    # use 1.0 as base_learning_rate
    scaled_lr = params['base_learning_rate'] * (
            params['train_batch_size'] / 256.0)

    # lr_schedule = get_lr_schedule(
    #     # train_steps=params['train_steps'],
    #     train_steps=params['train_steps'],
    #     num_train_rows=params['num_train_rows'],
    #     train_batch_size=params['train_batch_size'])

    lr_schedule = get_lr_schedule_from_config_dict(params)

    decay_rate = (scaled_lr * lr_schedule[0][0] *
                  current_epoch / lr_schedule[0][1])

    for mult, start_epoch in lr_schedule:
        decay_rate = tf.where(current_epoch < start_epoch,
                              decay_rate, scaled_lr * mult)

    return decay_rate


# Follow the structure of the code proposed by Lakshmanan, then,
# remove main function and pass parameters to train_and_evaluate function
# ToDo: replace params with tsf_params
# time_series_forecaster is the model function (model_fn) of the TPUEstimator
def time_series_forecaster(features, labels, mode, params):
    # ToDo: global_step might be moved to TRAIN scope
    global_step = tf.train.get_global_step()

    # instantiate network topology from the corresponding class
    # forecaster_topology = EDALSTM()
    # call operator to forecaster_topology, over features
    # forecast = forecaster_topology(features)

    # add a conditional sentence to manage precision for Cloud TPU
    if params['precision'] == 'bfloat16':
        with tf.tpu.bfloat16_scope():
            forecaster_topology = EDALSTM()

            forecast = forecaster_topology(features=features,
                                           # all hyperparameters are passed to the model topology,
                                           # then the model uses only the ones it requires
                                           model_params=params)

        # cast result from bfloat16 to float32 again
        forecast = tf.cast(forecast, tf.float32)

    elif params['precision'] == 'float32':
        forecaster_topology = EDALSTM()

        forecast = forecaster_topology(features=features,
                                       # all hyperparameters are passed to the model topology,
                                       # then the model uses only the ones it requires
                                       model_params=params)

    # predictions are stored in a dictionary for further use at inference stage
    predictions = {
        "forecast": forecast
    }

    # 1. changed model function structure according to Gillard's architecture
    # 2. changed model function according to Lakshmanan's architecture
    # 3. changed model function according official ResNet TPUEstimator tutorial

    # Estimator in TRAIN or EVAL mode
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        # use labels and predictions to define training loss and evaluation loss
        # generate summaries for TensorBoard
        # with tf.name_scope('loss'):
        loss = tf.losses.mean_squared_error(labels=labels, predictions=forecast)
        # loss = tf.keras.losses.MeanSquaredError(labels, forecast)

        # for TPUEstimatorSpec, a metric function (which runs on CPU) is required
        # ToDo: return this metric to TensorBoard and logging, reference in
        #  http://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/tpu/TPUEstimatorSpec

        # ToDo: use different names in metric function parameters
        #   to avoid shadowing variables in outer scope
        def metric_fn(metric_fn_labels, metric_fn_predictions):
            # MSE is training_loss (as loss) is already returned for evaluation
            # then try a different metric as evaluation metric (previously val_loss)
            return {'rmse': tf.metrics.root_mean_squared_error(
                    labels=metric_fn_labels, predictions=metric_fn_predictions)}


        # eval_metrics as a list
        # eval_metrics = (metric_fn, [labels, forecast])
        # eval_metrics as a dictionary
        eval_metrics = (metric_fn, {
            # keys in dictionary must be consistent with metric_fn parameters!!!
            'metric_fn_labels': labels,
            'metric_fn_predictions': forecast})

        # this variable is only required for PREDICT mode, then
        prediction_hooks = None
        # pass host_call in TPUEstimatorSpec for summary activity on CPU
        # only in training mode, when params[skip_host_call] is false
        host_call = None

        # Estimator in TRAIN mode ONLY
        if mode == tf.estimator.ModeKeys.TRAIN:
            steps_per_epoch = params['num_train_rows'] / params['train_batch_size']
            current_epoch = (tf.cast(global_step, tf.float32) /
                             steps_per_epoch)

            learning_rate = learning_rate_schedule(params, current_epoch)

            # ToDo: test this optimizer that is used on ResNet 50 by TensorFlow team
            # optimizer = tf.train.MomentumOptimizer(
            #     learning_rate=learning_rate,
            #     momentum=params['momentum'],
            #     use_nesterov=True)

            # replace base learning rate from parameters dictionary
            # with learning rate from schedule
            # optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

            # change flow to wrap the optimizer for TPU
            if params['use_tpu']:
                # optimizer = tf.estimator.tpu.CrossShardOptimizer(optimizer)  # TPU change 1
                optimizer = tf.tpu.CrossShardOptimizer(optimizer)  # TPU change 1

            # This is needed for batch normalization, but has no effect otherwise
            # update_ops = tf.compat.v1.get_collection(key=tf.compat.v1.GraphKeys.UPDATE_OPS)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(control_inputs=update_ops):
                train_op = optimizer.minimize(loss, global_step)

            if not params['skip_host_call']:
                # start writing global step and loss only
                # def host_call_fn(gs, loss, lr, ce):
                # ToDo: replace loss with host_call_fn_loss
                def host_call_fn(gs, loss, lr, ce):
                    """Training host call. Creates scalar summaries for training metrics.
                    This function is executed on the CPU and should not directly reference
                    any Tensors in the rest of the `model_fn`. To pass Tensors from the
                    model to the `metric_fn`, provide as part of the `host_call`. See
                    https://www.tensorflow.org/api_docs/python/tf/estimator/tpu/TPUEstimatorSpec
                    for more information.
                    Arguments should match the list of `Tensor` objects passed as the second
                    element in the tuple passed to `host_call`.
                    Args:
                      gs: `Tensor with shape `[batch]` for the global_step
                      loss: `Tensor` with shape `[batch]` for the training loss.
                      lr: `Tensor` with shape `[batch]` for the learning_rate.
                      ce: `Tensor` with shape `[batch]` for the current_epoch.
                    Returns:
                      List of summary ops to run on the CPU host.
                    """
                    gs = gs[0]
                    # Host call fns are executed params['iterations_per_loop'] times after
                    # one TPU loop is finished, setting max_queue value to the same as
                    # number of iterations will make the summary writer only flush the data
                    # to storage once per loop.
                    with tf2.summary.create_file_writer(
                            # FLAGS.model_dir,
                            # model_dir is not passed via Flags anymore, then...
                            # get it from the copy of model_dir passed to the
                            # configuration dictionary in main()
                            params['model_dir'],
                            max_queue=params['iterations_per_loop']).as_default():
                        with tf2.summary.record_if(True):
                            tf2.summary.scalar('loss', loss[0], step=gs)
                            tf2.summary.scalar('learning_rate', lr[0], step=gs)
                            tf2.summary.scalar('current_epoch', ce[0], step=gs)

                        return tf.summary.all_v2_summary_ops()

                # To log the loss, current learning rate, and epoch for Tensorboard, the
                # summary op needs to be run on the host CPU via host_call. host_call
                # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
                # dimension. These Tensors are implicitly concatenated to
                # [params['batch_size']].
                gs_t = tf.reshape(global_step, [1])
                loss_t = tf.reshape(loss, [1])
                lr_t = tf.reshape(learning_rate, [1])
                ce_t = tf.reshape(current_epoch, [1])

                host_call = (host_call_fn, [gs_t, loss_t, lr_t, ce_t])
                # end of host_call

            predictions = None  # this is not required in TRAIN mode
            eval_metric_ops = None
            training_hooks = None
            # training_hooks = train_hook_list
            evaluation_hooks = None

        else:  # Estimator in EVAL mode ONLY
            # loss = loss
            train_op = None
            training_hooks = None
            # eval_metrics is already when TPU is used
            # eval_metric_ops = {'val_loss': val_loss}
            evaluation_hooks = None

    # Estimator in PREDICT mode ONLY
    else:
        loss = None
        train_op = None
        # eval_metric_ops = None
        eval_metrics = None
        training_hooks = None
        evaluation_hooks = None
        prediction_hooks = None  # this might change as we are in PREDICT mode
        # pass host_call in TPUEstimatorSpec for summary activity on CPU, when training
        host_call = None

    return tf.estimator.tpu.TPUEstimatorSpec(  # TPU change 2
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        # TPUEstimatorSpec.eval_metrics is a tuple of metrics_fn and tensors
        eval_metrics=eval_metrics,
        # ToDo: do I need to use export_outputs?
        # export_outputs=not_used_yet (for TensorFlow Serving, redirected from predictions if omitted)
        host_call=host_call,
        # scaffold_fn=not_used_yet
        # ToDo: verify use of training_hooks
        # temporarily disable training hooks
        training_hooks=training_hooks,
        evaluation_hooks=evaluation_hooks,
        prediction_hooks=prediction_hooks
    )


def load_global_step_from_checkpoint_dir(checkpoint_dir):
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(
            tf.train.latest_checkpoint(checkpoint_dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0


# TPUEstimator does not have a train_and_evaluate method
# then it has to be rolled up, as in Lakshmanan, 2018
# https://medium.com/tensorflow/how-to-write-a-custom-estimator-model-for-the-cloud-tpu-7d8bd9068c26
# Lakshmanan defines the custom TPUEstimator inside the train_and_evaluate function
def train_and_evaluate(model_dir, config_dict):

    tf.summary.FileWriterCache.clear()  # ensure file writer cache is clear for TensorBoard events file
    iterations_per_loop = config_dict['iterations_per_loop']
    train_steps = config_dict['train_steps']
    # ToDo: verify if eval_batch_size should equal train_batch_size as batch_size
    eval_batch_size = config_dict['eval_batch_size']
    eval_batch_size = eval_batch_size - eval_batch_size % config_dict['num_cores']  # divisible by num_cores
    # at the beginning of training stage, report batch sizes and training steps
    tf.logging.info('train_batch_size=%d  eval_batch_size=%d  train_steps=%d',
                    config_dict['train_batch_size'],
                    eval_batch_size,
                    train_steps)

    # TPU change 3
    if config_dict['use_tpu']:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            config_dict['tpu'],
            zone=config_dict['tpu_zone'],
            project=config_dict['project'])
        config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=model_dir,
            save_checkpoints_steps=iterations_per_loop,
            save_summary_steps=config_dict['save_summary_steps'],
            log_step_count_steps=config_dict['log_step_count_steps'],
            keep_checkpoint_max=config_dict['keep_checkpoint_max'],
            tpu_config=tf.estimator.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                per_host_input_for_training=True))
    else:
        # CPU-based execution
        config = tf.estimator.tpu.RunConfig(
            save_summary_steps=config_dict['save_summary_steps'],
            log_step_count_steps=config_dict['log_step_count_steps']
        )

    # instantiate base estimator class for custom model function
    # tsf_estimator = tf.estimator.Estimator(
    tsf_estimator = tf.estimator.tpu.TPUEstimator(  # TPU change 4
        model_fn=time_series_forecaster,
        config=config,
        # ToDo: confirm!!!
        #   params argument for TPUEstimator identifies an optional dictionary 'params_dict'
        params=config_dict,
        model_dir=model_dir,
        train_batch_size=config_dict['train_batch_size'],
        eval_batch_size=config_dict['eval_batch_size'],
        use_tpu=config_dict['use_tpu'])

    # set up training and evaluation in a loop
    # ToDo: at the beginning, use a single-device TPU, TF is not distributed on multiple nodes

    # train dataset is train.tfrecord file in the SLDB path
    train_data_path = '{}/train.tfrecord'.format(config_dict['data_dir'])
    train_input_fn = make_input_fn(tfrecord_path=train_data_path,
                                   num_train_rows=config_dict['num_train_rows'],
                                   tensor_shapes=set_make_input_fn_shapes(
                                       config_dict=config_dict),
                                   mode=tf.estimator.ModeKeys.TRAIN,
                                   num_cores=config_dict['num_cores'])

    # eval dataset is eval.tfrecord file in the SLDB path
    eval_data_path = '{}/eval.tfrecord'.format(config_dict['data_dir'])
    eval_input_fn = make_input_fn(tfrecord_path=eval_data_path,
                                  # num_train_rows is now, in fact, num_eval_rows
                                  num_train_rows=config_dict['num_eval_rows'],
                                  tensor_shapes=set_make_input_fn_shapes(
                                      config_dict=config_dict),
                                  mode=tf.estimator.ModeKeys.EVAL,
                                  num_cores=config_dict['num_cores'])

    # load last checkpoint and start from there
    current_step = load_global_step_from_checkpoint_dir(model_dir)
    # add the number of training rows in the SLDB to evaluate steps per epoch
    # ToDo: set the value for num_train_rows key
    # at TPUEstimator initialization, report session steps, session epochs, and current step
    steps_per_epoch = config_dict['num_train_rows'] // config_dict['train_batch_size']
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current step %d.',
                    train_steps,
                    train_steps / steps_per_epoch,
                    current_step)

    # use time performance counter
    # start_timestamp = time.time()  # This time will include compilation time
    start_timestamp = time.perf_counter()

    while current_step < config_dict['train_steps']:
        # train for up to iterations_per_loop number of steps.
        # at the end of training, a checkpoint will be written to --model_dir
        next_checkpoint = min(current_step + iterations_per_loop, train_steps)
        # cast next_checkpoint to int to avoid TPU error
        tsf_estimator.train(input_fn=train_input_fn, max_steps=int(next_checkpoint))
        current_step = next_checkpoint
        # at checkpoint writing, report training extent and elapsed time
        tf.logging.info('Finished training up to step %d. Elapsed seconds %.4f.',
                        # elapsed_time to float value
                        # next_checkpoint, int(time.time() - start_timestamp))
                        next_checkpoint, time.perf_counter() - start_timestamp)

        # evaluate the model after checkpoint writing
        # evaluate the model on the most recent model in --model_dir.
        # since evaluation happens in batches of --eval_batch_size, some SLDB-rows
        # may be excluded modulo the batch size, as long as the batch size is
        # consistent, the evaluated rows are also consistent.
        # ToDo: put the next three commands inside a conditional sentence
        #  to execute them only in 'train_and_eval' mode
        #  so they can be avoided in 'train' mode to speed up TPU-based training
        if config_dict['mode'] == 'train_and_eval':
            tf.logging.info('Starting to evaluate at step %d', next_checkpoint)
            eval_results = tsf_estimator.evaluate(
                input_fn=eval_input_fn,
                # ToDo: set the value for num_eval_rows key
                steps=config_dict['num_eval_rows'] // eval_batch_size)
            tf.logging.info('Eval results at step %d: %s', next_checkpoint, eval_results)

    if config_dict['mode'] == 'train_and_eval':
        # elapsed_time to float value
        # elapsed_time = int(time.time() - start_timestamp)
        elapsed_time = time.perf_counter() - start_timestamp
        tf.logging.info('Finished training and evaluation up to step %d. Elapsed seconds %.4f.',
                        train_steps,
                        elapsed_time)

    # serving_input_receiver_fn is not callable,
    # therefore no arguments can be passed from train_and_evaluate to serving_input_fn,
    # then serving_input_receiver_fn is defined inside train_and_evaluate function
    def serving_input_fn():
        # TPU are not optimized for serving, so it is assumed the predictions server is CPU or GPU-based
        # inputs is equivalent to example protocol buffers
        feature_placeholders = {'example_bytes': tf.placeholder(tf.string, shape=())}

        # the serving input function does not require the label
        # the underscore discards the target (not required for serving predictions)
        features, _ = _parse_dataset_function(example_proto=feature_placeholders['example_bytes'],
                                              read_features=edalstm_features,
                                              objective_shapes=set_serving_input_fn_shapes(
                                                  config_dict=config_dict),
                                              # parse timestamps to use them in plotting results
                                              parse_timestamp=True)

        # re-shape to original model spec
        return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)


    # export similar to Cloud ML Engine convention
    tf.logging.info('Starting to export model.')
    tsf_estimator.export_saved_model(
        export_dir_base=os.path.join(model_dir, 'export/exporter'),
        serving_input_receiver_fn=serving_input_fn)


def main(unused_argv):
    # how to override parameter values in the configuration file with Abseil Flags
    # if FLAGS.model_dir:
    #     DMSLSTM_CFG['model_dir'] = FLAGS.model_dir

    storage_client = storage.Client()
    bucket = storage_client.get_bucket('cbidmltsf')

    # build a blob with the configuration file
    blob = bucket.blob('parameters/{}'.format(FLAGS.config_json_file))

    # download the file as string and decode it
    json_data_string = blob.download_as_string().decode('utf8')

    # load the parameters dictionary from the blob
    configuration = json.loads(json_data_string)

    # get the experiment identifier
    experiment_id = FLAGS.config_json_file.replace('.json', '')
    # use experiment identifier and execution number to build the model directory
    model_dir = 'gs://cbidmltsf/models/{}_{:02d}'.format(experiment_id,
                                                         FLAGS.execution)

    # temporarily pass model_dir to the configuration dictionary,
    # so it can be parsed to host_call_fn when skip_host_call is False;
    # this value is lost for metadata, but is not required
    # as it can be fetched from JSON configuration file
    configuration['model_dir'] = model_dir

    # model_dir value is now both at FLAGS.model_dir and parameters['training']['model_dir]
    # get it from the parameter dictionary for consistency
    # ToDo: take model_dir out of the configuration dictionary
    train_and_evaluate(model_dir, config_dict=configuration)


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # next line is suggested to avoid duplicate posts in logging
    # tf.logging._logger.propagate = False
    tf.disable_v2_behavior()
    app.run(main)
