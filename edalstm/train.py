import os
import time

import json

import numpy as np

from absl import flags
from absl import app

from google.cloud import storage

# ToDo: pass this code to the setup.py file of the final module!!!
# just a temporary workaround
# import sys
# _ROOT_DIR = '{0}/gcp/cbidmltsf'.format(os.getenv("HOME"))
# sys.path.append(_ROOT_DIR)
# temporarily solved: added dplstm root dir to PYTHONPATH by exporting in .bashrc!!!

from configs.edalstm_config import training_parameters
# import sldb_parameters and architecture_parameters, only to persist them in stats/ directory
from configs.edalstm_config import sldb_parameters, architecture_parameters

from configs import common_hparams_flags
from configs import common_tpu_params

from edalstm.data import make_input_fn, serving_input_fn

from edalstm.model import EDALSTM

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2

common_hparams_flags.define_common_hparams_flags()
common_tpu_params.define_common_tpu_flags()

FLAGS = flags.FLAGS

# define specific hyper-parameter flags
flags.DEFINE_float('base_learning_rate', default=None,
                   help=('Base learning rate'
                         'when train batch size is 256.'))

flags.DEFINE_integer('save_summary_steps', default=None,
                     help=('The number of steps at  which the Estimator RunConfig'
                           'saves summaries of loss (scalar) to TensorBoard.'
                           'Remember these summaries are independent'
                           'from those declared in the host_call.'))

flags.DEFINE_integer('log_step_count_steps', default=None,
                     help=('The number of steps at which the global step per second'
                           'and the examples per second are written to logging (stdout)'
                           'and to TensorBoard summaries.'
                           'Remember these summaries are independent'
                           'from those declared in the host_call.'))

# ToDo: break train_and_evaluate function into the three execution modes
#  included in the official ResNet model: 'train', 'evaluate', and 'train_and_eval'.
#  When training in TPU, evaluation stage is not necessarily required, and
#  the model can be immediately tested after training is complete
flags.DEFINE_string('mode', default='train_and_eval',
                    help=('One of'
                          '{"train_and_eval", "train", "eval"}.'))

flags.DEFINE_bool('persist_parameters', default=None,
                  help=('Indicates whether saving parameter json files to parameters/'
                        'or not.'))

# ToDo: it seems invoking the host_call during CPU-based training does not alter performance,
#  then compare invoking or not the host call just for TPU-based training


# from GitHub TensorFlow Cloud TPU Resnet official
def get_lr_schedule(train_steps, num_train_rows, train_batch_size):
    # """learning rate schedule."""
    # steps_per_epoch = np.floor(num_train_rows / train_batch_size)
    # train_epochs = train_steps / steps_per_epoch
    # return [  # (multiplier, epoch to start) tuples
    #     (1.0, np.floor(5 / 90 * train_epochs)),
    #     (0.1, np.floor(30 / 90 * train_epochs)),
    #     (0.01, np.floor(60 / 90 * train_epochs)),
    #     (0.001, np.floor(80 / 90 * train_epochs))
    # ]
    return training_parameters['lrs']


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
    # ToDo: change training_parameters['learning_rate']
    #  to training_parameters['base_learning_rate']
    # ToDo: verify params dictionary scope (it is outside the model function)
    # use 1.0 as base_learning_rate
    scaled_lr = params['base_learning_rate'] * (
            params['train_batch_size'] / 256.0)

    lr_schedule = get_lr_schedule(
        # train_steps=params['train_steps'],
        train_steps=params['train_steps'],
        num_train_rows=params['num_train_rows'],
        train_batch_size=params['train_batch_size'])
    decay_rate = (scaled_lr * lr_schedule[0][0] *
                  current_epoch / lr_schedule[0][1])
    for mult, start_epoch in lr_schedule:
        decay_rate = tf.where(current_epoch < start_epoch,
                              decay_rate, scaled_lr * mult)
    return decay_rate


# Follow the structure of the code proposed by Lakshmanan, then,
# remove main function and pass parameters to train_and_evaluate function
def time_series_forecaster(features, labels, mode, params):
    # instantiate network topology from the corresponding class
    # forecaster_topology = DMSLSTM()

    # ToDo: global_step might be moved to TRAIN scope
    global_step = tf.train.get_global_step()

    # call operator to forecaster_topology, over features
    # forecast = forecaster_topology(features)

    # add a conditional sentence to manage precision for Cloud TPU
    if params['precision'] == 'bfloat16':
        with tf.tpu.bfloat16_scope():
            forecaster_topology = EDALSTM()
            forecast = forecaster_topology(features)
        forecast = tf.cast(forecast, tf.float32)
    elif params['precision'] == 'float32':
        forecaster_topology = EDALSTM()
        forecast = forecaster_topology(features)

    # predictions are stored in a dictionary for further use at inference stage
    predictions = {
        "forecast": forecast
    }

    # CHANGE MODEL FUNCTION STRUCTURE ACCORDING TO GILLARD'S ARCHITECTURE
    # CHANGE IT AGAIN, ACCORDING TO TPU ESTIMATOR TUTORIAL FROM TENSORFLOW AND LAKSHMANAN
    # ToDo: change model function again, now accordingly to TPUEstimator API usage
    #   as in ResNet training demo (tensorflow/tpu)

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

        # use different names in metric function parameters to avoid shadowing variables in outer scope
        def metric_fn(forecast, labels):
            # MSE is training_loss (as loss) is already returned for evaluation
            # then try a different metric as evaluation metric (previously val_loss)
            return {'rmse': tf.metrics.root_mean_squared_error(
                    labels=labels, predictions=forecast)}

        # eval_metrics as a list
        # eval_metrics = (metric_fn, [labels, forecast])
        # eval_metrics as a dictionary
        eval_metrics = (metric_fn, {
            # keys in dictionary must be consistent with metric_fn parameters!!!
            'labels': labels,
            'forecast': forecast})

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
                            # at this time, params['model_dir'] has been already set to FLAGS.model_dir
                            # FLAGS.model_dir,
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

                # start writing global step and loss only
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
def train_and_evaluate(model_dir, parameters):
    tf.summary.FileWriterCache.clear()  # ensure file writer cache is clear for TensorBoard events file
    iterations_per_loop = parameters['iterations_per_loop']
    train_steps = parameters['train_steps']
    # change original for the time series forecasting case
    # eval_batch_size = min(1024, hparams['num_eval_images'])
    # ToDo: verify if eval_batch_size should equal train_batch_size as batch_size
    eval_batch_size = parameters['eval_batch_size']
    eval_batch_size = eval_batch_size - eval_batch_size % parameters['num_cores']  # divisible by num_cores
    # at the beginning of training stage, report batch sizes and training steps
    tf.logging.info('train_batch_size=%d  eval_batch_size=%d  train_steps=%d',
                    parameters['train_batch_size'],
                    eval_batch_size,
                    train_steps)

    # TPU change 3
    if parameters['use_tpu']:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            parameters['tpu'],
            zone=parameters['tpu_zone'],
            project=parameters['project'])
        config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=model_dir,
            save_checkpoints_steps=iterations_per_loop,
            # ToDo: put next two lines outside of the conditional sentence
            save_summary_steps=FLAGS.save_summary_steps,
            log_step_count_steps=FLAGS.log_step_count_steps,
            keep_checkpoint_max=parameters['keep_checkpoint_max'],
            tpu_config=tf.estimator.tpu.TPUConfig(
                iterations_per_loop=iterations_per_loop,
                per_host_input_for_training=True))
    else:
        # CPU-based execution
        config = tf.estimator.tpu.RunConfig(
            # ToDo: put next two lines outside of the conditional sentence
            save_summary_steps=FLAGS.save_summary_steps,
            log_step_count_steps=FLAGS.log_step_count_steps
        )

    # instantiate base estimator class for custom model function
    # tsf_estimator = tf.estimator.Estimator(
    tsf_estimator = tf.estimator.tpu.TPUEstimator(  # TPU change 4
        model_fn=time_series_forecaster,
        config=config,
        # params argument for TPUEstimator identifies an optional dictionary
        params=parameters,
        model_dir=model_dir,
        train_batch_size=parameters['train_batch_size'],
        eval_batch_size=parameters['eval_batch_size'],
        use_tpu=parameters['use_tpu'])

    # set up training and evaluation in a loop
    # ToDo: at the beginning, use a single-device TPU, TF is not distributed

    # train dataset is train.tfrecord file in the SLDB path
    train_data_path = '{}/train.tfrecord'.format(parameters['data_dir'])
    train_input_fn = make_input_fn(train_data_path, mode=tf.estimator.ModeKeys.TRAIN)

    # eval dataset is eval.tfrecord file in the SLDB path
    eval_data_path = '{}/eval.tfrecord'.format(parameters['data_dir'])
    eval_input_fn = make_input_fn(eval_data_path, mode=tf.estimator.ModeKeys.EVAL)

    # load last checkpoint and start from there
    current_step = load_global_step_from_checkpoint_dir(model_dir)
    # add the number of training rows in the SLDB to evaluate steps per epoch
    # ToDo: set the value for num_train_rows key
    # at TPUEstimator initialization, report session steps, session epochs, and current step
    steps_per_epoch = parameters['num_train_rows'] // parameters['train_batch_size']
    tf.logging.info('Training for %d steps (%.2f epochs in total). Current step %d.',
                    train_steps,
                    train_steps / steps_per_epoch,
                    current_step)

    # use time performance counter
    # start_timestamp = time.time()  # This time will include compilation time
    start_timestamp = time.perf_counter()

    while current_step < parameters['train_steps']:
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
        if FLAGS.mode == 'train_and_eval':
            tf.logging.info('Starting to evaluate at step %d', next_checkpoint)
            eval_results = tsf_estimator.evaluate(
                input_fn=eval_input_fn,
                # ToDo: set the value for num_eval_rows key
                steps=training_parameters['num_eval_rows'] // eval_batch_size)
            tf.logging.info('Eval results at step %d: %s', next_checkpoint, eval_results)

    if FLAGS.mode == 'train_and_eval':
        # elapsed_time to float value
        # elapsed_time = int(time.time() - start_timestamp)
        elapsed_time = time.perf_counter() - start_timestamp
        tf.logging.info('Finished training and evaluation up to step %d. Elapsed seconds %.4f.',
                        train_steps,
                        elapsed_time)

    # export similar to Cloud ML Engine convention
    tf.logging.info('Starting to export model.')
    tsf_estimator.export_saved_model(
        export_dir_base=os.path.join(model_dir, 'export/exporter'),
        serving_input_receiver_fn=serving_input_fn)


def main(unused_argv):
    # ToDo: pass the following code to a function
    # override parameter values in configuration file with arguments parsed by Abseil Flags
    # the following parameters are mandatory to pass via flags
    training_parameters['model_dir'] = FLAGS.model_dir
    training_parameters['data_dir'] = FLAGS.data_dir
    training_parameters['use_tpu'] = FLAGS.use_tpu

    # the following parameters are defined in configuration file
    # but can be overridden via flags
    # ToDo: automate the process for all of the flags passed as arguments
    if FLAGS.train_steps:
        training_parameters['train_steps'] = FLAGS.train_steps
    if FLAGS.train_batch_size:
        training_parameters['train_batch_size'] = FLAGS.train_batch_size
    if FLAGS.base_learning_rate:
        training_parameters['base_learning_rate'] = FLAGS.base_learning_rate
    if FLAGS.skip_host_call:
        training_parameters['skip_host_call'] = FLAGS.skip_host_call
    if FLAGS.iterations_per_loop:
        training_parameters['iterations_per_loop'] = FLAGS.iterations_per_loop
    if FLAGS.precision:
        training_parameters['precision'] = FLAGS.precision

    # model_dir value is now both at FLAGS.model_dir and training_parameters['model_dir]
    # get it from the parameter dictionary for consistency
    train_and_evaluate(training_parameters['model_dir'], training_parameters)

    # ToDo: persist the configuration dictionaries not as the config Python file
    #  but as a JSON file with the sldb, architecture, and training parameter dictionaries
    #  (at this point, even training_parameters is already complete)

    # finally, persist the DMSLSTM configuration file for further analysis
    # ToDo: pass the following code to a function save_config_file_to_gs

    # persist hyperparameters only if the appropriate flag is on
    if FLAGS.persist_parameters:
        # persist json file to vm block storage
        with open('temp/sldb_parameters.json', 'w') as outfile:
            json.dump(sldb_parameters, outfile, indent=4)
        with open('temp/architecture_parameters.json', 'w') as outfile:
            json.dump(architecture_parameters, outfile, indent=4)
        with open('temp/training_parameters.json', 'w') as outfile:
            json.dump(training_parameters, outfile, indent=4)

        # a client to Google Storage
        storage_client = storage.Client()
        # a reference to the project's bucket
        bucket = storage_client.get_bucket('cbidmltsf')
        # build the path to Google Storage
        # get the process identifier from the model_dir string
        process_id = training_parameters['model_dir'].replace('gs://cbidmltsf/models/', '')
        # use the process identifier to build the model folder in stats/
        # update: avoid saving hyperparameters for each execution
        # and save them for experiment only, then
        # use the model identifier to build the model folder in parameters/
        # just cancel the suffix for the experiment id
        model_id = process_id[:-3]

        # build path to Google Storage, and upload parameters dictionary from VM's block storage
        sldb_parameters_dict_path = 'parameters/{}/sldb_parameters.json'.format(model_id)
        blob = bucket.blob(sldb_parameters_dict_path)
        blob.upload_from_filename('temp/sldb_parameters.json')

        # build path to Google Storage, and upload parameters dictionary from VM's block storage
        architecture_parameters_dict_path = 'parameters/{}/architecture_parameters.json'.format(model_id)
        blob = bucket.blob(architecture_parameters_dict_path)
        blob.upload_from_filename('temp/architecture_parameters.json')

        # build path to Google Storage, and upload parameters dictionary from VM's block storage
        training_parameters_dict_path = 'parameters/{}/training_parameters.json'.format(model_id)
        blob = bucket.blob(training_parameters_dict_path)
        blob.upload_from_filename('temp/training_parameters.json')


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # next line is suggested to avoid duplicate posts in logging
    # tf.logging._logger.propagate = False
    tf.disable_v2_behavior()
    app.run(main)
