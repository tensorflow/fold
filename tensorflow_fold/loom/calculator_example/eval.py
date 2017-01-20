# Copyright 2017 Google Inc. All Rights Reserved.
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

"""Runs the evaluator for the calculator smoke test."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

# import google3
import tensorflow as tf
from tensorflow_fold.loom.calculator_example import calculator_pb2
from tensorflow_fold.loom.calculator_example import helpers
from tensorflow_fold.loom.calculator_example import model


tf.flags.DEFINE_string(
    'validation_data_path',
    '',
    'TF Record containing the validation dataset of expressions.')
tf.flags.DEFINE_integer(
    'embedding_length', 5,
    'How long to make the expression embedding vectors.')
tf.flags.DEFINE_string(
    'eval_master', '',
    'Tensorflow master to use.')
tf.flags.DEFINE_string(
    'logdir', '/tmp/calculator_smoketest',
    'Directory where we read models and write event logs.')
tf.flags.DEFINE_integer(
    'eval_interval_secs', 60,
    'Time interval between eval runs. Zero to do a single eval then exit.')
FLAGS = tf.flags.FLAGS


def string_to_expression(string):
  expression = calculator_pb2.CalculatorExpression()
  expression.ParseFromString(string)
  return expression


def main(unused_argv):
  validation_table = tf.python_io.tf_record_iterator(FLAGS.validation_data_path)
  print('Reading validation table...')
  validation_data = [string_to_expression(v) for v in validation_table]
  print('Done reading validation table...')

  with tf.Graph().as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    classifier = model.CalculatorSignClassifier(FLAGS.embedding_length)

    loss = classifier.loss()
    accuracy = classifier.accuracy()

    saver = tf.train.Saver()
    supervisor = tf.train.Supervisor(
        logdir=FLAGS.logdir,
        recovery_wait_secs=FLAGS.eval_interval_secs)
    sess = supervisor.PrepareSession(
        FLAGS.eval_master,
        wait_for_checkpoint=True,
        start_standard_services=False)

    while not supervisor.ShouldStop():
      ckpt = tf.train.get_checkpoint_state(FLAGS.logdir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      else:
        continue
      step, validation_loss, validation_accuracy = sess.run(
          [global_step, loss, accuracy],
          feed_dict=classifier.build_feed_dict(validation_data))
      print('Step %d:  loss=%f accuracy=%f' % (
          step, validation_loss, validation_accuracy))
      helpers.EmitValues(supervisor, sess, step,
                         {'Validation Loss': validation_loss,
                          'Validation Accuracy': validation_accuracy})
      if not FLAGS.eval_interval_secs: break
      time.sleep(FLAGS.eval_interval_secs)

    supervisor.Stop()


if __name__ == '__main__':
  tf.app.run()
