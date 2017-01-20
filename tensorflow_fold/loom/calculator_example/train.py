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

"""Runs the trainer for the calculator smoketest."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import google3
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.loom.calculator_example import calculator_pb2
from tensorflow_fold.loom.calculator_example import helpers
from tensorflow_fold.loom.calculator_example import model


tf.flags.DEFINE_string(
    'train_data_path', '',
    'TF Record file containing the training dataset of expressions.')
tf.flags.DEFINE_integer(
    'batch_size', 1000, 'How many samples to read per batch.')
tf.flags.DEFINE_integer(
    'embedding_length', 5,
    'How long to make the expression embedding vectors.')
tf.flags.DEFINE_integer(
    'max_steps', 1000000,
    'The maximum number of batches to run the trainer for.')

# Replication flags:
tf.flags.DEFINE_string('logdir', '/tmp/calculator_smoketest',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS


def iterate_over_tf_record_protos(table_path, message_type):
  while True:
    for v in tf.python_io.tf_record_iterator(table_path):
      message = message_type()
      message.ParseFromString(v)
      yield message


def main(unused_argv):
  train_iterator = iterate_over_tf_record_protos(
      FLAGS.train_data_path, calculator_pb2.CalculatorExpression)

  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):

      # Build the graph.
      global_step = tf.Variable(0, name='global_step', trainable=False)
      classifier = model.CalculatorSignClassifier(FLAGS.embedding_length)

      variables = classifier.variables()
      loss = classifier.loss()
      accuracy = classifier.accuracy()

      optr = tf.train.GradientDescentOptimizer(0.01)
      trainer = optr.minimize(loss, global_step=global_step, var_list=variables)

      # Set up the supervisor.
      supervisor = tf.train.Supervisor(
          logdir=FLAGS.logdir,
          is_chief=(FLAGS.task == 0),
          save_summaries_secs=10,
          save_model_secs=30)
      sess = supervisor.PrepareSession(FLAGS.master)

      # Run the trainer.
      for _ in xrange(FLAGS.max_steps):
        batch = [next(train_iterator) for _ in xrange(FLAGS.batch_size)]

        _, step, batch_loss, batch_accuracy = sess.run(
            [trainer, global_step, loss, accuracy],
            feed_dict=classifier.build_feed_dict(batch))
        print('step=%d:  batch loss=%f accuracy=%f' % (
            step, batch_loss, batch_accuracy))
        helpers.EmitValues(supervisor, sess, step,
                           {'Batch Loss': batch_loss,
                            'Batch Accuracy': batch_accuracy})

if __name__ == '__main__':
  tf.app.run()
