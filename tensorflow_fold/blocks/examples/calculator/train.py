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

r"""Runs the trainer for the calculator example.

This file is a minor modification to loom/calculator_example/train.py.
To run, first make the data set:

  ./tensorflow_fold/loom/calculator_example/make_dataset \
    --output_path=DIR/calc_data.dat

Then run the trainer:

  ./tensorflow_fold/blocks/examples/calculator/train \
    --train_data_path=DIR/calc_data.dat
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
# import google3
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks.examples.calculator import model
from tensorflow_fold.loom.calculator_example import calculator_pb2
from tensorflow_fold.util import proto_tools

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
tf.flags.DEFINE_string('logdir', '/tmp/calculator_example',
                       'Directory in which to write event logs.')
tf.flags.DEFINE_string('master', '',
                       'Tensorflow master to use.')
tf.flags.DEFINE_integer('task', 0,
                        'Task ID of the replica running the training.')
tf.flags.DEFINE_integer('ps_tasks', 0,
                        'Number of PS tasks in the job.')
FLAGS = tf.flags.FLAGS


# Find the root of the bazel repository.
def source_root():
  root = __file__
  for _ in xrange(5):
    root = os.path.dirname(root)
  return root

CALCULATOR_SOURCE_ROOT = source_root()
CALCULATOR_PROTO_FILE = ('tensorflow_fold/loom/'
                         'calculator_example/calculator.proto')
CALCULATOR_EXPRESSION_PROTO = ('tensorflow_fold.loom.'
                               'calculator_example.CalculatorExpression')


# Make sure serialized_message_to_tree can find the calculator example proto:
proto_tools.map_proto_source_tree_path('', CALCULATOR_SOURCE_ROOT)
proto_tools.import_proto_file(CALCULATOR_PROTO_FILE)


def iterate_over_tf_record_protos(table_path, unused_message_type):
  while True:
    for v in tf.python_io.tf_record_iterator(table_path):
      yield proto_tools.serialized_message_to_tree(
          CALCULATOR_EXPRESSION_PROTO, v)


def emit_values(supervisor, session, step, values):
  summary = tf.Summary()
  for name, value in six.iteritems(values):
    summary_value = summary.value.add()
    summary_value.tag = name
    summary_value.simple_value = float(value)
  supervisor.summary_computed(session, summary, global_step=step)


def main(unused_argv):
  train_iterator = iterate_over_tf_record_protos(
      FLAGS.train_data_path, calculator_pb2.CalculatorExpression)

  with tf.Graph().as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):

      # Build the graph.
      classifier = model.CalculatorModel(FLAGS.embedding_length)
      loss = classifier.loss
      accuracy = classifier.accuracy
      train_op = classifier.train_op
      global_step = classifier.global_step

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
        fdict = classifier.build_feed_dict(batch)

        _, step, loss_v, accuracy_v = sess.run(
            [train_op, global_step, loss, accuracy],
            feed_dict=fdict)
        print('step=%d: loss=%f accuracy=%f' % (step, loss_v, accuracy_v))
        emit_values(supervisor, sess, step,
                    {'Batch Loss': loss_v,
                     'Batch Accuracy': accuracy_v})

if __name__ == '__main__':
  tf.app.run()
