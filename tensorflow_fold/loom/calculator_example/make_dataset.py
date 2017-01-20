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

"""make_dataset randomly constructs datsets for the calculator smoketest.

The datasets are in TF Record format.  The contents of the TF record are
CalculatorExpression protos.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import google3
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.loom.calculator_example import calculator

tf.flags.DEFINE_string('output_path', '',
                       'Where to write the TFRecord file with the expressions.')
tf.flags.DEFINE_integer('max_expression_depth', 5,
                        'Maximum expression depth.')
tf.flags.DEFINE_integer('num_samples', 1000,
                        'How many samples to put into the table.')
FLAGS = tf.flags.FLAGS


def make_expression():
  expression = calculator.random_expression(FLAGS.max_expression_depth)
  expression.result = calculator.evaluate_expression(expression)
  return expression.SerializeToString()


def main(unused_argv):
  record_output = tf.python_io.TFRecordWriter(FLAGS.output_path)
  for _ in xrange(FLAGS.num_samples):
    record_output.write(make_expression())
  record_output.close()


if __name__ == '__main__':
  tf.app.run()
