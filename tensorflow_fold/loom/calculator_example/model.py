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

"""A Loom Model for the calculator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import google3
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_fold.loom.calculator_example import calculator
from tensorflow_fold.loom.calculator_example import calculator_pb2
from tensorflow_fold.public import loom


class CombineLoomOp(loom.LoomOp):
  """A LoomOp for combining vectors for use in recursive neural nets."""

  def __init__(self, num_args, embedding_length, weights):
    """A LoomOp for recursive neural nets.

    Args:
      num_args: the number of inputs being fused.
      embedding_length: the length of the input and output vectors.
      weights: a (num_args * embedding_length) x embedding_length matrix.
    """
    self._num_args = num_args
    self._embedding_length = embedding_length
    self._type_shape = loom.TypeShape('float32', (embedding_length,))
    self._weights = weights
    self._input_type_shapes = [self._type_shape] * self._num_args
    self._output_type_shapes = [self._type_shape]

  def instantiate_batch(self, inputs):
    return [tf.nn.relu(tf.matmul(tf.concat(inputs, 1), self._weights))]


class CalculatorLoom(object):
  """Wraps a Loom so it can accept CalculatorExpressions."""

  def __init__(self, embedding_length):
    self._embedding_length = embedding_length
    self._named_tensors = {}

    for n in xrange(10):
      # Note: the examples only have the numbers 0 through 9 as terminal nodes.
      name = 'terminal_' + str(n)
      self._named_tensors[name] = tf.Variable(
          tf.truncated_normal([embedding_length],
                              dtype=tf.float32,
                              stddev=1),
          name=name)

    self._combiner_weights = {}
    self._loom_ops = {}
    for name in calculator_pb2.CalculatorExpression.OpCode.keys():
      weights_var = tf.Variable(
          tf.truncated_normal([2 * embedding_length, embedding_length],
                              dtype=tf.float32,
                              stddev=1),
          name=name)
      self._combiner_weights[name] = weights_var
      self._loom_ops[name] = CombineLoomOp(2, embedding_length, weights_var)

    self._loom = loom.Loom(
        named_tensors=self._named_tensors,
        named_ops=self._loom_ops)

    self._output = self._loom.output_tensor(
        loom.TypeShape('float32', [embedding_length]))

  def _build_expression(self, weaver, expression):
    if expression.HasField('number'):
      return weaver.named_tensor('terminal_' + str(expression.number))

    left_expression = self._build_expression(weaver, expression.left)
    right_expression = self._build_expression(weaver, expression.right)
    op_name = calculator_pb2.CalculatorExpression.OpCode.Name(expression.op)
    return weaver.op(op_name, [left_expression, right_expression])[0]

  def build_feed_dict(self, expression_list):
    for e in expression_list:
      calculator.validate_expression(e)

    weaver = self._loom.make_weaver()
    roots = [self._build_expression(weaver, e) for e in expression_list]
    return weaver.build_feed_dict(roots)

  def output(self):
    return self._output

  def variables(self):
    return (list(self._named_tensors.values()) +
            list(self._combiner_weights.values()))


class CalculatorSignClassifier(object):
  """A CalculatorLoom with a three-way classifier on its output."""

  def __init__(self, embedding_length):
    self._calculator_loom = CalculatorLoom(embedding_length)

    self._labels_placeholder = tf.placeholder(tf.float32)
    self._classifier_weights = tf.Variable(
        tf.truncated_normal([embedding_length, 3],
                            dtype=tf.float32,
                            stddev=1),
        name='classifier_weights')

    self._output_weights = tf.matmul(
        self._calculator_loom.output(), self._classifier_weights)
    self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=self._output_weights, labels=self._labels_placeholder))

    self._true_labels = tf.argmax(self._labels_placeholder, dimension=1)
    self._prediction = tf.argmax(self._output_weights, dimension=1)

    self._accuracy = tf.reduce_mean(tf.cast(
        tf.equal(self._true_labels, self._prediction),
        dtype=tf.float32))

  def _sign_label(self, n):
    """Labels: -/0/+."""
    if n < 0:
      return np.array([1, 0, 0], dtype='float32')
    elif n == 0:
      return np.array([0, 1, 0], dtype='float32')
    else:
      return np.array([0, 0, 1], dtype='float32')

  def variables(self):
    return self._calculator_loom.variables() + [self._classifier_weights]

  def loss(self):
    return self._loss

  def accuracy(self):
    return self._accuracy

  def build_feed_dict(self, expression_list):
    feed_dict = self._calculator_loom.build_feed_dict(expression_list)
    feed_dict[self._labels_placeholder] = np.vstack(
        [self._sign_label(e.result) for e in expression_list])
    return feed_dict
