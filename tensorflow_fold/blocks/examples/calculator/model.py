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

"""This is the model for the TensorFlow Fold calculator example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td

# The protobuf we're using here is from
# //tensorflow_fold/loom/calculator_example/calculator.proto

NUM_LABELS = 3  # negative, zero and positive.


def preprocess_expression(expr):
  # Set the op field for numbers, so we can handle cases uniformly.
  if expr['number'] is not None:
    expr['op'] = {'name': 'NUM'}
  return expr


def result_sign(result):
  if result < 0: return 0
  if result == 0: return 1
  return 2


class CalculatorModel(object):
  """A Fold model for calculator examples."""

  def __init__(self, state_size):
    # Expressions are either constants, or calculator ops that take other
    # expressions as their arguments.  Since an Expression is a recursive type,
    # the model must likewise be recursive.  A ForwardDeclaration declares the
    # type of expression, so it can be used before it before it is defined.
    expr_decl = td.ForwardDeclaration(td.PyObjectType(), state_size)

    # Create a block for each type of expression.
    # The terminals are the digits 0-9, which we map to vectors using
    # an embedding table.
    digit = (td.GetItem('number') >> td.Scalar(dtype='int32') >>
             td.Function(td.Embedding(10, state_size, name='terminal_embed')))

    # For non terminals, recursively apply expression to the left/right sides,
    # concatenate the results, and pass them through a fully-connected layer.
    # Each operation uses different weights in the FC layer.
    def bin_op(name):
      return (td.Record([('left', expr_decl()), ('right', expr_decl())]) >>
              td.Concat() >>
              td.FC(state_size, name='FC_'+name))

    # OneOf will dispatch its input to the appropriate case, based on the value
    # in the 'op'.'name' field.
    cases = td.OneOf(lambda x: x['op']['name'],
                     {'NUM': digit,
                      'PLUS': bin_op('PLUS'),
                      'MINUS': bin_op('MINUS'),
                      'TIMES': bin_op('TIMES'),
                      'DIV': bin_op('DIV')})

    # We do preprocessing to add 'NUM' as a distinct case.
    expression = td.InputTransform(preprocess_expression) >> cases
    expr_decl.resolve_to(expression)

    # Get logits from the root of the expression tree
    expression_logits = (expression >>
                         td.FC(NUM_LABELS, activation=None, name='FC_logits'))

    # The result is stored in the expression itself.
    # We ignore it in td.Record above, and pull it out here.
    expression_label = (td.GetItem('result') >>
                        td.InputTransform(result_sign) >>
                        td.OneHot(NUM_LABELS))

    # For the overall model, return a pair of (logits, labels)
    # The AllOf block will run each of its children on the same input.
    model = td.AllOf(expression_logits, expression_label)
    self._compiler = td.Compiler.create(model)

    # Get the tensorflow tensors that correspond to the outputs of model.
    # `logits` and `labels` are TF tensors, and we can use them to
    # compute losses in the usual way.
    (logits, labels) = self._compiler.output_tensors

    self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))

    self._accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1),
                         tf.argmax(logits, 1)),
                dtype=tf.float32))

    self._global_step = tf.Variable(0, name='global_step', trainable=False)
    optr = tf.train.GradientDescentOptimizer(0.01)
    self._train_op = optr.minimize(self._loss, global_step=self._global_step)

  @property
  def loss(self):
    return self._loss

  @property
  def accuracy(self):
    return self._accuracy

  @property
  def train_op(self):
    return self._train_op

  @property
  def global_step(self):
    return self._global_step

  def build_feed_dict(self, expressions):
    return self._compiler.build_feed_dict(expressions)
