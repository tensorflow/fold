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

"""Expression evaluation and generation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# import google3
from tensorflow_fold.loom.calculator_example import calculator_pb2


def random_expression(max_depth):
  """Recursively build a random CalculatorExpression."""
  def build(expression, max_depth):
    if max_depth == 0 or random.uniform(0, 1) < 1.0 / 3.0:
      expression.number = random.choice(range(10))
    else:
      expression.op = random.choice(
          calculator_pb2.CalculatorExpression.OpCode.values())
      build(expression.left, max_depth - 1)
      build(expression.right, max_depth - 1)
  expression = calculator_pb2.CalculatorExpression()
  build(expression, max_depth)
  return expression


def validate_expression(expression, recurse=True):
  """Check that 'expression' has the correct subset of fields set.

  Args:
    expression: An expression to validate.
    recurse: whether to recurse to the left and right fields if present.
      (Default: true).

  Raises:
    NameError: If an unknown op is found.
    TypeError: If expr contains things that aren't tuples or ints, or if it
      contains a tuple of the wrong arity.
  """
  if expression.HasField('number') == expression.HasField('op'):
    raise TypeError('Exactly one of {number, op} is required.')

  if expression.HasField('op') != expression.HasField('left'):
    raise TypeError('left should be present if and only if op is.')

  if expression.HasField('op') != expression.HasField('right'):
    raise TypeError('right should be present if and only if op is.')

  if expression.HasField('op'):
    if expression.op not in calculator_pb2.CalculatorExpression.OpCode.values():
      raise NameError('Unrecognized op : ', expression.op)
    if recurse:
      validate_expression(expression.left, True)
      validate_expression(expression.right, True)


def expression_depth(expression):
  validate_expression(expression, recurse=False)
  if expression.HasField('op'):
    return 1 + max(expression_depth(expression.left),
                   expression_depth(expression.right))
  return 0  # expression is a terminal (number).


def evaluate_expression(expression):
  """Computes an integer from an expression by performing the operations."""
  validate_expression(expression, recurse=False)
  if expression.HasField('number'):
    return expression.number
  a = evaluate_expression(expression.left)
  b = evaluate_expression(expression.right)
  if expression.op == calculator_pb2.CalculatorExpression.PLUS:
    return a + b
  if expression.op == calculator_pb2.CalculatorExpression.MINUS:
    return a - b
  if expression.op == calculator_pb2.CalculatorExpression.TIMES:
    return a * b
  if expression.op == calculator_pb2.CalculatorExpression.DIV:
    if b == 0:
      return 0
    else:
      return a // b
  else:
    raise NameError('Unrecognized op: ' + expression.op)
