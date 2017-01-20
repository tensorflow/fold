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

"""Tests for calculator example."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

# import google3
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from google.protobuf import text_format
from tensorflow_fold.loom.calculator_example import calculator
from tensorflow_fold.loom.calculator_example import calculator_pb2


def evaluate_expression(string):
  return calculator.evaluate_expression(
      text_format.Parse(string, calculator_pb2.CalculatorExpression()))


class CalculatorTest(tf.test.TestCase):

  def test_generated_expression_depth(self):
    random.seed(0xdeadbeef)  # Make RandomExpression deterministic.
    for _ in xrange(1000):
      expression = calculator.random_expression(5)
      calculator.validate_expression(expression)
      self.assertTrue(calculator.expression_depth(expression) <= 5)

  def test_eval(self):
    self.assertEqual(0, evaluate_expression(
        """op: DIV left<number: 3> right<number: 0>"""))

    # Division by zero defaults to zero.
    for n in xrange(10):
      self.assertEqual(n, evaluate_expression(
          """number: {n}""".format(n=n)))
      self.assertEqual(3 + n, evaluate_expression(
          """op: PLUS left<number: 3> right<number: {n}>""".format(n=n)))
      self.assertEqual(2 * n, evaluate_expression(
          """op: PLUS left<number: {n}> right<number: {n}>""".format(n=n)))
      self.assertEqual(0, evaluate_expression(
          """op: MINUS left<number: {n}> right<number: {n}>""".format(n=n)))
      self.assertEqual(n * n * n, evaluate_expression(
          """op: TIMES
             left<number: {n}>
             right<op: TIMES left<number: {n}> right<number: {n}>>
          """.format(n=n)))
      self.assertEqual(n, evaluate_expression(
          """op: DIV left<number: {x}> right<number: 5>""".format(
              x=5 * n + 3)))

if __name__ == '__main__':
  tf.test.main()
