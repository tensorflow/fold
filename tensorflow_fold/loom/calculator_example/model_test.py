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

"""Smoke test for Loom Calculator Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

# import google3
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_fold.loom.calculator_example import calculator
from tensorflow_fold.loom.calculator_example import model


class ModelTest(tf.test.TestCase):

  def test_loss_goes_down(self):
    # Ensure determinism:
    tf.set_random_seed(0xdeadbeef)
    random.seed(0xdeadbeef)

    expression_list = []
    for _ in xrange(1000):
      expression = calculator.random_expression(5)
      expression.result = calculator.evaluate_expression(expression)
      expression_list.append(expression)

    classifier = model.CalculatorSignClassifier(embedding_length=3)

    variables = classifier.variables()
    loss = classifier.loss()

    optr = tf.train.GradientDescentOptimizer(0.001)
    trainer = optr.minimize(loss, var_list=variables)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      old_loss = classifier.loss().eval(
          feed_dict=classifier.build_feed_dict(expression_list))

      for _ in xrange(20):
        sess.run([trainer],
                 feed_dict=classifier.build_feed_dict(expression_list))

      new_loss = classifier.loss().eval(
          feed_dict=classifier.build_feed_dict(expression_list))

      self.assertLess(new_loss, old_loss)


if __name__ == '__main__':
  tf.test.main()
