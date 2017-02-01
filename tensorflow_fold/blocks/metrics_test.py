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
"""Tests for tensorflow_fold.blocks.metrics."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import numpy as np
import six
import tensorflow as tf
from tensorflow_fold.blocks import test_lib
import tensorflow_fold.blocks.block_compiler as tdc
import tensorflow_fold.blocks.blocks as tdb
import tensorflow_fold.blocks.metrics as tdm


def _ispositive(x):
  if isinstance(x, list):
    return x[0] > 0
  else:
    return x > 0


def _pos_neg_block(shape):
  """Returns a Tensor block of shape, adding positive/negative metrics."""
  c = tdb.Composition()
  with c.scope():
    tdb.OneOf(_ispositive,
              (tdm.Metric('negative'), tdm.Metric('positive')),
              pre_block=tdb.Tensor(shape)).reads(c.input)
    c.output.reads(tdb.Tensor(shape).reads(c.input))
  return c


class MetricsTest(test_lib.TestCase):

  def test_metrics_scalar(self):
    block = tdb.Map(_pos_neg_block([])) >> tdb.Sum()

    with self.test_session() as sess:
      compiler = tdc.Compiler.create(block)
      sess.run(tf.global_variables_initializer())

      fd = compiler.build_feed_dict([[1, 2, 3, 4]])
      self.assertSameStructure(
          [10.], sess.run(compiler.output_tensors[0], fd).tolist())

      positive = compiler.metric_tensors['positive']
      negative = compiler.metric_tensors['negative']

      fd = compiler.build_feed_dict([[1, -2, 3, -4, 5, -6]])
      pos, neg = sess.run([positive, negative], fd)
      np.testing.assert_equal(pos, [1, 3, 5])
      np.testing.assert_equal(neg, [-2, -4, -6])

      fd = compiler.build_feed_dict([[-1, -2, 3, -4, -5, -6]])
      pos, neg = sess.run([positive, negative], fd)
      np.testing.assert_equal(pos, [3])   # test single value
      np.testing.assert_equal(neg, [-1, -2, -4, -5, -6])

      fd = compiler.build_feed_dict([[1, 2, 3, 4, 5, 6]])
      pos, neg = sess.run([positive, negative], fd)
      np.testing.assert_equal(pos, [1, 2, 3, 4, 5, 6])
      np.testing.assert_equal(neg, [])   # test no values

      # test batches
      fd = compiler.build_feed_dict([[1, 2, -3, -4], [5, 6, -7, -8, 0]])
      pos, neg = sess.run([positive, negative], fd)
      np.testing.assert_equal(pos, [1, 2, 5, 6])
      np.testing.assert_equal(neg, [-3, -4, -7, -8, 0])

  def test_metrics_vector(self):
    block = tdb.Map(_pos_neg_block([2])) >> tdb.Sum()

    with self.test_session() as sess:
      compiler = tdc.Compiler.create(block)
      sess.run(tf.global_variables_initializer())

      positive = compiler.metric_tensors['positive']
      negative = compiler.metric_tensors['negative']

      fd = compiler.build_feed_dict([[[1, 2], [-2, -3], [4, 5]]])
      pos, neg = sess.run([positive, negative], fd)
      np.testing.assert_equal(pos, [[1, 2], [4, 5]])
      np.testing.assert_equal(neg, [[-2, -3]])

  def test_metrics_raises(self):
    sp0 = _pos_neg_block([])
    spn = _pos_neg_block([2])
    block = {'foo': sp0, 'bar:': spn} >> tdb.Concat()
    six.assertRaisesRegex(
        self, TypeError, 'Metric [a-z]+tive has incompatible types',
        tdc.Compiler.create, block)

  def test_metrics_labeled(self):
    tree1 = [1, 'a', [2, 'b'], [3, 'c'], [4, 'd']]
    tree2 = [5, 'e', [6, 'f', [7, 'g']]]
    fwd = tdb.ForwardDeclaration()

    leaf = (tdb.Scalar('int32'), tdb.Identity()) >>  tdm.Metric('leaf')
    internal = tdb.AllOf(
        (tdb.Scalar('int32'), tdb.Identity())  >> tdm.Metric('internal'),
        tdb.Slice(start=2) >> tdb.Map(fwd())) >> tdb.Void()
    tree = tdb.OneOf(key_fn=lambda expr: len(expr) > 2,
                     case_blocks=(leaf, internal))
    fwd.resolve_to(tree)

    with self.test_session() as sess:
      c = tdc.Compiler.create(tree)
      feed_dict, labels = c.build_feed_dict([tree1, tree2], metric_labels=True)
      self.assertEqual(['b', 'c', 'd', 'g'], labels['leaf'])
      self.assertEqual(['a', 'e', 'f'], labels['internal'])
      leaf_values, internal_values = sess.run(
          [c.metric_tensors['leaf'], c.metric_tensors['internal']], feed_dict)
      np.testing.assert_equal([2, 3, 4, 7], leaf_values)
      np.testing.assert_equal([1, 5, 6], internal_values)

if __name__ == '__main__':
  test_lib.main()
