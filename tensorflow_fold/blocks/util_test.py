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
"""Tests for tensorflow_fold.blocks.util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import test_lib
from tensorflow_fold.blocks import util


class UtilTest(test_lib.TestCase):

  def test_edible_iterator_int(self):
    with self.test_session() as sess:
      i = util.EdibleIterator(x for x in [2, 4, 6])
      x = tf.placeholder(tf.int32)
      self.assertEqual([4, 8, 12], sess.run(x + x, {x: i}).tolist())
      self.assertEqual([3, 5, 7], sess.run(x + 1, {x: i}).tolist())

  def test_edible_iterator_str(self):
    with self.test_session() as sess:
      i = util.EdibleIterator(x for x in ['foo', 'bar'])
      x = tf.placeholder(tf.string)
      self.assertEqual(b'foo', sess.run(x[0], {x: i}))
      self.assertEqual(b'bar', sess.run(x[1], {x: i}))
      self.assertEqual([b'foo', b'bar', b'foo', b'bar'],
                       sess.run(tf.concat([x, x], 0), {x: i}).tolist())

  def test_edible_iterator_empty(self):
    with self.test_session() as sess:
      i = util.EdibleIterator(iter([]))
      x = tf.placeholder(tf.string)
      self.assertEqual([[]], sess.run(tf.expand_dims(x, 0), {x: i}).tolist())
      self.assertEqual([b'foo'],
                       sess.run(tf.concat([['foo'], x], 0), {x: i}).tolist())

  def test_group_by_batches(self):
    self.assertEqual([], list(util.group_by_batches([], 2)))
    self.assertEqual([[1], [2], [3]], list(util.group_by_batches([1, 2, 3], 1)))
    self.assertEqual([[1, 2], [3]], list(util.group_by_batches([1, 2, 3], 2)))

  def test_group_by_batches_truncated(self):
    self.assertEqual([], list(util.group_by_batches([], 2, truncate=True)))
    self.assertEqual([[1], [2], [3]],
                     list(util.group_by_batches([1, 2, 3], 1, truncate=True)))
    self.assertEqual([[1, 2]],
                     list(util.group_by_batches([1, 2, 3], 2, truncate=True)))

  def test_epochs(self):
    self.assertEqual([[0, 0]] * 5,
                     [list(x) for x in util.epochs((0 for _ in xrange(2)), 5)])
    epochs = util.epochs(xrange(5), shuffle=False)
    self.assertSequenceEqual(list(next(epochs)), xrange(5))
    self.assertSequenceEqual(list(next(epochs)), xrange(5))
    self.assertSequenceEqual(list(next(epochs)), xrange(5))
    epochs = util.epochs(xrange(5))
    self.assertSequenceEqual(list(next(epochs)), xrange(5))
    self.assertEqual(set(next(epochs)), set(xrange(5)))
    self.assertEqual(set(next(epochs)), set(xrange(5)))

  def test_epochs_n_is_one(self):
    items = [1]
    result, = list(util.epochs(items, 1))
    self.assertIs(items, result)


if __name__ == '__main__':
  test_lib.main()
