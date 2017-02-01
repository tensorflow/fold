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
"""Tests for tensorflow_fold.blocks.block_compiler."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import multiprocessing
# import google3
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import test_lib
import tensorflow_fold.blocks.block_compiler as tdc
import tensorflow_fold.blocks.blocks as tdb


class CompilerTest(test_lib.TestCase):

  def test_init_raises(self):
    six.assertRaisesRegex(
        self, TypeError, 'root must have at least one output',
        tdc.Compiler.create, tdb.Record([]))
    six.assertRaisesRegex(
        self, TypeError, 'root outputs must all be tensors',
        tdc.Compiler.create, tdb.GetItem('foo'))
    six.assertRaisesRegex(
        self, TypeError, 'root output may not contain sequences',
        tdc.Compiler.create, tdb.Map(tdb.Scalar()))

  def test_init_loom_loom_input_tensor(self):
    loom_input_tensor = tf.placeholder('string')
    c = tdc.Compiler()
    c.compile(tdb.Scalar())
    c.init_loom(0, loom_input_tensor)
    with self.test_session() as sess:
      inp, = list(c.build_loom_inputs([42]))
      self.assertAllEqual([42], sess.run(
          c.output_tensors[0], {loom_input_tensor: inp}))

  def test_compiler_input_tensor(self):
    input_tensor = tf.Variable(['foobar', 'baz'],
                               dtype=tf.string, name='input_variable')
    init_op = tf.global_variables_initializer()
    root_block = tdb.InputTransform(len) >> tdb.Scalar()
    compiler = tdc.Compiler()
    compiler.compile(root_block)
    compiler.init_loom(max_depth=1, input_tensor=input_tensor)
    output_tensor, = compiler.output_tensors
    with self.test_session() as sess:
      sess.run(init_op)
      results = sess.run(output_tensor)
      self.assertEqual(len(results), 2)
      self.assertEqual(results[0], 6.)
      self.assertEqual(results[1], 3.)
      sess.run(input_tensor.assign(['foo', 'blah']))
      results = sess.run(output_tensor)
      self.assertEqual(len(results), 2)
      self.assertEqual(results[0], 3.)
      self.assertEqual(results[1], 4.)

  def test_loom_input_tensor(self):
    c = tdc.Compiler.create(tdb.Scalar())
    batches = [range(i) for i in xrange(1, 5)]
    loom_inputs = [c.build_loom_input_batched(batch) for batch in batches]
    output_tensor, = c.output_tensors
    with self.test_session() as sess:
      for loom_input, desired in zip(loom_inputs, batches):
        actual = sess.run(output_tensor, {c.loom_input_tensor: loom_input})
        self.assertAllEqual(desired, actual)
      desired = [x for batch in batches for x in batch]
      # We can feed loom_inputs because for some reason numpy doesn't
      # accept lists of objects with __array__, only at the top
      # level. Instead we feed a list of the actual arrays.
      actual = sess.run(output_tensor, {c.loom_input_tensor:
                                        [x.value for x in loom_inputs]})
      self.assertAllEqual(desired, actual)

  def test_register_tensor_types(self):
    b = tdb.Composition()
    with b.scope():
      b.output.reads(b.input[0])
    c = tdc.Compiler.create((tdb.Scalar('int32'), tdb.Scalar('float32')) >> b)
    output_tensor, = c.output_tensors
    with self.test_session() as sess:
      self.assertAllEqual(
          [42], sess.run(output_tensor, c.build_feed_dict([(42, 0.0)])))

  def test_multiprocessing(self):
    with self.test_session() as sess:
      c = tdc.Compiler.create(tdb.Scalar())
      out = c.output_tensors[0]
      def build(expect_nondeterminism=True):
        # test build_feed_dict
        fd = c.build_feed_dict(range(5))
        self.assertAllEqual(range(5), sess.run(out + 0, fd))
        fd = c.build_feed_dict(range(5), batch_size=1, ordered=True)
        self.assertAllEqual(range(5), sess.run(out + 0, fd))
        fd = c.build_feed_dict(range(5), batch_size=2)
        self.assertAllEqual(range(5), sorted(sess.run(out + 0, fd)))
        fd, ml = c.build_feed_dict(range(5), metric_labels=True)
        self.assertAllEqual(range(5), sess.run(out + 0, fd))
        self.assertEqual({}, ml)
        # test build_loom_input_batched
        batches = list(c.build_loom_input_batched(
            range(5), batch_size=2, ordered=True))
        self.assertEqual(len(batches), 3)
        self.assertAllEqual(range(5), sess.run(
            out + 0, {c.loom_input_tensor: batches}))
        self.assertAllEqual([0, 1], sess.run(
            out + 0, {c.loom_input_tensor: batches[0]}))
        self.assertAllEqual([2, 3], sess.run(
            out + 0, {c.loom_input_tensor: batches[1]}))
        self.assertAllEqual([4], sess.run(
            out + 0, {c.loom_input_tensor: batches[2]}))
        # test build_loom_inputs
        inputs = c.build_loom_inputs(range(5))
        for i in range(5):
          self.assertAllEqual([i], sess.run(
              out + 0, {c.loom_input_tensor: next(inputs)}))
        self.assertRaises(StopIteration, next, inputs)
        inputs = c.build_loom_inputs(range(5), chunk_size=1, ordered=True)
        for i in range(5):
          self.assertAllEqual([i], sess.run(
              out + 0, {c.loom_input_tensor: next(inputs)}))
        self.assertRaises(StopIteration, next, inputs)
        inputs = c.build_loom_inputs(range(5), chunk_size=2)
        self.assertAllEqual(range(5), sorted(sess.run(
            out + 0, {c.loom_input_tensor: list(inputs)})))
        inputs = c.build_loom_inputs(range(5), metric_labels=True)
        for i in range(5):
          inp, ml = next(inputs)
          self.assertAllEqual([i], sess.run(
              out + 0, {c.loom_input_tensor: inp}))
          self.assertEqual({}, ml)
        self.assertRaises(StopIteration, next, inputs)
        # test that we are actually using a pool
        a = list(c.build_loom_inputs(xrange(100), chunk_size=1))
        b = list(c.build_loom_inputs(xrange(100), chunk_size=1))
        self.assertEqual(a != b, expect_nondeterminism)
      with c.multiprocessing_pool(2):
        pool1, = c._pools
        self.assertTrue(isinstance(pool1, multiprocessing.pool.Pool))
        self.assertEqual(pool1, c.pool)
        build()
        with c.multiprocessing_pool(1):
          pool2 = c._pools[-1]
          self.assertEqual(pool2, c.pool)
          build(expect_nondeterminism=False)  # pool has only one subprocess
        with c.multiprocessing_pool():
          pool3 = c._pools[-1]
          self.assertEqual(pool3, c.pool)
          self.assertEqual([pool1, pool3], c._pools)
          build()
        self.assertEqual([pool1], c._pools)
        build()
      self.assertEqual([], c._pools)
      self.assertEqual(None, c.pool)
      build(expect_nondeterminism=False)  # no longer using a pool


class InitUninitializedTest(test_lib.TestCase):

  def test_empty(self):
    with self.test_session() as sess:
      self.assertEqual([], tdc._init_uninitialized(sess))

  def test_all_initialized(self):
    with self.test_session() as sess:
      x = tf.Variable(tf.zeros([]))
      sess.run(tf.initialize_variables([x]))
      self.assertEqual([], tdc._init_uninitialized(sess))

  def test_some_initialized(self):
    with self.test_session() as sess:
      x = tf.Variable(tf.zeros([]))
      self.assertEqual([x], tdc._init_uninitialized(sess))
      self.assertEqual(0, sess.run(x))
      y = tf.assign_add(x, 1)
      self.assertEqual([], tdc._init_uninitialized(sess))
      self.assertEqual(1, sess.run(y))
      self.assertEqual([], tdc._init_uninitialized(sess))
      # If we had done initialize_all_variables we'd see 1.
      self.assertEqual(2, sess.run(y))


if __name__ == '__main__':
  test_lib.main()
