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

"""Tests for tensorflow_fold.loom."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import google3
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow_fold.loom import loom


class BinaryLoomOp(loom.LoomOp):

  def __init__(self, type_shape, op):
    self._op = op
    super(BinaryLoomOp, self).__init__([type_shape, type_shape], [type_shape])

  def instantiate_batch(self, inputs):
    return [self._op(inputs[0], inputs[1])]


class CatLoomOp(loom.LoomOp):

  def __init__(self, left_type_shape, right_type_shape):
    if left_type_shape.dtype != right_type_shape.dtype:
      raise TypeError('CatLoomOp needs TypeShapes with the same dtype.')
    if left_type_shape.shape[1:] != right_type_shape.shape[1:]:
      raise TypeError('CatLoomOp needs TypeShapes with shapes agreeing on '
                      'all but the first dimension.')
    dim0 = left_type_shape.shape[0] + right_type_shape.shape[0]
    cat_type_shape = loom.TypeShape(
        left_type_shape.dtype, (dim0,) + left_type_shape.shape[1:])
    super(CatLoomOp, self).__init__(
        [left_type_shape, right_type_shape], [cat_type_shape])

  def instantiate_batch(self, inputs):
    return [tf.concat(inputs, 1)]


def group_values(xs, group_size):
  return [xs[i * group_size : (i + 1) * group_size]
          for i in xrange(len(xs) // group_size)]


class LoomTest(tf.test.TestCase):

  def test_type_shape(self):
    self.assertEqual(loom.TypeShape('float32', ()).dtype, 'float32')
    self.assertRaises(TypeError, loom.TypeShape, np.float32, ())
    self.assertRaises(TypeError, loom.TypeShape, object, ())
    self.assertRaises(TypeError, loom.TypeShape, 'foobar', ())
    self.assertRaises(TypeError, loom.TypeShape, 'float32', (3.2,))
    self.assertRaises(TypeError, loom.TypeShape, 'float32', (-4,))

  def test_path_through_loom_op(self):
    shape = loom.TypeShape('int64', (3,))
    op = loom.PassThroughLoomOp(shape)
    self.assertEqual([shape], op.input_type_shapes)
    self.assertEqual([shape], op.output_type_shapes)
    test_obj = tf.constant([1, 2, 3], dtype=shape.dtype)
    self.assertEqual([test_obj], op.instantiate_batch([test_obj]))

  def test_loom_type_shapes(self):
    shape = loom.TypeShape('int64', (500,))
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    the_loom = loom.Loom(named_ops=ops)
    self.assertEqual([shape], the_loom.type_shapes)

  def test_loom_build_graph(self):
    shape = loom.TypeShape('int64', (500,))
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    _ = loom.Loom(named_ops=ops)

  def test_loom_type_shapes2(self):
    shape1 = loom.TypeShape('float32', (20,))
    shape2 = loom.TypeShape('float32', (30,))
    ops = {'add': BinaryLoomOp(shape1, tf.add),
           'mul': BinaryLoomOp(shape2, tf.multiply)}
    the_loom = loom.Loom(named_ops=ops)
    self.assertEqual([shape1, shape2], the_loom.type_shapes)

  def test_loom_type_shapes2_get_type_shape(self):
    shape1 = loom.TypeShape('float32', (20,))
    shape2 = loom.TypeShape('float32', (30,))
    ops = {'add': BinaryLoomOp(shape1, tf.add),
           'mul': BinaryLoomOp(shape2, tf.multiply)}
    the_loom = loom.Loom(named_ops=ops)
    weaver = the_loom.make_weaver()
    self.assertEqual(
        weaver.get_type_shape(weaver(np.zeros((20,), 'float32'))),
        shape1)
    self.assertEqual(
        weaver.get_type_shape(weaver(np.zeros((30,), 'float32'))),
        shape2)

  def test_loom_type_shapes_output_shapes(self):
    shape1 = loom.TypeShape('float32', (20,))
    shape2 = loom.TypeShape('float32', (30,))
    ops = {'add': BinaryLoomOp(shape1, tf.add),
           'mul': BinaryLoomOp(shape2, tf.multiply)}
    the_loom = loom.Loom(named_ops=ops)
    self.assertEqual([None, 20],
                     the_loom.output_tensor(shape1).get_shape().as_list())
    self.assertEqual([None, 30],
                     the_loom.output_tensor(shape2).get_shape().as_list())

  def test_loom_type_shapes_2_distinct_tags(self):
    shape1 = loom.TypeShape('float32', (20,), 'alice')
    shape2 = loom.TypeShape('float32', (20,), 'bob')
    ops = {'add': BinaryLoomOp(shape1, tf.add),
           'mul': BinaryLoomOp(shape2, tf.multiply)}
    the_loom = loom.Loom(named_ops=ops)
    self.assertEqual([shape1, shape2], the_loom.type_shapes)

  def test_good_constant(self):
    shape = loom.TypeShape('int64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(named_ops=ops)
    weaver = the_loom.make_weaver()
    self.assertTrue(weaver(np.array([1, 2, 3], dtype='int64'))
                    is not None)

  def test_constant_network(self):
    shape = loom.TypeShape('int64', (3,))
    value = np.array([1, 2, 3], dtype='int64')
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c = weaver(value)
      result = output_tensor.eval(feed_dict=weaver.build_feed_dict([c]))
    self.assertTrue((result[0] == value).all())

  def test_add_output_method(self):
    shape = loom.TypeShape('int64', (3,))
    value = np.array([1, 2, 3], dtype='int64')
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      weaver.add_output(weaver(value))
      result = output_tensor.eval(feed_dict=weaver.build_feed_dict())
    self.assertTrue((result[0] == value).all())

  def test_constant_network_with_tags(self):
    shape1 = loom.TypeShape('int64', (3,), 'alpha')
    shape2 = loom.TypeShape('int64', (3,), 'beta')
    value1 = np.array([1, 2, 3], dtype='int64')
    value2 = np.array([4, 5, 6], dtype='int64')
    ops = {'add1': BinaryLoomOp(shape1, tf.add),
           'add2': BinaryLoomOp(shape2, tf.add)}
    the_loom = loom.Loom(named_ops=ops)
    output_tensor1 = the_loom.output_tensor(shape1)
    output_tensor2 = the_loom.output_tensor(shape2)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(value1, tag='alpha')
      c2 = weaver(value2, tag='beta')
      result1 = output_tensor1.eval(
          feed_dict=weaver.build_feed_dict([c2, c1]))
      result2 = output_tensor2.eval(
          feed_dict=weaver.build_feed_dict([c2, c1]))
    self.assertTrue((result1[0] == value1).all())
    self.assertTrue((result2[0] == value2).all())

  def test_constant_network_with_tags_dry_run(self):
    shape1 = loom.TypeShape('int64', (3,), 'alpha')
    shape2 = loom.TypeShape('int64', (3,), 'beta')
    value1 = np.array([1, 2, 3], dtype='int64')
    value2 = np.array([4, 5, 6], dtype='int64')
    ops = {'add1': BinaryLoomOp(shape1, tf.add),
           'add2': BinaryLoomOp(shape2, tf.add)}
    the_loom = loom.Loom(named_ops=ops, dry_run=True)
    output_tensor1 = the_loom.output_tensor(shape1)
    output_tensor2 = the_loom.output_tensor(shape2)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(value1, tag='alpha')
      c2 = weaver(value2, tag='beta')
      result1 = output_tensor1.eval(
          feed_dict=weaver.build_feed_dict([c2, c1]))
      result2 = output_tensor2.eval(
          feed_dict=weaver.build_feed_dict([c2, c1]))
    zero_vec = np.zeros_like(value1)
    self.assertTrue((result1[0] == zero_vec).all())
    self.assertTrue((result2[0] == zero_vec).all())

  def test_simple_sum_network(self):
    shape = loom.TypeShape('int64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(np.array([1, 1, 8], dtype='int64'))
      c2 = weaver(np.array([2, 3, 9], dtype='int64'))
      sum_result = weaver.add(c1, c2)
      result = output_tensor.eval(
          feed_dict=weaver.build_feed_dict([sum_result]))
    self.assertTrue((result == np.array([[3, 4, 17]], dtype='int64')).all())

  def test_simple_sum_network_with_max_depth(self):
    shape = loom.TypeShape('int64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(max_depth=2, named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(np.array([1, 1, 8], dtype='int64'))
      c2 = weaver(np.array([2, 3, 9], dtype='int64'))
      sum_result = weaver.add(c1, c2)
      result = output_tensor.eval(
          feed_dict=weaver.build_feed_dict([sum_result]))
    self.assertTrue((result == np.array([[3, 4, 17]], dtype='int64')).all())

  def test_simple_sum_network_with_batch_inputs(self):
    batch_vectors = tf.placeholder('int64')
    shape = loom.TypeShape('int64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(named_ops=ops, batch_inputs={shape: batch_vectors})
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      sum_result = weaver.add(weaver.batch_input(shape, 0),
                              weaver.batch_input(shape, 1))
      weaver.add_output(sum_result)
      fd = {the_loom.input_tensor: weaver.serialize(),
            batch_vectors: np.array([[1, 2, 3], [4, 5, 6]], dtype='int64')}
      result = output_tensor.eval(feed_dict=fd)
    self.assertTrue((result == np.array([[5, 7, 9]], dtype='int64')).all())

  def test_two_layer_sum_network(self):
    shape = loom.TypeShape('int64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(np.array([1, 2, 3], dtype='int64'))
      c2 = weaver(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver(np.array([3, 6, 9], dtype='int64'))
      c4 = weaver(np.array([4, 8, 12], dtype='int64'))
      sum_1_2 = weaver.add(c1, c2)
      sum_3_4 = weaver.add(c3, c4)
      sum_1_2_3_4 = weaver.add(sum_1_2, sum_3_4)
      result = output_tensor.eval(
          feed_dict=weaver.build_feed_dict([sum_1_2_3_4]))
    self.assertTrue((result == np.array([[10, 20, 30]], dtype='int64')).all())

  def test_three_layer_sum_network(self):
    shape = loom.TypeShape('int64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add)}
    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)

    with self.test_session():
      weaver = the_loom.make_weaver()
      vals = [weaver(np.array([0, 1, 1 << k], dtype='int64'))
              for k in range(8)]
      for _ in xrange(3):
        vals = [weaver.add(*args) for args in group_values(vals, 2)]
      big_sum = vals[0]
      result = output_tensor.eval(
          feed_dict=weaver.build_feed_dict([big_sum]))
    self.assertTrue((result == np.array([[0, 8, 255]], dtype='int64')).all())

  def test_two_ops_network(self):
    shape = loom.TypeShape('int64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(np.array([1, 2, 3], dtype='int64'))
      c2 = weaver(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver(np.array([3, 6, 9], dtype='int64'))
      sum_2_3 = weaver.add(c2, c3)
      sum_12_13 = weaver.mul(c1, sum_2_3)
      result = output_tensor.eval(
          feed_dict=weaver.build_feed_dict([sum_12_13]))
    self.assertTrue((result == np.array([[5, 20, 45]], dtype='int64')).all())

  def test_two_ops_network_with_merge(self):
    shape = loom.TypeShape('int64', (3,))
    named_tensors = {'c1': tf.constant([1, 2, 3], dtype='int64')}
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    the_loom = loom.Loom(named_tensors=named_tensors, named_ops=ops)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver1 = the_loom.make_weaver()
      c1 = weaver1.c1
      c2 = weaver1(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver1(np.array([3, 6, 9], dtype='int64'))
      sum_2_3 = weaver1.add(c2, c3)
      sum_12_13 = weaver1.mul(c1, sum_2_3)
      weaver1.add_output(sum_12_13)

      weaver2 = the_loom.make_weaver()
      c1 = weaver2.c1
      c2 = weaver2(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver2(np.array([3, 6, 9], dtype='int64'))
      sum_1_3 = weaver2.add(c1, c3)
      sum_21_23 = weaver2.mul(c2, sum_1_3)
      weaver2.add_output(sum_21_23)

      weaver_merged = [weaver1.serialize(), weaver2.serialize()]

      result = output_tensor.eval(
          feed_dict={the_loom.input_tensor: weaver_merged})
    self.assertTrue((result == np.array(
        [[5, 20, 45], [8, 32, 72]], dtype='int64')).all())

  def test_two_ops_network_with_merge_serialized(self):
    loom_input_tensor = tf.placeholder('string')
    shape = loom.TypeShape('int64', (3,))
    named_tensors = {'c1': tf.constant([1, 2, 3], dtype='int64')}
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    the_loom = loom.Loom(named_tensors=named_tensors, named_ops=ops,
                         loom_input_tensor=loom_input_tensor)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver1 = the_loom.make_weaver()
      c1 = weaver1.c1
      c2 = weaver1(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver1(np.array([3, 6, 9], dtype='int64'))
      sum_2_3 = weaver1.add(c2, c3)
      sum_12_13 = weaver1.mul(c1, sum_2_3)
      weaver1.add_output(sum_12_13)

      weaver2 = the_loom.make_weaver()
      c1 = weaver2.c1
      c2 = weaver2(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver2(np.array([3, 6, 9], dtype='int64'))
      sum_1_3 = weaver2.add(c1, c3)
      sum_21_23 = weaver2.mul(c2, sum_1_3)
      weaver2.add_output(sum_21_23)

      input1_str = weaver1.serialize()
      input2_str = weaver2.serialize()

      result = output_tensor.eval(
          feed_dict={loom_input_tensor: [input1_str, input2_str]})
    self.assertTrue((result == np.array(
        [[5, 20, 45], [8, 32, 72]], dtype='int64')).all())

  def test_two_ops_network_tagged_named_tensorx(self):
    shape = loom.TypeShape('int64', (3,), tag='x')
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    named_tensors = {
        'c1': (tf.constant(np.array([1, 2, 3], dtype='int64')), 'x'),
        'c2': (tf.constant(np.array([2, 4, 6], dtype='int64')), 'x'),
        'c3': (tf.constant(np.array([3, 6, 9], dtype='int64')), 'x')
    }
    the_loom = loom.Loom(named_ops=ops, named_tensors=named_tensors)
    output_tensor = the_loom.output_tensor(shape)
    with self.test_session():
      weaver = the_loom.make_weaver()
      sum_2_3 = weaver.add(weaver.c2, weaver.c3)
      sum_12_13 = weaver.mul(weaver.c1, sum_2_3)
      result = output_tensor.eval(
          feed_dict=weaver.build_feed_dict([sum_12_13]))
    self.assertTrue((result == np.array([[5, 20, 45]], dtype='int64')).all())

  def test_two_shapes_network(self):
    shape3 = loom.TypeShape('int64', (3,))
    shape6 = loom.TypeShape('int64', (6,))
    cat_op = CatLoomOp(shape3, shape3)
    self.assertEqual(cat_op.output_type_shapes[0], shape6)
    ops = {'add3': BinaryLoomOp(shape3, tf.add),
           'add6': BinaryLoomOp(shape6, tf.add),
           'cat': cat_op}

    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape6)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(np.array([1, 2, 3], dtype='int64'))
      c2 = weaver(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver(np.array([3, 6, 9], dtype='int64'))
      c4 = weaver(np.array([4, 8, 12], dtype='int64'))
      self.assertEqual(0, weaver.deepest)
      self.assertEqual(0, weaver.depth(c1))
      v_3 = weaver.add3(c1, c2)
      v_7 = weaver.add3(c3, c4)
      self.assertEqual(1, weaver.deepest)
      self.assertEqual(1, weaver.depth(v_7))
      v_3_v_7 = weaver.cat(v_3, v_7)
      self.assertEqual(2, weaver.deepest)
      self.assertEqual(2, weaver.depth(v_3_v_7))
      v_5 = weaver.add3(c1, c4)
      v_5_v_2 = weaver.cat(v_5, c2)  # Level skip.
      v_8_v_9 = weaver.add6(v_3_v_7, v_5_v_2)
      self.assertEqual(3, weaver.deepest)
      self.assertEqual(3, weaver.depth(v_8_v_9))
      result = output_tensor.eval(
          feed_dict=weaver.build_feed_dict([v_8_v_9]))
    self.assertTrue((result == np.array(
        [[8, 16, 24, 9, 18, 27]], dtype='int64')).all())

  def test_two_shapes_network_with_serialization(self):
    shape3 = loom.TypeShape('int64', (3,))
    shape6 = loom.TypeShape('int64', (6,))
    cat_op = CatLoomOp(shape3, shape3)
    self.assertEqual(cat_op.output_type_shapes[0], shape6)
    ops = {'add3': BinaryLoomOp(shape3, tf.add),
           'add6': BinaryLoomOp(shape6, tf.add),
           'cat': cat_op}

    the_loom = loom.Loom(named_ops=ops)
    output_tensor = the_loom.output_tensor(shape6)
    with self.test_session():
      weaver = the_loom.make_weaver()
      c1 = weaver(np.array([1, 2, 3], dtype='int64'))
      c2 = weaver(np.array([2, 4, 6], dtype='int64'))
      c3 = weaver(np.array([3, 6, 9], dtype='int64'))
      c4 = weaver(np.array([4, 8, 12], dtype='int64'))
      self.assertEqual(0, weaver.deepest)
      v_3 = weaver.add3(c1, c2)
      v_7 = weaver.add3(c3, c4)
      self.assertEqual(1, weaver.deepest)
      v_3_v_7 = weaver.cat(v_3, v_7)
      self.assertEqual(2, weaver.deepest)
      v_5 = weaver.add3(c1, c4)
      v_5_v_2 = weaver.cat(v_5, c2)  # Level skip.
      v_8_v_9 = weaver.add6(v_3_v_7, v_5_v_2)
      self.assertEqual(3, weaver.deepest)
      weaver.add_output(v_8_v_9)
      weaver_proto = weaver.serialize()
      new_weaver = the_loom.deserialize_weaver(weaver_proto)

      result = output_tensor.eval(feed_dict=new_weaver.build_feed_dict())
    self.assertTrue((result == np.array(
        [[8, 16, 24, 9, 18, 27]], dtype='int64')).all())

  def test_gradient(self):
    x_var = tf.Variable(tf.zeros([3], dtype='float64'), name='x')
    shape = loom.TypeShape('float64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    the_loom = loom.Loom(named_tensors={'x': x_var}, named_ops=ops)

    output_tensor = the_loom.output_tensor(shape)
    output = tf.reduce_sum(output_tensor)
    gradient = tf.gradients(output, [x_var])[0]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      weaver = the_loom.make_weaver()
      m = weaver(np.array([1, 2, 3], dtype='float64'))
      b = weaver(np.array([47, 9, -1], dtype='float64'))
      mx = weaver.mul(m, weaver.x)
      mx_plus_b = weaver.add(mx, b)
      result = gradient.eval(feed_dict=weaver.build_feed_dict([mx_plus_b]))
    self.assertTrue((result == np.array(
        [1.0, 2.0, 3.0], dtype='float64')).all())

  def test_gradient_with_direct_feed_dict(self):
    x_var = tf.Variable(tf.zeros([3], dtype='float64'), name='x')
    shape = loom.TypeShape('float64', (3,))
    ops = {'add': BinaryLoomOp(shape, tf.add),
           'mul': BinaryLoomOp(shape, tf.multiply)}
    the_loom = loom.Loom(named_tensors={'x': x_var}, named_ops=ops,
                         direct_feed_dict=True)

    output_tensor = the_loom.output_tensor(shape)
    output = tf.reduce_sum(output_tensor)
    gradient = tf.gradients(output, [x_var])[0]
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

      weaver = the_loom.make_weaver()
      m = weaver(np.array([1, 2, 3], dtype='float64'))
      b = weaver(np.array([47, 9, -1], dtype='float64'))
      mx = weaver.mul(m, weaver.x)
      mx_plus_b = weaver.add(mx, b)
      result = gradient.eval(feed_dict=weaver.build_feed_dict([mx_plus_b]))
    self.assertTrue((result == np.array(
        [1.0, 2.0, 3.0], dtype='float64')).all())


if __name__ == '__main__':
  tf.test.main()
