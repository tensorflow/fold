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
"""Tests for tensorflow_fold.blocks.blocks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import functools
import itertools
# import google3
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import test_lib
import tensorflow_fold.blocks.block_compiler as tdc
import tensorflow_fold.blocks.blocks as tdb
import tensorflow_fold.blocks.layers as tdl
import tensorflow_fold.blocks.metrics as tdm
import tensorflow_fold.blocks.result_types as tdt
from tensorflow_fold.util import test_pb2


class BlocksTest(test_lib.TestCase):

  def __init__(self, *args, **kwargs):
    super(BlocksTest, self).__init__(*args, **kwargs)
    self.longMessage = True

  def _assertBuilds(self, desired, block, inp, fd, use_while_loop=True):
    actual = block.eval(inp, feed_dict=fd, tolist=True,
                        use_while_loop=use_while_loop)
    if isinstance(desired, itertools.repeat):
      # itertools.repeat is not a value type
      self.assertTrue(isinstance(actual, itertools.repeat))
      actual = next(actual)
      desired = next(desired)
    msg = '\n%s vs.\n%s\n' % (desired, actual)
    self.assertSameStructure(desired, actual, 'desired', 'actual', msg)

  def assertBuilds(self, desired, block, inp, max_depth=1, feed_dict=None):
    if max_depth is not None: self.assertEqual(max_depth, block.max_depth(inp))
    with self.test_session() as sess:
      # Test using a while loop.
      self._assertBuilds(desired, block, inp, feed_dict)
      # Test with explicit unrolling.
      self._assertBuilds(desired, block, inp, feed_dict, use_while_loop=False)
      # If the block can be compiled, test batching (otherwise we can't).
      try:
        c = tdc.Compiler.create(block)
      except TypeError:
        return
      fd = c.build_feed_dict([inp, inp])
      if feed_dict is not None: fd.update(feed_dict)
      batch_out = sess.run(c.output_tensors, feed_dict=fd)
      if c.metric_tensors: desired, _ = desired
      desired = block.output_type.flatten(desired)
      for out, desired_out in zip(batch_out, desired):
        self.assertSameStructure(out[0].tolist(), out[1].tolist())
        self.assertSameStructure(desired_out, out[0].tolist())

  def assertBuildsConst(self, desired, block, inp):
    # TODO(moshelooks): actually test constness
    self.assertBuilds(desired, block, inp, max_depth=None)

  def test_scalar(self):
    self.assertBuildsConst(42., tdb.Scalar(), 42)

  def test_vector(self):
    self.assertBuildsConst([1., 2., 3.], tdb.Vector(3), [1, 2, 3])

  def test_from_tensor(self):
    t = tf.placeholder('int32', [2, 1])
    block = tdb.FromTensor(t)
    fd = {t: [[1], [2]]}
    self.assertBuilds([[1], [2]], block, None, max_depth=0, feed_dict=fd)

  def test_from_tensor_const(self):
    block = tdb.FromTensor(tf.constant([1, 2, 3], 'int32'))
    self.assertBuildsConst([1, 2, 3], block, None)

  def test_from_tensor_const_ndarray(self):
    block = tdb.FromTensor(np.zeros([2, 2], 'int32'))
    self.assertBuildsConst([[0, 0], [0, 0]], block, None)

  def test_from_tensor_variable(self):
    v = tf.Variable(np.ones(2))
    block = tdb.FromTensor(v)
    self.assertBuilds([1., 1.], block, None, max_depth=0)

  def test_from_tensor_raises(self):
    self.assertRaisesWithLiteralMatch(
        TypeError, '42 is not a tensor or np.ndarray', tdb.FromTensor, 42)
    self.assertRaisesWithLiteralMatch(
        TypeError, 'shape <unknown> is not fully defined; call set_shape()',
        tdb.FromTensor, tf.placeholder('int32'))
    self.assertRaisesWithLiteralMatch(
        TypeError, 'shape (1, 2, ?) is not fully defined; call set_shape()',
        tdb.FromTensor, tf.placeholder('int32', [1, 2, None]))

  def test_pyobject_composition(self):
    block = tdb.AllOf(tdb.Identity(), tdb.Identity())
    self.assertBuildsConst(('foo', 'foo'), block, 'foo')
    block = (tdb.AllOf(tdb.Identity(), tdb.Identity()) >>
             (tdb.Scalar(), tdb.Scalar()))
    self.assertBuilds((42.0, 42.0), block, 42, max_depth=None)

  def test_function_composition(self):
    sc = tdb.Scalar()
    fn1 = times_scalar_block(2.0)
    c = sc >> fn1
    self.assertBuilds(42., c, 21)

  def test_function_composition_with_block(self):
    c = tdb.Composition()
    with c.scope():
      scalar = tdb.Scalar().reads(c.input)
      c.output.reads(times_scalar_block(2.0).reads(scalar))
    self.assertBuilds(42., c, 21)

  def test_function_tuple_in_out(self):
    # f(a, (b, c)) := ((c, a), b)
    b = ((tdb.Vector(1), (tdb.Vector(2), tdb.Vector(3))) >>
         tdb.Function(lambda x, y: ((y[1], x), y[0])))
    self.assertBuilds((([4., 5., 6.], [1.]), [2., 3.]), b,
                      ([1], ([2, 3], [4, 5, 6])))

  def test_function_raises(self):
    self.assertRaisesWithLiteralMatch(
        TypeError, 'tf_fn is not callable: 42', tdb.Function, 42)

  def test_composition_diamond(self):
    sc = tdb.Scalar()
    fn1 = times_scalar_block(2.0)
    fn2 = times_scalar_block(3.0)
    fn3 = tdb.Function(tf.add)

    # out = in*2 + in*3
    c = tdb.Composition([sc, fn1, fn2, fn3])
    c.connect(c.input, sc)
    c.connect(sc, fn1)
    c.connect(sc, fn2)
    c.connect((fn1, fn2), fn3)
    c.connect(fn3, c.output)

    self.assertBuilds(25., c, 5, max_depth=2)

  def test_composition_diamond_with_block(self):
    # out = in*2 + in*3
    c = tdb.Composition()
    with c.scope():
      scalar = tdb.Scalar().reads(c.input)
      fn1 = times_scalar_block(2.0).reads(scalar)
      fn2 = times_scalar_block(3.0).reads(scalar)
      c.output.reads(tdb.Function(tf.add).reads(fn1, fn2))

    self.assertBuilds(25., c, 5, max_depth=2)

  def test_composition_nested(self):
    fn1 = times_scalar_block(2.0)
    fn2 = times_scalar_block(3.0)
    c = tdb.Composition([fn1, fn2])
    c.connect(c.input, fn1)
    c.connect(c.input, fn2)
    c.connect((fn1, fn2), c.output)
    c2 = tdb.Scalar() >> c >> tdb.Function(tf.add)
    self.assertBuilds(5.0, c2, 1.0, max_depth=2)

  def test_composition_toposort(self):
    fn0 = tdb.Scalar()
    fn1 = times_scalar_block(2.0)
    fn2 = times_scalar_block(3.0)
    fn3 = times_scalar_block(1.0)
    fn4 = tdb.Function(tf.add)

    c = tdb.Composition([fn4, fn3, fn0, fn2, fn1])
    c.connect(c.input, fn0)
    c.connect(fn0, fn1)
    c.connect(fn0, fn2)
    c.connect(fn2, fn3)
    c.connect((fn1, fn3), fn4)
    c.connect(fn4, c.output)
    self.assertBuilds(5.0, c, 1.0, max_depth=3)

  def test_composition_toposort_output(self):
    block = tdb.Composition()
    with block.scope():
      s = tdb.Scalar('int32').reads(block.input)
      block.output.reads(s, s)
    self.assertBuildsConst((3, 3), block, 3)

  def test_composition_connect_raises(self):
    self.assertRaises(TypeError, tdb.Pipe, tdb.Scalar(), tdb.Concat())

  def test_composition_raises_cycle(self):
    fn1 = times_scalar_block(2.0)
    fn2 = times_scalar_block(3.0)
    c = tdb.Composition([fn2, fn1]).set_input_type(tdt.VoidType())
    c.connect(fn1, fn2)
    c.connect(fn2, c.output)
    c.connect(fn2, fn1)   # cycle
    self.assertRaisesWithLiteralMatch(
        ValueError, 'Composition cannot have cycles.', c._validate, None)

  def test_composition_void(self):
    c = tdb.Composition()
    with c.scope():
      a = tdb.Scalar().reads(c.input)
      b = tdb.Function(tf.negative).reads(a)
      tdm.Metric('foo').reads(b)
      c.output.reads(a)
    self.assertBuilds((42., {'foo': [-42.]}), c, 42, max_depth=2)

  def test_composition_raises_no_output(self):
    c = tdb.Composition()
    six.assertRaisesRegex(
        self, TypeError, 'Composition block has no output', c._validate, None)

  def test_composition_no_output_void_type(self):
    b = tdb.AllOf(tdb.Void(), tdb.Scalar()) >> tdb.GetItem(1)
    self.assertBuildsConst(42., b, 42)

  def test_composition_rasies_read_output(self):
    a = tdb.Scalar()
    c = tdb.Composition([a])
    self.assertRaisesWithLiteralMatch(
        ValueError, 'cannot read from composition output',
        c.connect, c.output, a)

  def test_composition_rasies_write_input(self):
    a = tdb.Scalar()
    c = tdb.Composition([a])
    self.assertRaisesWithLiteralMatch(
        ValueError, 'cannot write to composition input', c.connect, a, c.input)

  def test_composition_rasies_foreign_io(self):
    a = tdb.Scalar()
    c = tdb.Composition([a])
    c2 = tdb.Composition()
    six.assertRaisesRegex(
        self, ValueError, 'is the input or output of a different composition',
        c.connect, c2.input, a)

  def test_composition_raises_unused(self):
    fn0 = tdb.Scalar()
    fn1 = times_scalar_block(2.0)
    c = tdb.Composition([fn1, fn0])
    c.connect(c.input, fn0)
    c.connect(fn0, fn1)
    c.connect(fn0, c.output)
    six.assertRaisesRegex(
        self, TypeError, 'children have unused outputs: .*', c._validate, None)

  def test_composition_raises_double_connect(self):
    a = tdb.Scalar()
    c = tdb.Composition([a])
    c.connect(c.input, a)
    self.assertRaisesWithLiteralMatch(
        ValueError,
        'input of block is already connected: <td.Scalar dtype=\'float32\'>',
        c.connect, c.input, a)

  def test_composition_raises_double_connect_output(self):
    a = tdb.Scalar()
    b = tdb.Scalar()
    c = tdb.Composition([a, b])
    c.connect(a, c.output)
    self.assertRaisesWithLiteralMatch(
        ValueError,
        'input of block is already connected: <td.Composition.output>',
        c.connect, b, c.output)

  def test_composition_nested_with_block(self):
    c1 = tdb.Composition()
    with c1.scope():
      scalar = tdb.Scalar().reads(c1.input)
      c2 = tdb.Composition().reads(scalar)
      with c2.scope():
        fn1 = times_scalar_block(2.0).reads(c2.input)
        fn2 = times_scalar_block(3.0).reads(c2.input)
        c2.output.reads(fn1, fn2)
      c1.output.reads(tdb.Function(tf.add).reads(c2))
    self.assertBuilds(5.0, c1, 1.0, max_depth=2)

  def test_composition_slice(self):
    c1 = tdb.Composition().set_input_type(tdt.VoidType())
    with c1.scope():
      t = tdb.AllOf(*[np.array(t) for t in range(5)]).reads(c1.input)
      c1.output.reads(tdb.Function(tf.add).reads(t[1:-1:2]))
    self.assertBuilds(4, c1, None, max_depth=1)

  def test_composition_backward_type_inference(self):
    b = tdb.Map(tdb.Identity()) >> tdb.Identity() >> tdb.Identity()
    six.assertRaisesRegex(
        self, TypeError, 'bad output type VoidType',
        b.output.set_output_type, tdt.VoidType())

  def test_composition_forward_type_inference(self):
    b = tdb.Identity() >> tdb.Identity() >> tdb.Map(tdb.Function(tf.negative))
    six.assertRaisesRegex(
        self, TypeError, 'bad input type PyObjectType',
        b.input.set_input_type, tdt.PyObjectType())

  def test_map_pyobject_type_inference(self):
    b = tdb.Map(tdb.Identity()) >> tdb.Vector(2)
    self.assertBuildsConst([1., 2.], b, [1, 2])

  def test_function_otype_inference_tensor_to_tensor(self):
    infer = tdb._infer_tf_output_type_from_input_type

    self.assertEqual(tdt.TensorType([]),
                     infer(tf.negative, tdt.TensorType([])))
    self.assertEqual(tdt.TensorType([2, 3]),
                     infer(tf.negative, tdt.TensorType([2, 3])))

    self.assertEqual(tdt.TensorType([], 'int32'),
                     infer(tf.negative, tdt.TensorType([], 'int32')))
    self.assertEqual(tdt.TensorType([2, 3], 'int32'),
                     infer(tf.negative, tdt.TensorType([2, 3], 'int32')))

    f = lambda x: tf.cast(x, 'int32')
    self.assertEqual(tdt.TensorType([], 'int32'),
                     infer(f, tdt.TensorType([], 'float32')))
    self.assertEqual(tdt.TensorType([2, 3], 'int32'),
                     infer(f, tdt.TensorType([2, 3], 'float64')))

  def test_function_otype_inference_tuple_to_tensor(self):
    infer = tdb._infer_tf_output_type_from_input_type
    f = tf.matmul
    self.assertEqual(tdt.TensorType([1, 1]), infer(
        f, tdt.TupleType(tdt.TensorType([1, 1]), tdt.TensorType([1, 1]))))
    self.assertEqual(tdt.TensorType([3, 5]), infer(
        f, tdt.TupleType(tdt.TensorType([3, 2]), tdt.TensorType([2, 5]))))

  def test_function_otype_inference_tuple_to_tuple(self):
    infer = tdb._infer_tf_output_type_from_input_type
    def f(x, y):
      return [tf.matmul(x, y), tf.placeholder('int32', [None, 42])]
    self.assertEqual(
        tdt.TupleType(tdt.TensorType([1, 1]), tdt.TensorType([42], 'int32')),
        infer(f, tdt.TupleType(tdt.TensorType([1, 1]), tdt.TensorType([1, 1]))))
    self.assertEqual(
        tdt.TupleType(tdt.TensorType([3, 5]), tdt.TensorType([42], 'int32')),
        infer(f, tdt.TupleType(tdt.TensorType([3, 2]), tdt.TensorType([2, 5]))))

  def test_function_otype_inference_raises(self):
    def infer(result):
      itype = tdt.TensorType([])
      f = lambda _: result
      return tdb._infer_tf_output_type_from_input_type(f, itype)
    self.assertRaisesWithLiteralMatch(
        TypeError, '42 is not a TF tensor', infer, 42)
    six.assertRaisesRegex(
        self, TypeError, 'unspecified rank', infer, tf.placeholder('float32'))
    six.assertRaisesRegex(
        self, TypeError, 'expected a batch tensor, saw a scalar', infer,
        tf.placeholder('float32', []))
    six.assertRaisesRegex(
        self, TypeError, r'leading \(batch\) dimension should be None', infer,
        tf.placeholder('float32', [0, 2]))
    six.assertRaisesRegex(
        self, TypeError, 'instance shape is not fully defined', infer,
        tf.placeholder('float32', [None, 42, None, 5]))

  def test_record(self):
    d = tdb.Record(collections.OrderedDict([('b', tdb.Scalar()),
                                            ('a', tdb.Scalar())]))
    c = d >> tdb.Function(tf.subtract)
    self.assertBuilds(4.0, c, {'a': 1.0, 'b': 5.0})

  def test_record_one_child(self):
    self.assertBuildsConst((42,), tdb.Record({0: tdb.Scalar('int32')}), {0: 42})

  def test_record_composition(self):
    d = tdb.Record({'a': tdb.Scalar(), 'b': tdb.Scalar()})
    fn1 = times_scalar_block(2.0)
    fn2 = times_scalar_block(3.0)
    fn3 = tdb.Function(tf.add)

    c = tdb.Composition([d, fn1, fn2, fn3])
    c.connect(c.input, d)
    c.connect(d['a'], fn1)
    c.connect(d['b'], fn2)
    c.connect((fn1, fn2), fn3)
    c.connect(fn3, c.output)

    self.assertBuilds(17.0, c, {'a': 1.0, 'b': 5.0}, max_depth=2)

  def test_record_tuple(self):
    block = (tdb.AllOf(tdb.Scalar(), tdb.OneHot(3, dtype='int32')) >>
             (tdb.Function(tf.square), tdb.Function(tf.negative)))
    self.assertBuilds((4., [0, 0, -1]), block, 2)

  def test_record_raises(self):
    six.assertRaisesRegex(
        self, RuntimeError,
        'created with an unordered dict cannot take ordered',
        tdb.Pipe, (tdb.Scalar(), tdb.Scalar()),
        {'a': tdb.Identity(), 'b': tdb.Identity()})

  def test_record_slice_key(self):
    b = tdb.Record([
        (0, tdb.Scalar()),
        (slice(1, 3), (tdb.Scalar(), tdb.Scalar()) >> tdb.Concat())])
    self.assertBuilds((1., [2., 3.]), b, [1, 2, 3])

  def test_forward_declarations(self):
    # Define a simple expression data structure
    nlit = lambda x: {'op': 'lit', 'val': x}
    nadd = lambda x, y: {'op': 'add', 'left': x, 'right': y}
    nexpr = nadd(nadd(nlit(3.0), nlit(5.0)), nlit(2.0))

    # Define a recursive block using forward declarations
    expr_fwd = tdb.ForwardDeclaration(tdt.PyObjectType(),
                                      tdt.TensorType((), 'float32'))
    lit_case = tdb.GetItem('val') >> tdb.Scalar()
    add_case = (tdb.Record({'left': expr_fwd(), 'right': expr_fwd()})
                >> tdb.Function(tf.add))
    expr = tdb.OneOf(lambda x: x['op'], {'lit': lit_case, 'add': add_case})
    expr_fwd.resolve_to(expr)

    self.assertBuilds(10.0, expr, nexpr, max_depth=2)

  def test_forward_declaration_orphaned(self):
    fwd = tdb.ForwardDeclaration(tdt.VoidType(), tdt.TensorType([]))
    b = tdb.AllOf(fwd(), fwd()) >> tdb.Sum()
    fwd.resolve_to(tdb.FromTensor(tf.ones([])))
    self.assertBuilds(2., b, None)

  def test_forward_declaration_orphaned_nested(self):
    fwd1 = tdb.ForwardDeclaration(tdt.VoidType(), tdt.TensorType([]))
    fwd2 = tdb.ForwardDeclaration(tdt.SequenceType(tdt.TensorType([])),
                                  tdt.TensorType([]))
    b = tdb.Map(tdb.Scalar()) >> fwd2() >> tdb.Function(tf.negative)
    fwd2.resolve_to(tdb.Fold(tdb.Function(tf.add), fwd1()))
    fwd1.resolve_to(tdb.FromTensor(tf.ones([])))
    self.assertBuilds(-8., b, [3, 4], max_depth=3)

  def test_map(self):
    block = tdb.Map(tdb.Scalar() >> tdb.Function(tf.abs))
    self.assertBuilds([], block, [], max_depth=0)
    self.assertBuilds([1.], block, [-1])
    self.assertBuilds([1., 2., 3.], block, [-1, -2, -3])

  def test_map_const(self):
    block = tdb.Map(tdb.Scalar())
    self.assertBuildsConst([], block, [])
    self.assertBuildsConst([1.], block, [1])
    self.assertBuildsConst([1., 2., 3.], block, [1, 2, 3])

  def test_map_map(self):
    block = tdb.Map(tdb.Map(tdb.Scalar() >> tdb.Function(tf.abs)))
    self.assertBuilds([[]], block, [[]], max_depth=0)
    self.assertBuilds([[1., 2., 3.], [4., 5.], [6.], []], block,
                      [[-1, -2, -3], [-4, -5], [-6], []])

  def test_map_map_const(self):
    block = tdb.Map(tdb.Map(tdb.Scalar()))
    self.assertBuildsConst([[]], block, [[]])
    self.assertBuildsConst([[1., 2., 3.], [4., 5.], [6.], []], block,
                           [[1, 2, 3], [4, 5], [6], []])

  def test_map_tuple(self):
    block = (tdb.Scalar(), tdb.Scalar()) >> tdb.Map(tdb.Function(tf.negative))
    self.assertBuilds([-3., -4.], block, (3, 4))

  def test_fold(self):
    const_ten = np.array(10.0, dtype='float32')
    ten_plus_sum = (tdb.Map(tdb.Scalar()) >>
                    tdb.Fold(tdb.Function(tf.add), const_ten))
    self.assertBuilds(16.0, ten_plus_sum, [1.0, 2.0, 3.0], max_depth=3)
    self.assertBuilds(16.0, ten_plus_sum, [3.0, 2.0, 1.0], max_depth=3)
    self.assertBuilds(20.0, ten_plus_sum, [1.0, 2.0, 3.0, 4.0], max_depth=4)
    self.assertBuilds(20.0, ten_plus_sum, [4.0, 3.0, 2.0, 1.0], max_depth=4)

  def test_fold_tuple(self):
    block = ((tdb.Scalar(), tdb.Scalar()) >>
             tdb.Fold(tdb.Function(tf.add), tf.ones([])))
    self.assertBuilds(6., block, (2, 3), max_depth=2)

  def test_fold_pyobject(self):
    block = tdb.Fold((tdb.Identity(), tdb.Scalar()) >> tdb.Sum(), tdb.Zeros([]))
    self.assertBuilds(5., block, (2, 3), max_depth=None)

  def test_rnn(self):
    # We have to expand_dims to broadcast x over the batch.
    def f(x, st):
      return (tf.multiply(x, x), tf.add(st, tf.expand_dims(x, 1)))

    intup = (tdb.Map(tdb.Scalar()), tdb.Vector(2))
    block = intup >> tdb.RNN(tdb.Function(f), initial_state_from_input=True)
    self.assertBuilds(([], [0.0, 0.0]), block,
                      ([], [0.0, 0.0]), max_depth=0)
    self.assertBuilds(([1.0, 4.0, 9.0, 16.0], [10.0, 10.0]), block,
                      ([1.0, 2.0, 3.0, 4.0], [0.0, 0.0]), max_depth=4)
    self.assertBuilds(([1.0, 4.0, 9.0, 16.0], [10.0, 10.0]), block,
                      ([1.0, 2.0, 3.0, 4.0], [0.0, 0.0]), max_depth=4)

  def test_reduce(self):
    const_ten = np.array(10.0, dtype='float32')
    sum_or_ten = (tdb.Map(tdb.Scalar()) >>
                  tdb.Reduce(tdb.Function(tf.add), const_ten))
    self.assertBuilds(10.0, sum_or_ten, [], max_depth=0)
    self.assertBuilds(3.0, sum_or_ten, [1.0, 2.0], max_depth=1)
    self.assertBuilds(20.0, sum_or_ten, [2.0, 4.0, 6.0, 8.0], max_depth=2)
    self.assertBuilds(6.0, sum_or_ten, [1.0, 2.0, 3.0], max_depth=2)
    self.assertBuilds(21.0, sum_or_ten, range(7), max_depth=3)

  def test_reduce_default_zero(self):
    sum_or_ten = (tdb.Map(tdb.Scalar()) >> tdb.Reduce(tdb.Function(tf.add)))
    self.assertBuilds(0.0, sum_or_ten, [], max_depth=0)
    self.assertBuilds(3.0, sum_or_ten, [1.0, 2.0], max_depth=1)
    self.assertBuilds(20.0, sum_or_ten, [2.0, 4.0, 6.0, 8.0], max_depth=2)
    self.assertBuilds(6.0, sum_or_ten, [1.0, 2.0, 3.0], max_depth=2)
    self.assertBuilds(21.0, sum_or_ten, range(7), max_depth=3)

  def test_sum_2vectors(self):
    sum_2vectors = tdb.Map(tdb.Vector(2)) >> tdb.Sum()
    self.assertBuilds([3.0, 12.0], sum_2vectors,
                      [[1.0, 5.0], [2.0, 3.0], [0.0, 4.0]], max_depth=2)

  def test_min_2vectors(self):
    min_2vectors = tdb.Map(tdb.Vector(2)) >> tdb.Min()
    self.assertBuilds([0.0, 3.0], min_2vectors,
                      [[1.0, 5.0], [2.0, 3.0], [0.0, 4.0]], max_depth=2)

  def test_max_2vectors(self):
    max_2vectors = tdb.Map(tdb.Vector(2)) >> tdb.Max()
    self.assertBuilds([2.0, 5.0], max_2vectors,
                      [[1.0, 5.0], [2.0, 3.0], [0.0, 4.0]], max_depth=2)

  def test_mean_2vector(self):
    mean_2vectors = tdb.Map(tdb.Vector(2)) >> tdb.Mean()
    self.assertBuilds([1.0, 4.0], mean_2vectors,
                      [[1.0, 5.0], [2.0, 3.0], [0.0, 4.0]], max_depth=3)

  def test_mean_matrix(self):
    to_2_by_3_matrix = tdb.Tensor(shape=[2, 3], name='2by3Matrix')
    mean_matrix = tdb.Map(to_2_by_3_matrix) >> tdb.Mean()
    six_range = np.reshape(range(6), (2, 3))
    self.assertBuilds([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], mean_matrix,
                      [six_range+1.0, six_range, six_range+2.0],
                      max_depth=3)

  def test_mean_tuple(self):
    block = (tdb.Scalar(), tdb.Scalar(), tdb.Scalar()) >> tdb.Mean()
    self.assertBuildsConst(2., block, [0, 0, 6])

  def test_all_of_0(self):
    self.assertBuilds(None, scalar_all_of(), -3, max_depth=0)

  def test_all_of_1(self):
    self.assertBuilds((3.,), scalar_all_of(tf.negative), -3)

  def test_all_of_2(self):
    self.assertBuilds((-3., 3.), scalar_all_of(tf.identity, tf.abs), -3)

  def test_all_of_3(self):
    block = scalar_all_of(tf.identity, tf.abs, tf.negative)
    self.assertBuilds((3., 3., -3.), block, 3)
    self.assertBuilds((-3., 3., 3.), block, -3)

  def test_all_of_different_shapes(self):
    block = scalar_all_of(tf.negative, functools.partial(tf.expand_dims, dim=1))
    self.assertBuilds((-3., [3.]), block, 3)

  def test_seq_of_tuple(self):
    block = tdb.Map(scalar_all_of(tf.identity, tf.negative))
    self.assertBuilds([], block, [], max_depth=0)
    self.assertBuilds([(1., -1.), (2., -2.)], block, [1, 2])

  def test_tuple_of_seq(self):
    block = tdb.AllOf(
        tdb.Map(tdb.Scalar() >> tdb.Function(tf.negative)),
        tdb.Map(tdb.Scalar() >> tdb.Function(tf.identity)))
    self.assertBuilds(([], []), block, [], max_depth=0)
    self.assertBuilds(([-1., -2.], [1., 2.]), block, [1, 2])

  def test_input_transform(self):
    block = tdb.Map(tdb.InputTransform(lambda x: 1 + ord(x) - ord('a')) >>
                    tdb.Scalar('int32') >> tdb.Function(tf.negative))
    self.assertBuilds([-1, -2, -3, -4], block, 'abcd')

  def test_input_transform_const(self):
    block = tdb.Map(tdb.InputTransform(lambda x: 1 + ord(x) - ord('a')) >>
                    tdb.Scalar('int32'))
    self.assertBuildsConst([1, 2, 3, 4], block, 'abcd')

  def test_serialized_message_to_tree(self):
    block = tdb.SerializedMessageToTree('tensorflow.fold.Nested3')
    self.assertEqual(
        {'foo': 'x', 'nested2': {'bar': 'y', 'nested1': {'baz': None}}},
        block.eval(test_pb2.Nested3(
            foo='x', nested2=test_pb2.Nested2(
                bar='y', nested1=test_pb2.Nested1())).SerializeToString()))

  def test_serialized_message_to_tree_raises(self):
    self.assertRaisesWithLiteralMatch(
        TypeError, 'message type name must be a string; 42 has %s' % int,
        tdb.SerializedMessageToTree, 42)

  def test_get_item_pyobject(self):
    self.assertBuildsConst(2., tdb.GetItem(1) >> tdb.Scalar(), [1, 2, 3])

  def test_get_item_sequence(self):
    block = tdb.Map(tdb.Scalar()) >> tdb.GetItem(-1)
    self.assertBuildsConst(9., block, range(10))

  def test_get_item_tuple(self):
    block = (tdb.Scalar(), tdb.Scalar()) >> tdb.GetItem(-1)
    self.assertBuildsConst(2., block, (1, 2))

  def test_length(self):
    scalar_then_length = tdb.Map(tdb.Scalar('int32')) >> tdb.Length()
    scalar_then_int_length = (
        tdb.Map(tdb.Scalar(dtype='float64')) >> tdb.Length(dtype='int64'))
    self.assertBuilds(3.0, scalar_then_length, [1, 2, 3], max_depth=0)
    self.assertBuilds(10.0, scalar_then_length, range(10), max_depth=0)
    self.assertBuilds(3, scalar_then_int_length, [1, 2, 3], max_depth=0)

  def test_length_pyobject(self):
    self.assertBuilds(3, tdb.Length(dtype='int64'), [0, 1, 2], max_depth=0)
    self.assertBuilds(20, tdb.Length(dtype='int64'), range(20), max_depth=0)
    self.assertBuilds(4., tdb.Length(dtype='float32'), range(4), max_depth=0)

  def test_length_tuple(self):
    block = (tdb.Scalar(), tdb.Scalar()) >> tdb.Length(dtype='int32')
    self.assertBuildsConst(2, block, (0, 1))

  def test_length_empty_tuple(self):
    block = tdb.Record([]) >> tdb.Length(dtype='int32')
    self.assertBuildsConst(0, block, ())

  def test_slice_pyobject(self):
    self.assertBuildsConst('abc', tdb.Slice(), 'abc')
    self.assertBuildsConst('ab', tdb.Slice(stop=2), 'abc')
    self.assertBuildsConst('cba', tdb.Slice(step=-1), 'abc')

  def test_slice_sequence(self):
    self.assertBuildsConst(
        [0., 1., 2.], tdb.Map(tdb.Scalar()) >> tdb.Slice(), range(3))
    self.assertBuildsConst(
        [0., 1.], tdb.Map(tdb.Scalar()) >> tdb.Slice(stop=2), range(3))
    self.assertBuildsConst(
        [2., 1., 0.], tdb.Map(tdb.Scalar()) >> tdb.Slice(step=-1), range(3))

  def test_slice_tuple(self):
    self.assertBuildsConst(
        (1., 2.), (tdb.Scalar(), tdb.Scalar()) >> tdb.Slice(), (1, 2))
    self.assertBuildsConst(
        (), (tdb.Scalar(), tdb.Scalar()) >> tdb.Slice(stop=0), (1, 2))
    self.assertBuildsConst(
        (1,), (tdb.Scalar('int32'), tdb.Scalar()) >> tdb.Slice(stop=1), (1, 2))
    self.assertBuildsConst(
        (2., 1.), (tdb.Scalar(), tdb.Scalar()) >> tdb.Slice(step=-1), (1, 2))

  def test_slice_raises(self):
    six.assertRaisesRegex(
        self, TypeError, 'infinite sequence', tdb.Pipe,
        (tdb.Scalar() >> tdb.Broadcast()), tdb.Slice(stop=3))

  def test_one_of(self):
    block = tdb.OneOf(lambda x: x > 0,
                      {True: tdb.Scalar(),
                       False: tdb.Scalar() >> tdb.Function(tf.negative)})
    self.assertBuildsConst(3., block, 3)
    self.assertBuildsConst(3., block, -3)

  def test_one_of_pre(self):
    block = tdb.OneOf(lambda x: x['key'],
                      {'a': tdb.GetItem('bar') >> tdb.Scalar(),
                       'b': tdb.GetItem('baz') >> tdb.Scalar()},
                      tdb.GetItem('val'))
    self.assertBuildsConst(42., block,
                           {'key': 'a', 'val': {'bar': 42, 'baz': 0}})
    self.assertBuildsConst(0., block,
                           {'key': 'b', 'val': {'bar': 42, 'baz': 0}})

  def test_one_of_raises(self):
    six.assertRaisesRegex(
        self, TypeError, 'key_fn is not callable: 42',
        tdb.OneOf, 42, (tdb.Scalar(),))
    self.assertRaisesWithLiteralMatch(
        ValueError, 'case_blocks must be non-empty', tdb.OneOf, lambda x: x, {})
    six.assertRaisesRegex(
        self, TypeError, 'Type mismatch between output type',
        tdb.OneOf, lambda x: x, {0: tdb.Scalar(), 1: tdb.Vector(2)})

  def test_one_of_mixed_input_type(self):
    block = (tdb.Identity(), tdb.Scalar('int32')) >> tdb.OneOf(
        key_fn=tdb.GetItem(0),
        case_blocks=(tdb.Function(tf.square), tdb.Function(tf.negative)),
        pre_block=tdb.GetItem(1))
    self.assertBuilds(4, block, (0, 2))
    self.assertBuilds(-2, block, (1, 2))

  def test_optional(self):
    block = tdb.Optional(tdb.Vector(4))
    self.assertBuildsConst([1.0, 2.0, 3.0, 4.0], block, [1, 2, 3, 4])
    self.assertBuildsConst([0.0, 0.0, 0.0, 0.0], block, None)

    block2 = tdb.Optional(tdb.Scalar(), np.array(42.0, dtype='float32'))
    self.assertBuildsConst(6.0, block2, 6)
    self.assertBuildsConst(42.0, block2, None)

  def test_optional_default_none(self):
    block = tdb.Optional({'a': tdb.Map({'b': tdb.Scalar(), 'c': tdb.Scalar()}),
                          'd': tdb.Vector(3)})
    self.assertBuildsConst(([(0., 1.)], [2., 3., 4.]), block,
                           {'a': [{'b': 0, 'c': 1}], 'd': [2, 3, 4]})
    self.assertBuildsConst(([], [0., 0., 0.]), block, None)

  def test_optional_default_none_type_inference(self):
    child = tdb.Scalar() >> tdb.Function(tf.negative)
    block = tdb.Optional(child)
    self.assertEqual(child.output_type, None)
    child.set_output_type([])
    self.assertEqual(block.output_type, tdt.TensorType([]))

  def test_concat(self):
    block = {'a': tdb.Vector(1),
             'b': tdb.Vector(4),
             'c': tdb.Vector(1)} >> tdb.Concat()
    self.assertBuildsConst([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], block,
                           {'a': [1.0], 'b': [2.0, 3.0, 4.0, 5.0], 'c': [6.0]})

  def test_concat_scalar(self):
    block = {'a': tdb.Scalar(),
             'b': tdb.Vector(4),
             'c': tdb.Scalar()} >> tdb.Concat()
    self.assertBuildsConst([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], block,
                           {'a': 1.0, 'b': [2.0, 3.0, 4.0, 5.0], 'c': 6.0})

  def test_concat_dims(self):
    record = {'a': [[1.0, 2.0, 3.0]],
              'b': [[4.0, 5.0, 6.0]]}

    block = {'a': tdb.Tensor([1, 3]),
             'b': tdb.Tensor([1, 3])} >> tdb.Concat(concat_dim=0)
    self.assertBuildsConst([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]],
                           block, record)

    block = {'a': tdb.Tensor([1, 3]),
             'b': tdb.Tensor([1, 3])} >> tdb.Concat(concat_dim=1)
    self.assertBuildsConst([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                           block, record)

  def test_concat_nested(self):
    block = (tdb.AllOf(tdb.AllOf(tdb.Scalar(), tdb.Scalar()),
                       tdb.AllOf(tdb.Scalar(), tdb.Scalar())) >>
             tdb.Concat(flatten=True))
    self.assertBuildsConst([42.0] * 4, block, 42.0)

  def test_concat_raises(self):
    args = {'a': tdb.Scalar(),
            'b': tdb.Vector(4),
            'c': tdb.Tensor([3, 3])}
    six.assertRaisesRegex(
        self, TypeError, 'Shapes for concat don\'t match:',
        tdb.Pipe, args, tdb.Concat())

    args = {'a': tdb.Vector(2),
            'b': tdb.Vector(4)}
    six.assertRaisesRegex(
        self, TypeError, 'Concat argument.*has rank less than 2.',
        tdb.Pipe, args, tdb.Concat(concat_dim=1))

    args = {'a': tdb.Vector(2, dtype='int32'),
            'b': tdb.Vector(4)}
    six.assertRaisesRegex(
        self, TypeError,
        'Cannot concat tensors of different dtypes: int32 vs. float32',
        tdb.Pipe, args, tdb.Concat())

    args = ((tdb.Scalar(),), (tdb.Scalar(),))
    six.assertRaisesRegex(
        self, TypeError, 'contains nested tuples', tdb.Pipe, args, tdb.Concat())

    args = ()
    self.assertRaisesWithLiteralMatch(
        TypeError, 'Concat requires at least one tensor as input',
        tdb.Pipe, args, tdb.Concat())

  def test_broadcast(self):
    block = tdb.Scalar() >> tdb.Broadcast()
    self.assertBuildsConst(itertools.repeat(42.), block, 42)

  def test_broadcast_map(self):
    block = tdb.Scalar() >> tdb.Broadcast() >> tdb.Map(
        tdb.Function(tf.negative))
    self.assertBuilds(itertools.repeat(-42.), block, 42)

  def test_zip(self):
    block = {'x': tdb.Map(tdb.Scalar()),
             'y': tdb.Map(tdb.Scalar())} >> tdb.Zip()
    self.assertBuildsConst([(1., 3.)], block, {'x': [1, 2], 'y': [3]})

  def test_zip_some_broadcast(self):
    block = {'x': tdb.Map(tdb.Scalar()),
             'y': tdb.Scalar() >> tdb.Broadcast()} >> tdb.Zip()
    self.assertBuildsConst([(1., 3.), (2., 3.)], block, {'x': [1, 2], 'y': 3})

  def test_zip_all_broadcast(self):
    block = {'x': tdb.Scalar() >> tdb.Broadcast(),
             'y': tdb.Scalar() >> tdb.Broadcast()} >> tdb.Zip()
    self.assertBuildsConst(itertools.repeat((1., 2.)), block, {'x': 1, 'y': 2})

  def test_broadcast_zip_map(self):
    block = ({'x': tdb.Scalar() >> tdb.Broadcast(),
              'y': tdb.Map(tdb.Scalar())} >> tdb.Zip() >>
             tdb.Map(tdb.Function(tf.add)))
    self.assertBuilds([3., 4., 5.], block, {'x': 2, 'y': [1, 2, 3]})

  def test_zip_with(self):
    block = ((tdb.Map(tdb.Scalar()), tdb.Map(tdb.Scalar())) >>
             tdb.ZipWith(tdb.Function(tf.add)))
    self.assertBuilds([5., 7., 9.], block, ([1, 2, 3], [4, 5, 6]))

  def test_ngrams_1(self):
    block = tdb.Map(tdb.Scalar()) >> tdb.NGrams(1)
    self.assertBuildsConst([], block, [])
    self.assertBuildsConst([(1.,)], block, [1])
    self.assertBuildsConst([(1.,), (2.,)], block, [1, 2])

  def test_ngrams_2(self):
    block = tdb.Map(tdb.Scalar()) >> tdb.NGrams(2)
    self.assertBuildsConst([], block, [])
    self.assertBuildsConst([], block, [1.])
    self.assertBuildsConst([(1., 2.)], block, [1, 2])
    self.assertBuildsConst([(1., 2.), (2., 3.)], block, [1, 2, 3])

  def test_ngrams_broadscast(self):
    block = tdb.Scalar() >> tdb.Broadcast() >> tdb.NGrams(2)
    self.assertBuildsConst(itertools.repeat((42., 42.)), block, 42)

  def test_one_hot(self):
    self.assertBuildsConst([1., 0.], tdb.OneHot(2), 0)
    self.assertBuildsConst([0., 1.], tdb.OneHot(2), 1)
    self.assertBuildsConst([1., 0.], tdb.OneHot(100, 102), 100)
    self.assertBuildsConst([0., 1., 0.], tdb.OneHot(100, 103), 101)

  def test_one_hot_from_list(self):
    for strict in [True, False]:
      self.assertBuildsConst([1., 0.], tdb.OneHotFromList(
          [37, 23], strict=strict), 37)
      self.assertBuildsConst([0., 1.], tdb.OneHotFromList(
          [37, 23], strict=strict), 23)
      self.assertBuildsConst([0., 1., 0.], tdb.OneHotFromList(
          [5, 9, 6], strict=strict), 9)

    self.assertRaisesWithLiteralMatch(
        AssertionError, 'OneHotFromList was passed duplicate elements.',
        lambda: tdb.OneHotFromList([1, 3, 3], strict=True))
    self.assertRaisesWithLiteralMatch(
        AssertionError, 'OneHotFromList was passed duplicate elements.',
        lambda: tdb.OneHotFromList([1, 3, 3], strict=False))

    self.assertBuildsConst([0., 0.], tdb.OneHotFromList([37, 23], strict=False),
                           100)

    strict_one_hot = tdb.OneHotFromList([37, 23], strict=True)
    self.assertRaisesRegexp(
        KeyError, '',
        lambda: self.assertBuildsConst([0., 0.], strict_one_hot, 100))

  def test_output_type_inference(self):
    # Identity and composite compute their output types from input types.
    block = tdb.Scalar() >> (tdb.Identity() >> tdb.Identity())
    self.assertBuildsConst(42., block, 42)

    block = ({'a': tdb.Scalar(), 'b': tdb.Vector(2) >> tdb.Identity()} >>
             tdb.Identity() >> (tdb.Identity() >> tdb.Identity()) >>
             tdb.Identity())
    self.assertBuildsConst((42., [5., 1.]), block, {'a': 42, 'b': [5, 1]})

  def test_output_type_raises(self):
    block = tdb.Identity() >> tdb.Identity()
    self.assertRaisesWithLiteralMatch(
        TypeError, 'Cannot determine input type for Identity.',
        block._validate, None)

  def test_nth(self):
    block = (tdb.Map(tdb.Scalar('int32')), tdb.Identity()) >> tdb.Nth()
    for n in xrange(5):
      self.assertBuildsConst(n, block, (range(5), n))

  def test_nth_raises(self):
    six.assertRaisesRegex(
        self, TypeError, 'Nth block takes 2 inputs',
        tdb.Pipe, (tdb.Scalar(), tdb.Scalar(), tdb.Scalar()), tdb.Nth())
    six.assertRaisesRegex(
        self, TypeError, 'first input to Nth must be a sequence',
        tdb.Pipe, (tdb.Scalar(), tdb.Scalar()), tdb.Nth())
    six.assertRaisesRegex(
        self, TypeError, 'second input to Nth must be a PyObject',
        tdb.Pipe, (tdb.Map(tdb.Scalar()), tdb.Map(tdb.Scalar())), tdb.Nth())

  def test_zeros_void(self):
    block = tdb.Zeros(tdt.TupleType(tdt.VoidType(), tdt.TensorType(())))
    self.assertBuildsConst((None, 0.0), block, None)

  def test_max_depth(self):
    self.assertEqual(0, tdb.Scalar().max_depth(42))
    block = (tdb.Map(tdb.Scalar()) >>
             tdb.Fold(tdb.Function(tf.add), tf.zeros([])))
    for i in xrange(5):
      self.assertEqual(i, block.max_depth(range(i)))

  def test_eval_py_object(self):
    block = tdb.InputTransform(str)
    self.assertBuildsConst('42', block, 42)

  def test_eval_void(self):
    block = tdb.Identity().set_input_type(tdt.VoidType())
    self.assertBuildsConst(None, block, None)

  def test_eval_empty_tuple(self):
    block = tdb.Record([])
    self.assertBuildsConst((), block, ())

  def test_max_depth_metrics(self):
    elem_block = tdb.Composition()
    with elem_block.scope():
      s = tdb.Scalar('int32').reads(elem_block.input)
      tdm.Metric('foo').reads(s)
      elem_block.output.reads(s)
    block = (tdb.Map(elem_block), tdb.Identity()) >> tdb.Nth()
    self.assertBuilds((31, {'foo': list(xrange(32))}), block, (range(32), -1))

  def test_rnn_with_cells(self):
    gru_cell1 = tdl.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=16), 'gru1')
    gru_cell2 = tdl.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=16), 'gru2')

    with tf.variable_scope('gru3') as vscope:
      gru_cell3 = tdl.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=16), vscope)

    lstm_cell = tdl.ScopedLayer(
        tf.contrib.rnn.BasicLSTMCell(num_units=16), 'lstm')

    gru1 = (tdb.InputTransform(lambda s: [ord(c) for c in s]) >>
            tdb.Map(tdb.Scalar('int32') >>
                    tdb.Function(tdl.Embedding(128, 8))) >>
            tdb.RNN(gru_cell1))

    gru2 = (tdb.InputTransform(lambda s: [ord(c) for c in s]) >>
            tdb.Map(tdb.Scalar('int32') >>
                    tdb.Function(tdl.Embedding(128, 8))) >>
            tdb.RNN(gru_cell2, initial_state=tf.ones(16)))

    gru3 = (tdb.InputTransform(lambda s: [ord(c) for c in s]) >>
            tdb.Map(tdb.Scalar('int32') >>
                    tdb.Function(tdl.Embedding(128, 8))) >>
            tdb.RNN(gru_cell3, initial_state=tdb.FromTensor(tf.ones(16))))

    lstm = (tdb.InputTransform(lambda s: [ord(c) for c in s]) >>
            tdb.Map(tdb.Scalar('int32') >>
                    tdb.Function(tdl.Embedding(128, 8))) >>
            tdb.RNN(lstm_cell))

    with self.test_session():
      gru1.eval('abcde')
      gru2.eval('ABCDE')
      gru3.eval('vghj')
      lstm.eval('123abc')

  def test_record_doc_example(self):
    # Test to make sure examples from the documentation compile.
    example_datum = {'id': 8,
                     'name': 'Joe Smith',
                     'location': (2.5, 7.0)}
    num_ids = 16
    embed_len = 16
    td = tdb
    char_rnn = (td.InputTransform(lambda s: [ord(c) for c in s]) >>
                td.Map(td.Scalar('int32') >>
                       td.Function(tdl.Embedding(128, 16))) >>
                td.Fold(td.Concat() >> td.Function(tdl.FC(32)),
                        td.FromTensor(tf.zeros(32))))
    r = (td.Record([('id', (td.Scalar('int32') >>
                            td.Function(tdl.Embedding(num_ids, embed_len)))),
                    ('name', char_rnn),
                    ('location', td.Vector(2))])
         >> td.Concat() >> td.Function(tdl.FC(256)))
    with self.test_session():
      r.eval(example_datum)

  def test_lstm_cell(self):
    # Test to make sure examples from the documentation compile.
    td = tdb
    num_hidden = 32

    # Create an LSTM cell, the hard way.
    lstm_cell = td.Composition()
    with lstm_cell.scope():
      in_state = td.Identity().reads(lstm_cell.input[1])
      bx = td.Concat().reads(lstm_cell.input[0], in_state[1])
      bi = td.Function(tdl.FC(num_hidden, tf.nn.sigmoid)).reads(bx)
      bf = td.Function(tdl.FC(num_hidden, tf.nn.sigmoid)).reads(bx)
      bo = td.Function(tdl.FC(num_hidden, tf.nn.sigmoid)).reads(bx)
      bg = td.Function(tdl.FC(num_hidden, tf.nn.tanh)).reads(bx)
      bc = td.Function(lambda c, i, f, g: c*f + i*g).reads(
          in_state[0], bi, bf, bg)
      by = td.Function(lambda c, o: tf.tanh(c) * o).reads(bc, bo)
      out_state = td.Identity().reads(bc, by)
      lstm_cell.output.reads(by, out_state)

    str_lstm = (td.InputTransform(lambda s: [ord(c) for c in s]) >>
                td.Map(td.Scalar('int32') >>
                       td.Function(tdl.Embedding(128, 16))) >>
                td.RNN(lstm_cell,
                       initial_state=td.AllOf(tf.zeros(32), tf.zeros(32))))

    with self.test_session():
      str_lstm.eval('The quick brown fox.')

  def test_hierarchical_rnn(self):
    char_cell = tdl.ScopedLayer(
        tf.contrib.rnn.BasicLSTMCell(num_units=16), 'char_cell')
    word_cell = tdl.ScopedLayer(
        tf.contrib.rnn.BasicLSTMCell(num_units=32), 'word_cell')

    char_lstm = (tdb.InputTransform(lambda s: [ord(c) for c in s]) >>
                 tdb.Map(tdb.Scalar('int32') >>
                         tdb.Function(tdl.Embedding(128, 8))) >>
                 tdb.RNN(char_cell))
    word_lstm = (tdb.Map(char_lstm >> tdb.GetItem(1) >> tdb.Concat()) >>
                 tdb.RNN(word_cell))

    with self.test_session():
      word_lstm.eval(['the', 'cat', 'sat', 'on', 'a', 'mat'])

  def test_eval_metrics(self):
    b = tdb.Map(tdb.Scalar() >> tdb.AllOf(tdm.Metric('x'), tdb.Identity()))
    self.assertBuilds(([(None, 1.), (None, 2.)], {'x': [1., 2.]}), b, [1, 2,])

  def test_eval_metrics_different_lengths(self):
    b = tdb.Record((tdb.Map(tdb.Scalar('int32') >> tdm.Metric('x')),
                    tdb.Map(tdb.Scalar() >> tdm.Metric('y'))))
    desired = ([None, None], [None]), {'x': [1, 2], 'y': [3.]}
    self.assertBuilds(desired, b, ([1, 2], [3]))

  def test_repr(self):
    goldens = {
        tdb.Tensor([]): '<td.Tensor dtype=\'float32\' shape=()>',
        tdb.Tensor([1, 2], 'int32', name='foo'):
        '<td.Tensor \'foo\' dtype=\'int32\' shape=(1, 2)>',

        tdb.Scalar('int64'): '<td.Scalar dtype=\'int64\'>',

        tdb.Vector(42): '<td.Vector dtype=\'float32\' size=42>',

        tdb.FromTensor(tf.zeros(3)): '<td.FromTensor \'zeros:0\'>',

        tdb.Function(tf.negative,
                     name='foo'): '<td.Function \'foo\' tf_fn=\'negative\'>',

        tdb.Identity(): '<td.Identity>',
        tdb.Identity('foo'): '<td.Identity \'foo\'>',

        tdb.InputTransform(ord): '<td.InputTransform py_fn=\'ord\'>',

        tdb.SerializedMessageToTree('foo'):
        '<td.SerializedMessageToTree \'foo\' '
        'py_fn=\'serialized_message_to_tree\'>',

        tdb.GetItem(3, 'mu'): '<td.GetItem \'mu\' key=3>',

        tdb.Length(): '<td.Length dtype=\'float32\'>',

        tdb.Slice(stop=2): '<td.Slice key=slice(None, 2, None)>',
        tdb.Slice(stop=2, name='x'):
        '<td.Slice \'x\' key=slice(None, 2, None)>',

        tdb.ForwardDeclaration(name='foo')():
        '<td.ForwardDeclaration() \'foo\'>',

        tdb.Composition(name='x').input: '<td.Composition.input \'x\'>',
        tdb.Composition(name='x').output: '<td.Composition.output \'x\'>',
        tdb.Composition(name='x'): '<td.Composition \'x\'>',

        tdb.Pipe(): '<td.Pipe>',
        tdb.Pipe(tdb.Scalar(), tdb.Identity()): '<td.Pipe>',

        tdb.Record({}, name='x'): '<td.Record \'x\' ordered=False>',
        tdb.Record((), name='x'): '<td.Record \'x\' ordered=True>',

        tdb.AllOf(): '<td.AllOf>',
        tdb.AllOf(tdb.Identity()): '<td.AllOf>',
        tdb.AllOf(tdb.Identity(), tdb.Identity()): '<td.AllOf>',

        tdb.AllOf(name='x'): '<td.AllOf \'x\'>',
        tdb.AllOf(tdb.Identity(), name='x'): '<td.AllOf \'x\'>',
        tdb.AllOf(tdb.Identity(), tdb.Identity(), name='x'): '<td.AllOf \'x\'>',

        tdb.Map(tdb.Scalar(), name='x'):
        '<td.Map \'x\' element_block=<td.Scalar dtype=\'float32\'>>',

        tdb.Fold(tdb.Function(tf.add), tf.ones([]), name='x'):
        '<td.Fold \'x\' combine_block=<td.Function tf_fn=\'add\'> '
        'start_block=<td.FromTensor \'ones:0\'>>',

        tdb.RNN(tdl.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=8))):
        '<td.RNN>',
        tdb.RNN(tdl.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=8)), name='x'):
        '<td.RNN \'x\'>',
        tdb.RNN(tdl.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=8)),
                initial_state=tf.ones(8)):
        '<td.RNN>',
        tdb.RNN(tdl.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=8)),
                initial_state=tf.ones(8), name='x'):
        '<td.RNN \'x\'>',

        tdb.Reduce(tdb.Function(tf.add), name='x'):
        '<td.Reduce \'x\' combine_block=<td.Function tf_fn=\'add\'>>',

        tdb.Sum(name='foo'):
        '<td.Sum \'foo\' combine_block=<td.Function tf_fn=\'add\'>>',

        tdb.Min(name='foo'):
        '<td.Min \'foo\' combine_block=<td.Function tf_fn=\'minimum\'>>',

        tdb.Max(name='foo'):
        '<td.Max \'foo\' combine_block=<td.Function tf_fn=\'maximum\'>>',

        tdb.Mean(name='foo'): '<td.Mean \'foo\'>',

        tdb.OneOf(ord, (tdb.Scalar(), tdb.Scalar()), name='x'):
        '<td.OneOf \'x\'>',

        tdb.Optional(tdb.Scalar(), name='foo'):
        '<td.Optional \'foo\' some_case_block=<td.Scalar dtype=\'float32\'>>',

        tdb.Concat(1, True, 'x'):
        '<td.Concat \'x\' concat_dim=1 flatten=True>',

        tdb.Broadcast(name='x'): '<td.Broadcast \'x\'>',

        tdb.Zip(name='x'): '<td.Zip \'x\'>',

        tdb.NGrams(n=42, name='x'): '<td.NGrams \'x\' n=42>',

        tdb.OneHot(2, 3, name='x'):
        '<td.OneHot \'x\' dtype=\'float32\' start=2 stop=3>',
        tdb.OneHot(3): '<td.OneHot dtype=\'float32\' start=0 stop=3>',

        tdb.OneHotFromList(['a', 'b']): '<td.OneHotFromList>',
        tdb.OneHotFromList(['a', 'b'], name='foo'):
        '<td.OneHotFromList \'foo\'>',

        tdb.Nth(name='x'): '<td.Nth \'x\'>',

        tdb.Zeros([], 'x'): '<td.Zeros \'x\'>',

        tdb.Void(): '<td.Void>',
        tdb.Void('foo'): '<td.Void \'foo\'>',

        tdm.Metric('foo'): '<td.Metric \'foo\'>'}
    for block, expected_repr in sorted(six.iteritems(goldens),
                                       key=lambda kv: kv[1]):
      self.assertEqual(repr(block), expected_repr)


def times_scalar_block(n):
  return tdb.Function(functools.partial(tf.multiply, n))


def scalar_all_of(*fns):
  return tdb.Scalar() >> tdb.AllOf(*[tdb.Function(f) for f in fns])

if __name__ == '__main__':
  test_lib.main()
