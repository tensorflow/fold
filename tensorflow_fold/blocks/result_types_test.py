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
"""Tests for tensorflow_fold.blocks.types."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools
# import google3
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import result_types as tdt
from tensorflow_fold.blocks import test_lib
from tensorflow_fold.public import loom


class TypeTest(test_lib.TestCase):

  def assertHashes(self, constructor, *constructor_args):
    t1 = constructor(*constructor_args)
    t2 = constructor(*constructor_args)
    self.assertEqual(t1, t2)
    self.assertEqual(t1.__hash__(), t2.__hash__())
    self.assertEqual(set([t1, t2]), set([constructor(*constructor_args)]))


class VoidTypeTest(TypeTest):

  def test_hash(self):
    self.assertHashes(tdt.VoidType)

  def test_conversion(self):
    t = tdt.VoidType()
    self.assertEqual(repr(t), 'VoidType()')


class PyObjectTypeTest(TypeTest):

  def test_hash(self):
    self.assertHashes(tdt.PyObjectType)

  def test_conversion(self):
    t = tdt.PyObjectType()
    self.assertEqual(repr(t), 'PyObjectType()')


class TensorTypeTest(TypeTest):

  def test_hash(self):
    self.assertHashes(tdt.TensorType, (), 'int32')
    self.assertHashes(tdt.TensorType, (1, 2, 3))

  def test_conversion(self):
    t = tdt.TensorType((1, 2), 'int32')
    self.assertEqual(repr(t), 'TensorType((1, 2), \'int32\')')
    self.assertEqual(np.ones_like(t).dtype, np.int32)
    np.testing.assert_equal(np.ones_like(t), np.ones((1, 2)))
    self.assertEqual(t._type_shape, loom.TypeShape('int32', (1, 2)))

  def test_size(self):
    self.assertEqual(tdt.TensorType(()).size, 1)
    self.assertEqual(tdt.TensorType((1, 2, 3)).size, 6)
    self.assertEqual(tdt.TensorType((1, 0, 3)).size, 0)

  def test_flatten_unflatten(self):
    t = tdt.TensorType(())
    self.assertEqual(list(t.terminal_types()), [t])
    self.assertEqual(t.flatten(42), [42])
    self.assertEqual(t.unflatten(iter([42]), None), 42)


class TupleTypeTest(TypeTest):

  def test_hash(self):
    self.assertHashes(tdt.TupleType, tdt.TensorType(()))

  def test_conversion(self):
    t = tdt.TupleType(tdt.TensorType(()), tdt.TensorType((1,), 'int32'))
    self.assertEqual(
        repr(t),
        'TupleType(TensorType((), \'float32\'), TensorType((1,), \'int32\'))')
    self.assertEqual(repr(tdt.TupleType(tdt.TensorType(()))),
                     'TupleType(TensorType((), \'float32\'))')

  def test_tuple(self):
    t = tdt.TupleType(tdt.TensorType(()), tdt.TensorType((1,), 'int32'))
    self.assertEqual(tuple(t),
                     (tdt.TensorType(()), tdt.TensorType((1,), 'int32')))
    self.assertNotEqual(tuple(t), t)
    self.assertEqual(len(t), 2)
    self.assertEqual(t[0], tdt.TensorType(()))
    self.assertEqual(t[:], t)
    self.assertEqual(t[0:1], tdt.TupleType(tdt.TensorType(())))

  def test_size(self):
    scalar = tdt.TensorType(())
    vector3 = tdt.TensorType((3,))
    seq = tdt.SequenceType(scalar)
    self.assertEqual(tdt.TupleType().size, 0)
    self.assertEqual(tdt.TupleType(scalar).size, 1)
    self.assertEqual(tdt.TupleType(scalar, seq, scalar).size, None)
    self.assertEqual(tdt.TupleType(scalar, vector3, scalar).size, 5)

  def test_terminal_types(self):
    t0 = tdt.TensorType([])
    t1 = tdt.TensorType([1, 2])
    t = tdt.TupleType(tdt.TupleType(t0),
                      tdt.TupleType(tdt.TupleType(t1, t1), t0))
    self.assertEqual(list(t.terminal_types()), [t0, t1, t1, t0])

  def test_flatten_unflatten(self):
    instance = ([], [1], ([2], [3], ()), ((),))
    t = tdt.convert_to_type(instance)
    self.assertEqual(list(t.terminal_types()),
                     list(tdt.convert_to_type(([], [1], [2], [3]))))
    flat = t.flatten(instance)
    self.assertEqual(flat, [[], [1], [2], [3]])
    self.assertEqual(t.unflatten(iter(flat), None), instance)


class SequenceTypeTest(TypeTest):

  def test_hash(self):
    self.assertHashes(tdt.SequenceType, tdt.TensorType(()))

  def test_conversion(self):
    t = tdt.SequenceType(tdt.TensorType((1, 2)))
    self.assertEqual(repr(t), 'SequenceType(TensorType((1, 2), \'float32\'))')
    self.assertEqual(t.element_type, tdt.TensorType((1, 2)))

  def test_terminal_types(self):
    t = tdt.SequenceType(tdt.TupleType(tdt.TensorType([]), tdt.VoidType(),
                                       tdt.TupleType(tdt.PyObjectType())))
    t_elem = t.element_type
    self.assertEqual(list(t.terminal_types()),
                     [t_elem[0], t_elem[2][0]])

  def test_flatten_unflatten(self):
    instance = [([(1, 2), (3, 4)], 5)]
    t = tdt.SequenceType(
        tdt.TupleType(tdt.SequenceType(tdt.TupleType(tdt.TensorType([6]),
                                                     tdt.TensorType([7]))),
                      tdt.TensorType([8])))
    self.assertEqual(list(t.terminal_types()),
                     list(tdt.convert_to_type(([6], [7], [8]))))
    flat = t.flatten(instance)
    self.assertEqual(flat, [1, 2, 3, 4, 5])
    self.assertEqual(t.unflatten(iter(flat), [([(0, 0), (0, 0)], 0)]),
                     instance)


class BroadcastSequenceTypeTest(TypeTest):

  def test_flatten_unflatten(self):
    t = tdt.BroadcastSequenceType(tdt.SequenceType(tdt.TensorType([])))
    instance = itertools.repeat([1, 2])
    self.assertEqual(t.flatten(instance), [1, 2])
    unflat = t.unflatten(iter([1, 2]), itertools.repeat([0, 0]))
    self.assertTrue(isinstance(unflat, itertools.repeat))
    self.assertEqual(next(unflat), [1, 2])


class AsTypeTest(test_lib.TestCase):

  def assertConverts(self, type_like, desired_type):
    self.assertEqual(tdt.convert_to_type(type_like), desired_type)
    for i in xrange(4):
      self.assertEqual(tdt.convert_to_type((type_like,) * i),
                       tdt.TupleType((desired_type,) * i))

  def test_tensor_shape(self):
    self.assertConverts(tf.TensorShape([]), tdt.TensorType(()))
    self.assertConverts(tf.TensorShape([1]), tdt.TensorType((1,)))
    self.assertConverts(tf.TensorShape([1, 2]), tdt.TensorType((1, 2)))
    self.assertConverts(tf.TensorShape([1, 2, 3]), tdt.TensorType((1, 2, 3)))

  def test_default_dtype(self):
    self.assertConverts([], tdt.TensorType(()))
    self.assertConverts([1], tdt.TensorType((1,)))
    self.assertConverts([1, 2], tdt.TensorType((1, 2)))
    self.assertConverts([1, 2, 3], tdt.TensorType((1, 2, 3)))

  def test_explicit_dtype(self):
    self.assertConverts('int64', tdt.TensorType((), 'int64'))
    self.assertConverts(['int64'], tdt.TensorType((), 'int64'))
    self.assertConverts([1, 'int64'], tdt.TensorType((1,), 'int64'))
    self.assertConverts([1, 2, 'int64'], tdt.TensorType((1, 2), 'int64'))
    self.assertConverts([1, 2, 3, 'int64'], tdt.TensorType((1, 2, 3), 'int64'))

  def assertConvertsPairs(self, type_likes, desired_types):
    for first, desired_first in zip(type_likes, desired_types):
      for second, desired_second in zip(type_likes, desired_types):
        self.assertEqual(tdt.convert_to_type((first, second)),
                         tdt.TupleType(desired_first, desired_second))

  def test_pair(self):
    self.assertConvertsPairs(
        [[], [1], [1, 2], [1, 2, 3],
         'int64', ['int64'], [1, 'int64'], [1, 2, 'int64'], [1, 2, 3, 'int64']],
        [tdt.TensorType(()), tdt.TensorType((1,)), tdt.TensorType((1, 2)),
         tdt.TensorType((1, 2, 3,)), tdt.TensorType((), 'int64'),
         tdt.TensorType((), 'int64'), tdt.TensorType((1,), 'int64'),
         tdt.TensorType((1, 2), 'int64'), tdt.TensorType((1, 2, 3), 'int64')])


if __name__ == '__main__':
  test_lib.main()
