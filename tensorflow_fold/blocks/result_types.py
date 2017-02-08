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
"""Types library for TensorFlow Fold.  See README.md for further docs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import functools
import itertools
import numbers
# import google3
import numpy as np
import six
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.public import loom


@functools.total_ordering
class ResultType(object):
  """Base class for types that can be used as inputs/outputs to blocks."""

  def __str__(self):
    return repr(self)

  def __hash__(self):
    return repr(self).__hash__()

  def __eq__(self, other):
    return isinstance(other, ResultType) and repr(self) == repr(other)

  def __ne__(self, other):
    return not self.__eq__(other)

  def __le__(self, other):
    if not isinstance(other, ResultType):
      return NotImplementedError
    return repr(self) < repr(other)

  def __repr__(self):
    return '%s(%s)' % (type(self).__name__,
                       ', '.join(repr(x) for x in self._repr_args()))

  def _repr_args(self):
    return []

  @property
  def size(self):
    """Returns the total number of scalar elements in the type.

    Returns None if the size is not fixed -- e.g. for a variable-length
    sequence.
    """
    return None

  def terminal_types(self):
    """Returns an iterable of all terminal types in this type, in pre-order.

    Void is not considered to be a terminal type since no terminal values are
    needed to construct it. Instead, it has no terminal types.

    Returns:
      An iterable with the terminal types.
    """
    return (self,)

  # Traversal routines with default implementations for terminal types
  # (tensor/ pyobject / void). Must be overridden to handle
  # non-terminal types.

  def for_each_terminal(self, fn, instance):
    """Calls fn(terminal_type, value) for all terminal values in instance."""
    fn(self, instance)

  def unflatten(self, flat, unused_shaped_like):
    """Converts a iterator over terminal values to an instance of this type."""
    return next(flat)

  # Derived traversal routines, defined in terms of the above.

  def flatten(self, instance):
    """Converts an instance of this type to a flat list of terminal values."""
    items = []
    self.for_each_terminal(lambda _, item: items.append(item), instance)
    return items


class TensorType(ResultType):
  """Tensors (which may be numpy or tensorflow) of a particular shape.

  Tensor types implement the numpy array protocol, which means that
  e.g.  `np.ones_like(tensor_type)` will do what you expect it
  to. Calling `np.array(tensor_type)` returns a zeroed array.
  """

  def __init__(self, shape, dtype='float32'):
    """Creates a tensor type.

    Args:
      shape: A tuple or list of non-negative integers.
      dtype: A `tf.DType`, or stringified version thereof (e.g. `'int64'`).

    Raises:
      TypeError: If `shape` is not a tuple or list of non-negative integers.
      TypeError: If `dtype` cannot be converted to a TF dtype.
    """
    if not isinstance(shape, (tuple, list)):
      raise TypeError('shape must be a tuple or list: %s' % str(shape))
    self._type_shape = loom.TypeShape(dtype, shape)

  def _repr_args(self):
    return [self.shape, self.dtype]

  def __array__(self):
    """Returns a zeroed numpy array of this type."""
    return np.zeros(self.shape, self.dtype)

  @property
  def shape(self):
    return self._type_shape.shape

  @property
  def dtype(self):
    return self._type_shape.dtype

  @property
  def size(self):
    return np.prod(self.shape, dtype=np.int)

  @property
  def ndim(self):
    return len(self.shape)

  def flatten(self, instance):  # specialized for efficiency
    return [instance]


class VoidType(ResultType):
  """A type used for blocks that don't return inputs or outputs."""

  def terminal_types(self):
    return ()

  def for_each_terminal(self, unused_fn, unused_instance):
    pass

  def unflatten(self, unused_flat, unused_shaped_like):
    return None

  def flatten(self, unused_instance):
    return []


class PyObjectType(ResultType):
  """The type of an arbitrary python object (usually used as an input type)."""
  pass


class TupleType(ResultType, collections.Sequence):
  """Type for fixed-length tuples of items, each of a particular type.

  `TupleType` implements the sequence protocol, so e.g. `foo[0]` is
  the type of the first item, `foo[2:4]` is a `TupleType` with the
  expected item types, and `len(foo)` is the number of item types in
  the tuple.
  """

  def __init__(self, *item_types):
    """Creates a tuple type.

    Args:
      *item_types: A tuple of types or a single iterable of types.

    Raises:
      TypeError: If the items of `item_types` are not all types.
    """
    if len(item_types) == 1 and not isinstance(item_types[0], ResultType):
      item_types = tuple(item_types[0])
    for i, item_type in enumerate(item_types):
      if not isinstance(item_type, ResultType):
        raise TypeError('item_types[%s] is not a type: %s' % (i, item_type))
    self._item_types = item_types

  def _repr_args(self):
    return self._item_types

  def __len__(self):
    return len(self._item_types)

  def __getitem__(self, key):
    if isinstance(key, slice): return TupleType(self._item_types[key])
    return self._item_types[key]

  @property
  def size(self):
    try:
      return sum(item_type.size for item_type in self._item_types)
    except TypeError:
      return None

  def terminal_types(self):
    return itertools.chain.from_iterable(t.terminal_types() for t in self)

  def for_each_terminal(self, fn, instance):
    for t, i in zip(self, instance):
      t.for_each_terminal(fn, i)

  def unflatten(self, flat, shaped_like):
    if shaped_like is None:
      return tuple(t.unflatten(flat, None) for t in self)
    return tuple(t.unflatten(flat, i)
                 for t, i in zip(self, shaped_like))


class SequenceType(ResultType):
  """Type for variable-length sequences of elements all having the same type."""

  def __init__(self, elem_type):
    """Creates a sequence type.

    Args:
      elem_type: A type.

    Raises:
      TypeError: If `elem_type` is not a type.
    """
    if not isinstance(elem_type, ResultType):
      raise TypeError('%s is not a type' % str(elem_type))
    self._elem_type = elem_type

  def _repr_args(self):
    return [self._elem_type]

  @property
  def element_type(self):
    return self._elem_type

  def terminal_types(self):
    return self._elem_type.terminal_types()

  def for_each_terminal(self, fn, instance):
    rec = self._elem_type.for_each_terminal
    for i in instance:
      rec(fn, i)

  def unflatten(self, flat, shaped_like):
    rec = self._elem_type.unflatten
    return [rec(flat, i) for i in shaped_like]


class BroadcastSequenceType(SequenceType):
  """Type for infinite sequences of same element repeated."""

  def for_each_terminal(self, fn, instance):
    self._elem_type.for_each_terminal(fn, next(instance))

  def unflatten(self, flat, shaped_like):
    return itertools.repeat(self._elem_type.unflatten(flat, next(shaped_like)))


def convert_to_type(type_like):
  """Converts `type_like` to a `Type`.

  If `type_like` is already a `Type`, it is returned. The following
  conversions are performed:

  * Python tuples become `Tuple`s; items are recursively converted.

  * A `tf.TensorShape` becomes a corresponding `TensorType` with
  `dtype=float32`. Must be fully defined.

  * Lists of `shape + [dtype]` (e.g. `[3, 4, 'int32']`) become
  `TensorType`s, with the default `dtype=float32` if omitted.

  * A `tf.Dtype` or stringified version thereof (e.g. `'int64'`)
  becomes a corresponding scalar `TensorType((), dtype)`.

  * An integer `vector_len` becomes a corresponding vector
  `TensorType((vector_len,), dtype=float32)`.

  Args:
    type_like: Described above.

  Returns:
    A `Type`.

  Raises:
    TypeError: If `type_like` cannot be converted to a `Type`.

  """
  if isinstance(type_like, ResultType):
    return type_like
  if isinstance(type_like, tf.TensorShape):
    # Check this *before* calling as_list() otherwise it throws.
    if not type_like.is_fully_defined():
      raise TypeError('shape %s is not fully defined' % type_like)
    return TensorType(type_like.as_list())
  if isinstance(type_like, tuple):
    return TupleType(convert_to_type(item) for item in type_like)
  if isinstance(type_like, list):
    if type_like and isinstance(type_like[-1], six.string_types):
      return TensorType(type_like[:-1], dtype=type_like[-1])
    else:
      return TensorType(type_like)
  if isinstance(type_like, tf.DType) or isinstance(type_like, six.string_types):
    return TensorType((), dtype=type_like)
  if isinstance(type_like, numbers.Integral):
    return TensorType((type_like,))
  raise TypeError('Cannot covert %s to a type.' % (type_like,))


def canonicalize_type(type_like):
  """Returns a canonical representation of a type.

  Recursively applies a reduction rule that converts tuples/sequences
  of `PyObjectType` to a single terminal `PyObjectType`.

  ```python
  canonicalize_type((PyObjectType(), PyObjectType())) => PyObjectType()
  canonicalize_type(SequenceType(PyObjectType())) => PyObjectType()
  ```

  Args:
    type_like: A type or an object convertible to one by `convert_to_type`.

  Returns:
    A canonical representation of `type_like`.
  """
  def rec(typ):
    if isinstance(typ, TupleType) and typ:
      typ = TupleType(rec(t) for t in typ)
      if all(isinstance(t, PyObjectType) for t in typ): return PyObjectType()
    elif isinstance(typ, SequenceType):
      typ = type(typ)(rec(typ.element_type))
      if isinstance(typ.element_type, PyObjectType): return PyObjectType()
    return typ
  return rec(convert_to_type(type_like))


class _TypeVariable(object):
  """A type that may be fully specified, partially specified, or unknown."""

  def __init__(self, name, contained_by, expected, update):
    """Creates a type variable.

    Args:
      name: A string used in error messages; the name of the type variable.
      contained_by: The object that contains the type variable.
      expected: A tuple of allowed type classes, or None.
      update: A nullary function to call when the type becomes fully specified;
        guaranteed to be called at most once.
    """
    self._name = name
    self._contained_by = contained_by
    self._expected = expected
    self._update = update
    self._value = None

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, typ):
    typ = canonicalize_type(typ)
    if self.value is None:
      self.expected = (type(typ),)
      self._value = typ
      # Note that type-inference is monotonic; update() is only called
      # when current_type None, after a type has been assigned. This
      # guarantees termination.
      self._update()
    elif typ != self.value:
      raise TypeError(
          'Type mismatch between %s type %s and expected %s type %s in %s.' %
          (self._name, typ, self._name, self.value, self._contained_by))

  @property
  def expected(self):
    return self._expected

  @expected.setter
  def expected(self, type_classes):
    """Updates expected type classes to their intersection with type_classes."""
    if self.expected is None:
      self._expected = type_classes
      return
    expected = tuple(t for t in type_classes if issubclass(t, self._expected))
    if not expected:
      new_types_str = ' or '.join(x.__name__ for x in type_classes)
      old_types_str = ' or '.join(x.__name__ for x in self._expected)
      raise TypeError('bad %s type %s for %s, expected %s' % (
          self._name, new_types_str, self._contained_by, old_types_str))
    self._expected = expected


class IOBase(object):
  """Base class for objects with associated input/output types and names."""

  def __init__(self, input_type=None, output_type=None, name=None):
    if name is not None and not isinstance(name, six.string_types):
      raise TypeError('name must be a string: %s' % (name,))
    self._name = name
    def gen_update_and_propagate(update):
      """Returns a function that calls update(), then self.propagate_types()."""
      def update_and_propagate():
        update()
        self._propagate_types()
      return update_and_propagate
    self._input_type = _TypeVariable(
        'input', self, self._expected_input_types,
        gen_update_and_propagate(self._update_input_type))
    self._output_type = _TypeVariable(
        'output', self, self._expected_output_types,
        gen_update_and_propagate(self._update_output_type))
    self.set_input_type(input_type)
    self.set_output_type(output_type)

  # Subclasses can override with e.g. (TupleType,) or (TupleType, SequenceType)
  _expected_input_types = None
  _expected_output_types = None

  def __str__(self):
    return '<%s.%s>' % (type(self).__name__, self.name)

  @property
  def name(self):
    return self._name or type(self).__name__

  @property
  def input_type(self):
    """Returns the input type if known, else None."""
    return self._input_type.value

  @property
  def output_type(self):
    """Returns the output type if known, else None."""
    return self._output_type.value

  def set_input_type(self, input_type):
    """Updates the input type.

    Args:
      input_type: A type, or None.

    Returns:
      `self`

    Raises:
      TypeError: If `input_type` is not compatible with the current input type
        or its expected type classes.
    """
    if input_type is not None: self._input_type.value = input_type
    return self

  def set_input_type_classes(self, *input_type_classes):
    """Updates the type classes of the input type.

    Args:
      *input_type_classes: A tuple of type classes.

    Returns:
      `self`

    Raises:
      TypeError: If `input_type_classes` are not compatible with the current
        input type or its expected type classes.
    """
    self._input_type.expected = input_type_classes
    return self

  def _update_input_type(self):
    """Validates and propagates types from `self.input_type`."""
    pass

  def set_output_type(self, output_type):
    """Updates the output type.

    Args:
      output_type: A type, or None.

    Returns:
      `self`

    Raises:
      TypeError: If `output_type` is not compatible with the current output
        type.
    """
    if output_type is not None: self._output_type.value = output_type
    return self

  def set_output_type_classes(self, *output_type_classes):
    """Updates the type class of the output type.

    Args:
      *output_type_classes: A tuple of type classes.

    Returns:
      `self`

    Raises:
      TypeError: If `output_type_classes` are not compatible with the current
        output type or its expected type classes.
    """
    self._output_type.expected = output_type_classes
    return self

  def _update_output_type(self):
    """Validates and propagates types from `self.output_type`."""
    pass

  def _propagate_types(self):
    """Handler to call when input_type or output_type is assigned."""
    pass

  def set_io_types(self, other):
    """Updates input and output types of two `IOBase` objects to match.

    Args:
      other: An instance of IOBase.

    Returns:
      `self`

    Raises:
      TypeError: If the input/output types of self and other are incompatible.
    """
    other.set_input_type(self.input_type).set_output_type(self.output_type)
    self.set_input_type(other.input_type).set_output_type(other.output_type)
    return self

  def _check_input_type(self):
    if self.input_type is None:
      raise TypeError('Cannot determine input type for %s.' % self.name)

  def _check_output_type(self):
    if self.output_type is None:
      raise TypeError('Cannot determine output type for %s.' % self.name)
