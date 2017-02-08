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
"""Core blocks for TensorFlow Fold."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import contextlib
import functools
import itertools
import threading
# import google3
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import loom_ops
from tensorflow_fold.blocks import result_types as tdt
import tensorflow_fold.blocks.block_compiler
import tensorflow_fold.blocks.layers
from tensorflow_fold.util import proto_tools


_InputWire = collections.namedtuple('_InputWire', ['block', 'index'])
# Reference to the output of one block, which can be wired into the
# input of another block.

_COMPOSITION_CONTEXT = threading.local()
_COMPOSITION_CONTEXT.current = None


# Once a `Block` is constructed, its `_validate` method must be called at least
# once, and then its `_compile` method must be called at least once before
# the block can be evaluated on any input.  The user doesn't need to touch this
# because `eval` and `td.Compiler` handle this.
#
# In the event that fold's type inference is insufficient to deduce
# what types you need your block's inputs and outputs to be you can
# manually call `set_input_type` or `set_output_type` on the blocks in
# question to resolve the situation.
#
# The job of `validate` is to make sure the block's `children` (if any) have
# compatible types and to infer the `input_type` and `output_type` of the
# current block.    See [ResultType](#td.ResultType) for more details on
# fold's type system.


class Block(tdt.IOBase):
  """Base class for all blocks.

  A `Block` is an object which maps a data-structure (or queued TensorFlow
  operations, depending on a the block's input type) into queued TensorFlow
  operations.  (Except for [InputTransform](#td.InputTransform) which maps from
  data-structure to data-structure.)

  When interacting with Fold you can debug your blocks
  by calling [`eval`](#td.Block.eval) inside of a TF session.  (This has high
  per-call overhead and is not recommended for long-running jobs.)

  The efficient way to evaluate a block repeatedly is to pass the root of a tree
  of blocks to a persistent [`td.Compiler`](#td.Compiler) object (note that
  [`eval`](#td.Block.eval) creates a compiler object behind the scenes.)
  """

  def __init__(self, children=None, input_type=None, output_type=None,
               name=None):
    # TODO(delesley): write test cases for error condition
    self._parent = None
    self._children = []
    self._constructor_name = None
    self._constructor_args = None
    if children is not None:
      for b in children:
        self._add_child(b)
    super(Block, self).__init__(input_type, output_type, name)
    for child in self.children:
      self._propagate_types_from_child(child)

  def __repr__(self):
    strs = ['td.%s' % (self._constructor_name or  type(self).__name__,)]
    if self._name: strs.append('%r' % self._name)
    for k, v in sorted(six.iteritems(self._repr_kwargs())):
      if isinstance(v, functools.partial): v = v.func
      if hasattr(v, '__name__'): v = v.__name__
      strs.append('%s=%r' % (k, v))
    return '<%s>' % ' '.join(strs)

  def _repr_kwargs(self):
    return {}

  def __str__(self):
    return repr(self)

  def set_constructor_name(self, constructor_name):
    """Sets the constructor name, used for repr() and str(). Returns `self`."""
    if not isinstance(constructor_name, six.string_types):
      raise TypeError('constructor name must be a string; %s' %
                      (constructor_name,))
    self._constructor_name = constructor_name
    return self

  def set_constructor_args(self, *constructor_args):
    """Sets the constructor arguments for pretty printing. Returns `self`."""
    self._constructor_args = constructor_args
    return self

  def _add_child(self, b):
    assert isinstance(b, Block)  # internal consistency check
    if b.parent is not None:
      raise TypeError('block %s is already a child of %s' % (b, b.parent))
    b._parent = self  # pylint:disable=protected-access
    self._children.append(b)

  def _propagate_types(self):
    # pylint:disable=protected-access
    if self.parent is not None: self.parent._propagate_types_from_child(self)

  def _propagate_types_from_child(self, unused_child):
    """Checks and propagates types when a child's type is first set."""
    pass

  def reads(self, *other):
    """Sets `self` to read its inputs from `other`.

    Args:
      *other: which blocks to make the current block read from.

    Returns:
      `self`

    Raises:
      AssertionError: if no composition scope has been entered.
    """

    if _COMPOSITION_CONTEXT.current is None:
      raise AssertionError(
          "Block's reads method was called outside of a Composition scope.")
    _COMPOSITION_CONTEXT.current.connect(other, self)
    return self

  def __rshift__(self, rhs):
    """Function composition; `(a >> b).eval(x) => b(a(x))`."""
    return Pipe(self, rhs)

  def __rrshift__(self, lhs):
    """Function composition; `(a >> b).eval(x) => b(a(x))`."""
    return Pipe(lhs, self)

  @property
  def parent(self):
    return self._parent

  @property
  def children(self):
    return self._children

  @property
  def is_forward_declaration_ref(self):
    return False

  def __getitem__(self, i):
    """Return a reference to the i^th output from this block."""
    if isinstance(i, slice):
      if _COMPOSITION_CONTEXT.current is None:
        raise AssertionError(
            'Block sliced attempted outside of a composition scope.')
      return GetItem(i).reads(self)
    return _InputWire(self, i)

  def _validate(self, compiler_ctx):
    """Check types, infer missing types, and validate composition.

    A block is responsible for validating its children.

    When possible, we type check incrementally, and will raise a type
    error as possible. The main case where type errors are raises at
    validation time is the Composition block, where some checks cannot
    be performed until the full wiring diagram is known, and others
    (i.e. checking for cycles) would be harder to implement correctly
    without the full wiring diagram.

    Args:
      compiler_ctx: a context object created by the compiler.

    """
    for child in self.children:
      child._validate(compiler_ctx)  # pylint:disable=protected-access
    self._check_input_type()
    self._check_output_type()

  def _compile(self, compiler_ctx):
    """Prepare to execute this block, by performing setup and optimization.

    A block is not responsible for compiling its children; the compiler will
    will traverse children.

    Args:
      compiler_ctx: a context object created by the compiler.
    """
    pass

  def _evaluate(self, eval_ctx, x):
    raise NotImplementedError('Block %s does not implement _evaluate.' %
                              str(self))

  def eval(self, inp, feed_dict=None, session=None, tolist=False,
           use_while_loop=True):
    """Evaluates this block on `inp` in a TF session.

    Intended for testing and interactive development. If there are any
    uninitialized variables, they will be initialized prior to evaluation.

    Args:
      inp: An input to the block.
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
      session: The TF session to be used. Defaults to the default session.
      tolist: A bool; whether to return (possibly nested) Python lists
        in place of NumPy arrays.
      use_while_loop: A bool; whether to use a `tf.while_loop` in evaluation
        (default) or to unroll the loop. Provided for testing and debugging,
        should not affect the result.

    Returns:
      The result of running the block. If `output_type` is tensor, then a
      NumPy array (or Python list, if `tolist` is true). If a tuple, then a
      tuple. If a sequence, then a list, or an instance of itertools.repeat
      in the case of an infinite sequence. If metrics are defined then `eval`
      returns a `(result, metrics)` tuple, where `metrics` is a dict mapping
      metric names to NumPy arrays.

    Raises:
      ValueError: If `session` is none and no default session is registered.
        If the block contains no TF tensors or ops then a session is not
        required.
    """
    # pylint: disable=protected-access
    return tensorflow_fold.blocks.block_compiler.Compiler._interactive(  # pylint: disable=line-too-long
        self)._eval(inp, feed_dict, session, tolist, use_while_loop)

  def max_depth(self, inp):
    """Returns the loom `max_depth` needed to evaluate `inp`.

    Like `eval`, this is a convenience method for testing and
    interactive development. It cannot be called after the TF graph
    has been finalized (nb. [`Compiler.max_depth`](#td.Compiler.max_depth)
    does not have this limitation).

    Args:
      inp: A well-formed input to this block.

    Returns:
      An int (see above).
    """
    # pylint: disable=protected-access
    return tensorflow_fold.blocks.block_compiler.Compiler._interactive(  # pylint: disable=line-too-long
        self).max_depth(inp)


class Tensor(Block):
  """A block that converts its input from a python object to a tensor."""

  def __init__(self, shape, dtype='float32', name=None):
    super(Tensor, self).__init__(input_type=tdt.PyObjectType(),
                                 output_type=tdt.TensorType(shape, dtype),
                                 name=name)
    self._dtype = np.dtype(self.output_type.dtype)
    if not shape and tf.as_dtype(dtype).is_integer:  # memoize scalar ints
      self._evaluate = self._evaluate_memoized

  def _repr_kwargs(self):
    kwargs = {'dtype': self.output_type.dtype}
    if self._constructor_name == 'Vector':
      kwargs['size'] = self.output_type.shape[0]
    elif self._constructor_name != 'Scalar':
      kwargs['shape'] = self.output_type.shape
    return kwargs

  def _evaluate_memoized(self, eval_ctx, x):
    array = np.asarray(x, self._dtype)
    return eval_ctx.memoize_constant(self, array.item(), array)

  def _evaluate(self, eval_ctx, x):
    return eval_ctx.constant(np.asarray(x, self._dtype))


def Scalar(dtype='float32', name=None):  # pylint: disable=invalid-name
  """A block that converts its input to a scalar."""
  return Tensor(shape=[], dtype=dtype, name=name).set_constructor_name(
      'Scalar')


def Vector(size, dtype='float32', name=None):  # pylint: disable=invalid-name
  """A block that converts its input to a vector."""
  return Tensor(shape=[size], dtype=dtype, name=name).set_constructor_name(
      'Vector')


class FromTensor(Block):
  """A block that returns a particular TF tensor or NumPy array."""

  def __init__(self, tensor, name=None):
    """Creates the block.

    Args:
      tensor: A TF tensor or variable with a complete shape, or a NumPy array.
      name: A string. Defaults to the name of `tensor` if it has one.

    Raises:
      TypeError: If `tensor` is not a TF tensor or variable or NumPy array.
      TypeError: If  `tensor` does not have a complete shape.

    """
    if isinstance(tensor, np.ndarray):
      tensor = tf.constant(tensor)
    elif not isinstance(tensor, (tf.Tensor, tf.Variable)):
      raise TypeError('%s is not a tensor or np.ndarray' % str(tensor))
    shape, dtype = tensor.get_shape(), tensor.dtype
    # Check this *before* calling as_list() otherwise it throws.
    if not shape.is_fully_defined():
      raise TypeError(
          'shape %s is not fully defined; call set_shape()' % shape)
    shape = shape.as_list()
    if name is None:
      try:  # undocumented, but tensor.name throws if tensor is unnamed
        name = tensor.name
        # undocumented, name can be unicode
        if six.PY2 and isinstance(name, unicode):
          name = name.encode('ascii', 'replace')
      except ValueError:
        pass
    self._tensor = tensor
    super(FromTensor, self).__init__(
        name=name, input_type=tdt.VoidType(),
        output_type=tdt.TensorType(shape, dtype))

  @property
  def tensor(self):
    return self._tensor

  def _compile(self, compiler_ctx):
    self._tensor_name = compiler_ctx.register_tensor(self._tensor, self.name)

  def _evaluate(self, eval_ctx, _):
    return eval_ctx.named_tensor(self._tensor_name)


class Function(Block):
  """A TensorFlow function, wrapped in a block.

  The TensorFlow function that is passed into a `Function` block must be a batch
  version of the operation you want.  This doesn't matter for things like
  element-wise addition `td.Function(tf.add)`, but if you, for example, want a
  `Function` block that multiplies matrices, you need to call
  `td.Function(tf.batch_matmul)`.  This is done for efficiency reasons, so that
  calls to the same function can be batched together naturally and take
  advantage of TensorFlow's parallelism.
  """

  def __init__(self, tf_fn, name=None, infer_output_type=True):
    """Creates a `Function` block.

    Args:
      tf_fn: The batch version of the TensorFlow function to be evaluated.
      name: An optional string name for the block. If present, must be a valid
        name for a TensorFlow scope.
      infer_output_type: A bool; whether or not to infer the output type of
        of the block by invoking `tf_fn` once on dummy placeholder. If False,
        you will probably need to call `set_output_type()` explicitly.
    """
    if not callable(tf_fn):
      raise TypeError('tf_fn is not callable: %s' % str(tf_fn))
    super(Function, self).__init__(name=name)
    self._tf_fn = tf_fn
    if _is_layer(self._tf_fn): self.set_io_types(self._tf_fn)
    self._infer_output_type = infer_output_type

  def _repr_kwargs(self):
    return dict(tf_fn=self.tf_fn)

  @property
  def tf_fn(self):
    return self._tf_fn

  _expected_input_types = (tdt.TupleType, tdt.TensorType)
  _expected_output_types = (tdt.TupleType, tdt.TensorType)

  def _update_input_type(self):
    if self.input_type.size is None:
      raise TypeError('function inputs must be tensors: %s' % self.input_type)
    if not list(self.input_type.terminal_types()):
      raise TypeError('functions must take at least one tensor input')
    if _is_layer(self.tf_fn): self.set_io_types(self.tf_fn)

  def _update_output_type(self):
    if self.output_type.size is None:
      raise TypeError('function outputs must be tensors: %s' % self.output_type)
    if not list(self.output_type.terminal_types()):
      raise TypeError('functions must return at least one tensor output')
    if _is_layer(self.tf_fn): self.set_io_types(self.tf_fn)

  def _validate(self, compiler_ctx):
    if _is_layer(self.tf_fn): self.set_io_types(self.tf_fn)
    self._check_input_type()
    if self._infer_output_type and (
        not _is_layer(self.tf_fn) or self.output_type is None):
      self.set_output_type(_infer_tf_output_type_from_input_type(
          self.tf_fn, self.input_type))
    self._check_output_type()

  def _compile(self, compiler_ctx):
    fop = loom_ops.FuncallOp(self.tf_fn, self.input_type, self.output_type)
    self._loom_op_id = compiler_ctx.register_op(fop, self.name)

  def _evaluate(self, eval_ctx, x):
    # loom always expects lists of inputs and outputs
    return self.output_type.unflatten(
        iter(eval_ctx.op(self._loom_op_id, self.input_type.flatten(x))), None)


class Identity(Block):
  """A block that merely returns its input."""

  def __init__(self, name=None):
    super(Identity, self).__init__(name=name)

  def _update_input_type(self):
    self.set_output_type(self.input_type)

  def _update_output_type(self):
    self.set_input_type(self.output_type)

  def _evaluate(self, unused_eval_ctx, x):
    return x


class InputTransform(Block):
  """A Python function, lifted to a block."""

  def __init__(self, py_fn, name=None):
    if not callable(py_fn):
      raise TypeError('py_fn is not callable: %s' % str(py_fn))
    self._py_fn = py_fn
    super(InputTransform, self).__init__(
        [], input_type=tdt.PyObjectType(), output_type=tdt.PyObjectType(),
        name=name)

  def _repr_kwargs(self):
    return dict(py_fn=self.py_fn)

  @property
  def py_fn(self):
    return self._py_fn

  def _evaluate(self, _, x):
    return self._py_fn(x)


def SerializedMessageToTree(message_type_name):  # pylint: disable=invalid-name
  """A block that turns serialized protobufs into nested Python dicts and lists.

  The block's input and output types are both `PyObjectType`.

  Args:
    message_type_name: A string; the full name of the expected message type.

  Returns:
    A dictionary of the message's values by fieldname, where the
    function renders repeated fields as lists, submessages via
    recursion, and enums as dictionaries whose keys are `name`,
    `index`, and `number`. Missing optional fields are rendered as
    `None`. Scalar field values are rendered as themselves.

  Raises:
    TypeError: If `message_type_name` is not a string.

  """
  if not isinstance(message_type_name, six.string_types):
    raise TypeError('message type name must be a string; %s has %s' %
                    (message_type_name, type(message_type_name)))
  return InputTransform(functools.partial(
      proto_tools.serialized_message_to_tree, message_type_name),
                        name=message_type_name).set_constructor_name(
                            'SerializedMessageToTree')


class GetItem(Block):
  """A block that calls Pythons getitem operator (i.e. [] syntax) on its input.

  The input type may be a PyObject, a Tuple, or a finite Sequence.

  ```python
  (GetItem(key) >> block).eval(inp) => block.eval(inp[key])
  ```

  Will raise a `KeyError` if applied to an input where the key cannot be found.
  """

  def __init__(self, key, name=None):
    self._key = key
    super(GetItem, self).__init__(name=name)

  def _repr_kwargs(self):
    return dict(key=self.key)

  @property
  def key(self):
    return self._key

  _expected_input_types = (tdt.PyObjectType, tdt.TupleType, tdt.SequenceType)

  def _update_input_type(self):
    if isinstance(self.input_type, tdt.BroadcastSequenceType):
      raise TypeError('cannot get an item from an infinite sequence: %s' %
                      self.input_type)
    if isinstance(self.input_type, tdt.TupleType):
      self.set_output_type(self.input_type[self._key])
    elif (isinstance(self.input_type, tdt.SequenceType) and
          not isinstance(self._key, slice)):
      self.set_output_type(self.input_type.element_type)
    else:  # PyObjectType or a slice of a sequence
      self.set_output_type(self.input_type)

  def _evaluate(self, _, x):
    return x[self._key]


class Length(Block):
  """A block that returns the length of its input."""

  def __init__(self, dtype='float32', name=None):
    super(Length, self).__init__(
        output_type=tdt.TensorType([], dtype), name=name)
    self._dtype = np.dtype(self.output_type.dtype)

  def _repr_kwargs(self):
    return dict(dtype=self.output_type.dtype)

  _expected_input_types = (tdt.PyObjectType, tdt.SequenceType, tdt.TupleType)

  def _update_input_type(self):
    if isinstance(self.input_type, tdt.BroadcastSequenceType):
      raise TypeError('cannot get the length of an infinite sequence: %s' %
                      self.input_type)

  def _evaluate(self, eval_ctx, x):
    return eval_ctx.constant(np.asarray(len(x), self._dtype))


def Slice(*args, **kwargs):  # pylint: disable=invalid-name
  """A block which applies Python slicing to a PyObject, Tuple, or Sequence.

  For example, to reverse a sequence:
  ```python
  (Map(Scalar()) >> Slice(step=-1)).eval(range(5)) => [4, 3, 2, 1, 0]
  ```

  Positional arguments are not accepted in order to avoid the ambiguity
  of slice(start=N) vs. slice(stop=N).

  Args:
    *args: Positional arguments; must be empty (see above).
    **kwargs: Keyword arguments; `start=None, stop=None, step=None, name=None`.

  Returns:
    The block.
  """
  if args:
    raise TypeError('Slice does not accept positional arguments; allowed '
                    'keyword arguments are start, stop, and step')
  name = kwargs.pop('name', None)
  return GetItem(_get_slice(**kwargs), name=name).set_constructor_name('Slice')


def _get_slice(start=None, stop=None, step=None):
  return slice(start, stop, step)


class ForwardDeclaration(tdt.IOBase):
  """A ForwardDeclaration is used to define Blocks recursively.

  Usage:

  ```python
  fwd = ForwardDeclaration(in_type, out_type)  # declare type of block
  block = ... fwd() ... fwd() ...              # define block recursively
  fwd.resolve_to(block)                        # resolve forward declaration
  ```
  """

  def __init__(self, input_type=None, output_type=None, name=None):
    super(ForwardDeclaration, self).__init__(input_type, output_type, name)
    self._instances = []

  def __call__(self):
    """Return a block that references the recursive definition."""
    if self._instances is None:
      raise ValueError('Declaration has already been resolved.')
    ref = _ForwardDeclarationRef(self.input_type, self.output_type,
                                 name=self._name)
    self._instances.append(ref)
    return ref

  def resolve_to(self, target_block):
    """Resolve the forward declaration by setting it to the given block."""
    b = convert_to_block(target_block)
    self.set_io_types(b)
    for bi in self._instances:
      bi.set_io_types(b)
      bi._target_block = b    # pylint: disable=protected-access
    self._instances = None  # prevent further __call__s.


class _ForwardDeclarationRef(Block):
  """A _ForwardDeclarationRef is a recursive reference to a parent block."""

  def __init__(self, input_type, output_type, name):
    self._target_block = None
    super(_ForwardDeclarationRef, self).__init__(
        input_type=input_type, output_type=output_type, name=name)
    self.set_constructor_name('ForwardDeclaration()')

  def _update_input_type(self):
    if self._target_block is not None:
      self._target_block.set_input_type(self.input_type)
      self.set_output_type(self._target_block.output_type)

  def _update_output_type(self):
    if self._target_block is not None:
      self._target_block.set_output_type(self.output_type)
      self.set_input_type(self._target_block.input_type)

  @property
  def is_forward_declaration_ref(self):
    return True

  @property
  def target_block(self):
    return self._target_block

  def _validate(self, compiler_ctx):
    # The target block has not been (and cannot be) validated yet.
    # We assume that it has the declared input and output types, and then
    # verify that later.
    if self.target_block is None:
      raise TypeError('Forward declaration %s was never resolved.' % self.name)
    # Set input and output on the target block; these will be checked
    # later when the target block is validated.
    self.set_io_types(self.target_block)
    super(_ForwardDeclarationRef, self)._validate(compiler_ctx)

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    return self._target_block._evaluate(eval_ctx, x)


class _ComposeIO(Identity):
  """Placeholder used to define inputs and outputs in within `Composition`s."""

  def __init__(self, parent, parent_name, is_input):
    if is_input:
      placeholder_name = 'input'
      self._update_parent = parent.set_input_type
    else:
      placeholder_name = 'output'
      self._update_parent = parent.set_output_type
    super(_ComposeIO, self).__init__(name=parent_name)
    self.set_constructor_name('Composition.%s' % placeholder_name)
    self._parent = parent

  def _update_input_type(self):
    super(_ComposeIO, self)._update_input_type()
    self._update_parent(self.input_type)

  def _evaluate(self, unused_eval_ctx, unused_x):
    raise RuntimeError('internal error; a composition placeholder should never '
                       'be evaluated; please file a bug')


class Composition(Block):
  """A composition of blocks, which are connected in a DAG."""

  def __init__(self, children=None, name=None):
    # Dict from block to list of input_wires feeding into the block.
    self._child_input_wire_dict = {}
    # Dict from block to set of blocks reading from the block.
    self._child_output_block_dict = collections.defaultdict(set)
    self._child_to_index = {}
    self._child_input_wires = []
    self._input_ph = _ComposeIO(self, name, is_input=True)
    self._output_ph = _ComposeIO(self, name, is_input=False)
    self._output_wires = None
    if children is not None: children = [convert_to_block(c) for c in children]
    super(Composition, self).__init__(children=children, name=name)

  @property
  def input(self):
    """Return a placeholder whose output is the input to the composition."""
    return self._input_ph

  @property
  def output(self):
    """Return a placeholder whose input is the output of the composition."""
    return self._output_ph

  # pylint:disable=g-doc-return-or-yield
  # This linter warning doesn't really apply to a context manager.
  @contextlib.contextmanager
  def scope(self):
    """Creates a context for use with the python `with` statement.

    Entering this context enabled the use of a block's `reads` method.  Once
    inside a context calling `some_block.reads(...)` sets `some_block`'s inputs
    within the composition.

    For example, you could make a composition which computes $x^2 + 10*x$
    element-wise for vectors of length 3 as follows:

    ```python
    c = td.Composition()
    with c.scope():
      x = td.Vector(3).reads(c.input)
      x_squared = td.Function(tf.mul).reads(x, x)
      ten = td.FromTensor(10 * np.ones(3, dtype='float32'))
      ten_x = td.Function(tf.mul).reads(ten, x)
      c.output.reads(td.Function(tf.add).reads(x_squared, ten_x)
    ```
    """
    old_composition_context = _COMPOSITION_CONTEXT.current
    _COMPOSITION_CONTEXT.current = self
    try:
      yield
    finally:
      if _COMPOSITION_CONTEXT.current is not self:
        raise AssertionError('scope nesting violation.')
      _COMPOSITION_CONTEXT.current = old_composition_context
  # pylint:enable=g-doc-return-or-yield

  def _update_input_type(self):
    self.input.set_output_type(self.input_type)
    self._types_forward(self.input)

  def _update_output_type(self):
    self.output.set_input_type(self.output_type)
    self._types_backward(self.output)

  def _types_backward(self, child):
    in_type = child.input_type
    if in_type is None: return
    wires = self._child_input_wire_dict.get(child)
    if wires is None: return
    if isinstance(in_type, tdt.PyObjectType) or len(wires) == 1:
      in_type = (in_type,) * len(wires)
    elif len(wires) != len(in_type):
      raise TypeError('block %s has %d inputs but expects %d' %
                      (child, len(wires), len(in_type)))
    for i, ((block, index), itype) in enumerate(zip(wires, in_type)):
      if index is None:
        block.set_output_type(itype)
      elif block.output_type is not None and block.output_type[index] != itype:
        # We don't propagate partial type information backward because
        # we don't currently have a representation of partial tuple
        # types (e.g. A tuple whose first item is a tensor).
        if len(wires) == 1:
          raise TypeError(
              'output %d of block %s has type %s; cannot connect'
              'it to block %s with input type %s' %
              (index, block, block.output_type[index], child, in_type))
        raise TypeError(
            'output %d of block %s has type %s; cannot connect'
            'it to input %d of block %s having type %s' %
            (index, block, block.output_type[index], i, child, itype))

  def _types_forward(self, child):
    for block in self._child_output_block_dict[child]:
      wires = self._child_input_wire_dict[block]
      block.set_input_type(self._get_input_type(wires))

  def _propagate_types_from_child(self, child):
    self._types_backward(child)
    self._types_forward(child)

  def _make_input_wire(self, a):
    if isinstance(a, _InputWire): return a
    return _InputWire(convert_to_block(a), None)

  def _get_wire_value(self, input_wire, results):
    """Return the value on input_wire, given child results."""
    (a, i) = input_wire
    res = results[self._child_to_index[a]]
    if i is not None:
      return res[i]
    return res

  def _get_input_values(self, input_wires, results):
    if not input_wires:
      return None
    elif len(input_wires) == 1:
      return self._get_wire_value(input_wires[0], results)
    else:
      return tuple(self._get_wire_value(w, results) for w in input_wires)

  def _get_input_type(self, input_wires):
    if not input_wires: return tdt.VoidType()
    wire_types = []
    for a, i in input_wires:
      if a.output_type is None: return None
      wire_types.append(a.output_type if i is None else a.output_type[i])
    if len(wire_types) == 1: return wire_types[0]
    return tdt.TupleType(wire_types)

  def _maybe_add_child(self, child):
    if isinstance(child, _ComposeIO):
      if child.parent != self:
        raise ValueError('%s is the input or output of a different composition'
                         % child)
    elif child not in self._children:
      self._add_child(child)

  def _create_input_wires(self, a):
    # TODO(delesley): write test cases for error conditions
    if isinstance(a, _InputWire):   # disabiguate from tuple, below
      input_wires = (a,)
    elif isinstance(a, (tuple, list)):
      input_wires = tuple(self._make_input_wire(ai) for ai in a)
    else:
      input_wires = (self._make_input_wire(a),)
    return input_wires

  def connect(self, a, b):
    """Connect `a` to the input of `b`.

    The argument `a` can be either:

    * A block, in which case the output of `a` is fed into the input of `b`.

    * The i^th output of a block, obtained from `a[i]`.

    * A tuple or list of blocks or block outputs.

    Args:
      a: Inputs to the block (see above).
      b: The block to connect the inputs to.

    Raises:
      ValueError: if `a` includes the output of the composition.
      ValueError: if `b` is the input of the composition.
      ValueError: if the input of `b` is already connected.
    """
    b = convert_to_block(b)
    input_wires = self._create_input_wires(a)

    # Make sure everything we're connecting is a child.
    for iw in input_wires:
      self._maybe_add_child(iw.block)
      if iw.block is self.output:
        raise ValueError('cannot read from composition output')
    self._maybe_add_child(b)
    if b is self.input:
      raise ValueError('cannot write to composition input')

    if b in self._child_input_wire_dict:
      raise ValueError('input of block is already connected: %s' % (b,))
    self._child_input_wire_dict[b] = input_wires
    if b is self.output: self._output_wires = input_wires

    if not input_wires:
      b.set_input_type(tdt.VoidType())
    elif len(input_wires) > 1:
      b.set_input_type_classes(tdt.TupleType, tdt.PyObjectType)

    for block, index in input_wires:
      if index is not None:
        block.set_output_type_classes(tdt.TupleType)
        if block.output_type is not None:
          arity = len(block.output_type)
          if arity < index:
            raise TypeError('cannot get %d-th output of block %s with %d '
                            'outputs' % (index, block, arity))
      self._child_output_block_dict[block].add(b)

    for wire in input_wires:
      self._types_forward(wire.block)  # propagate types to `b`
    self._propagate_types_from_child(b)  # propagate types from `b`

  def _validate(self, compiler_ctx):
    # pylint: disable=protected-access
    # Sort children in topological order, and detect cycles.
    topo_children = []
    visited = {}

    def visit(child):
      """Depth-first traversal."""
      if child is self._input_ph:
        return
      visited[child] = 1    # mark as visiting for cycle detection
      if child in self._child_input_wire_dict:
        for (b, _) in self._child_input_wire_dict[child]:
          if b in visited:
            if visited[b] == 1:
              raise ValueError('Composition cannot have cycles.')
          else:
            visit(b)
      topo_children.append(child)
      visited[child] = 2    # mark as visited

    sinks = set(self.children)
    for input_wires in six.itervalues(self._child_input_wire_dict):
      for wire in input_wires:
        sinks.discard(wire.block)
    if self._output_wires:
      for source, _ in self._output_wires:
        if source not in visited: visit(source)

    # Reorder sinks to match children to avoid nondeterminism.
    sinks = [c for c in self.children if c in sinks]

    # Find any children that were not reachable from the output.
    for child in sinks:
      assert child not in visited
      visit(child)

    self._children = topo_children

    # TODO(delesley): write test cases for error conditions
    self._child_input_wires = [[]]
    self._child_to_index[self.input] = 0
    for (i, b) in enumerate(self.children, 1):
      # Find the list of input wires for child b
      self._child_to_index[b] = i
      in_wires = self._child_input_wire_dict.get(b, [])
      # Set _child_input_wires for fast lookup during evaluation
      self._child_input_wires.append(in_wires)
      # Validate child, which will ensure it has an output type.
      b._validate(compiler_ctx)

    # Check that all child outputs are used.
    unused = [b for b in sinks if not isinstance(b.output_type, tdt.VoidType)]
    if unused:
      raise TypeError('children have unused outputs: %s' %
                      ', '.join(str(u) for u in unused))

    # Corner case: set input type to Void if all children have Void input.
    if self.input_type is None:
      if all([isinstance(child.input_type, tdt.VoidType)
              for child in self._children]):
        self.set_input_type(tdt.VoidType())

    # Check that composition has an input type.
    self._check_input_type()

    # Infer the composition output type.
    if not (self._output_wires or isinstance(self.output_type, tdt.VoidType)):
      # We could infer void output type here but more likely the user made a
      # mistake, so we throw unless the VoidType was set explicitly.
      raise TypeError('Composition block has no output: %s' % self)
    self._check_output_type()

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    ch_results = [x]
    for (i, b) in enumerate(self.children, 1):
      in_x = self._get_input_values(self._child_input_wires[i], ch_results)
      r = b._evaluate(eval_ctx, in_x)
      ch_results.append(r)
    final_result = self._get_input_values(self._output_wires, ch_results)
    return final_result


def Pipe(*blocks, **kwargs):  # pylint: disable=invalid-name
  """Creates a composition which pipes each block into the next one.

  `Pipe(a, b, c)` is equivalent to `a >> b >> c`.

  ```python
  Pipe(a, b, c).eval(x) => c(b(a(x)))
  ```

  Args:
    *blocks: A tuple of blocks.
    **kwargs: `{'name': name_string}` or `{}`.

  Returns:
    A block.
  """
  return _pipe([convert_to_block(b) for b in blocks],
               **kwargs).set_constructor_name('Pipe')


def _pipe(blocks, name=None):
  """Internal implementation of Pipe."""
  if not blocks: return Identity(name=name)
  if len(blocks) == 1: return blocks[0]

  c = Composition(blocks, name=name)
  c.connect(c.input, blocks[0])
  prev = blocks[0]
  for b in blocks[1:]:
    c.connect(prev, b)
    prev = b
  c.connect(prev, c.output)
  return c


class Record(Block):
  """Dispatch each element of a dict, list, or tuple to child blocks.

  A Record block takes a python dict or list of key-block pairs, or a
  tuple of blocks, processes each element, and returns a tuple of
  results as the output.

  ```python
  Record({'a': a_block, 'b': b_block}).eval(inp) =>
    (a_block.eval(inp['a']), b_block.eval(inp['b']))
  ```

  ```python
  Record([('a', a_block), ('b', b_block)]).eval(inp) =>
      (a_block.eval(inp['a']), b_block.eval(inp['b']))
  ```

  ```python
  Record((a_block, b_block)).eval(inp) =>
      (a_block.eval(inp[0]), b_block.eval(inp[1]))
  ```
  """

  def __init__(self, named_children, name=None):
    """Create a Record Block.

    If named_children is list or tuple or ordered dict, then the
    output tuple of the Record will preserve child order, otherwise
    the output tuple will be ordered by key.

    Args:
      named_children: A dictionary, list of (key, block) pairs, or a
        tuple of blocks (in which case the keys are 0, 1, 2, ...).
      name: An optional string name for the block.
    """
    self._unordered = (isinstance(named_children, dict) and
                       not isinstance(named_children, collections.OrderedDict))
    named_children = _get_sorted_list(named_children)
    self._named_children = named_children
    self._keys = [k for k, _ in named_children]
    super(Record, self).__init__(
        children=[c for (_, c) in named_children], name=name)
    if not self.children: self.set_output_type(())
    if not self.children: self.set_output_type(())

  def _repr_kwargs(self):
    return dict(ordered=not self._unordered)

  def __getitem__(self, k):
    """Return a reference to the k^th output, where k is a child key."""
    return _InputWire(self, self._keys.index(k))

  _expected_input_types = (tdt.PyObjectType, tdt.TupleType)
  _expected_output_types = (tdt.PyObjectType, tdt.TupleType)

  def _update_input_type(self):
    if isinstance(self.input_type, tdt.PyObjectType):
      # If we have input type PyObjectType then all children must also.
      for b in self.children:
        b.set_input_type(tdt.PyObjectType())
    elif self._unordered:
      # Prevent user from shooting self in foot.
      raise RuntimeError(
          'record block %s created with an unordered dict cannot take ordered '
          'inputs; use an OrderedDict or list of (key, block) instead' % self)
    else:
      assert isinstance(self.input_type, tdt.TupleType)
      # If we have a tuple input type, then children must correspond to items.
      if len(self.children) != len(self.input_type):
        raise TypeError('Record block has %d children but %d inputs: %s' %
                        (len(self.children), len(self.input_type),
                         list(self.input_type)))
      for b, t in zip(self.children, self.input_type):
        b.set_input_type(t)
    self._infer_output_type_from_children()

  def _update_output_type(self):
    if isinstance(self.output_type, tdt.PyObjectType):
      # If we have output type PyObjectType then all children must also.
      for b in self.children:
        b.set_output_type(tdt.PyObjectType())
    else:
      assert isinstance(self.output_type, tdt.TupleType)
      # If we have a tuple output type, then children must correspond to items.
      if len(self.children) != len(self.output_type):
        raise TypeError('Record block has %d children but %d outputs: %s' %
                        (len(self.children), len(self.output_type),
                         list(self.output_type)))
      for b, t in zip(self.children, self.output_type):
        b.set_output_type(t)

  def _propagate_types_from_child(self, _):
    if not any(b.input_type is None for b in self.children):
      self.set_input_type(tdt.TupleType(b.input_type for b in self.children))
    self._infer_output_type_from_children()

  def _infer_output_type_from_children(self):
    if not any(b.output_type is None for b in self.children):
      self.set_output_type(tdt.TupleType(b.output_type for b in self.children))

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    return tuple(b._evaluate(eval_ctx, x[k]) for (k, b) in self._named_children)


def AllOf(*blocks, **kwargs):  # pylint: disable=invalid-name
  """A block that runs all of its children (conceptually) in parallel.

  ```python
  AllOf().eval(inp) => None
  AllOf(a).eval(inp) => (a.eval(inp),)
  AllOf(a, b, c).eval(inp) => (a.eval(inp), b.eval(inp), c.eval(inp))
  ```

  Args:
    *blocks: Blocks.
    **kwargs: {name: name_string} or {}.

  Returns:
    See above.
  """
  return _all_of([convert_to_block(b) for b in blocks],
                 **kwargs).set_constructor_name('AllOf')


def _all_of(blocks, name=None):
  """Internal implementation of AllOf."""
  if not blocks: return Void(name=name)
  if len(blocks) == 1:
    # TODO(moshelooks): fix composition to allow for tuple output.
    return Pipe(blocks[0], AllOf(Identity(), Identity()), Slice(stop=1),
                name=name)
  c = Composition(blocks, name=name)
  for block in blocks:
    if not isinstance(block.input_type, tdt.VoidType):
      c.connect(c.input, block)
  c.connect(blocks, c.output)
  return c


@six.add_metaclass(abc.ABCMeta)
class _SeqToSeqBlock(Block):
  """Helper for blocks that special-case on finite vs. infinite sequences."""

  _expected_input_types = (tdt.SequenceType, tdt.TupleType, tdt.PyObjectType)
  _expected_output_types = (tdt.SequenceType, tdt.PyObjectType)

  def _infer_output_type(self, output_elem_type):
    if output_elem_type is None: return
    if (isinstance(self.input_type, tdt.BroadcastSequenceType) or
        (isinstance(self.input_type, tdt.TupleType) and
         all(isinstance(t, tdt.BroadcastSequenceType)
             for t in self.input_type))):
      self.set_output_type(tdt.BroadcastSequenceType(output_elem_type))
      self._evaluate = self._evaluate_infinite
    else:
      self.set_output_type(tdt.SequenceType(output_elem_type))
      self._evaluate = self._evaluate_finite

  @abc.abstractmethod
  def _evaluate_infinite(self, eval_ctx, x):
    """Specialization of _evaluate() for infinite sequences."""
    # This method should return a generator.

  @abc.abstractmethod
  def _evaluate_finite(self, eval_ctx, x):
    """Specialization of _evaluate() for finite sequences."""
    # This method should return a list.


class Map(_SeqToSeqBlock):
  """Map a block over a sequence or tuple."""

  def __init__(self, elem_block, name=None):
    super(Map, self).__init__(
        children=[convert_to_block(elem_block)], name=name)

  def _repr_kwargs(self):
    return dict(element_block=self.element_block)

  @property
  def element_block(self):
    return self.children[0]

  def _update_input_type(self):
    self.element_block.set_input_type(_infer_element_type(self.input_type))
    self._infer_output_type(self.element_block.output_type)

  def _update_output_type(self):
    self.element_block.set_output_type(_infer_element_type(self.output_type))

  def _propagate_types_from_child(self, _):
    self._update_input_type()

  def _evaluate_infinite(self, eval_ctx, x):
    # pylint: disable=protected-access
    return itertools.repeat(self.element_block._evaluate(eval_ctx, next(x)))

  def _evaluate_finite(self, eval_ctx, x):
    # pylint: disable=protected-access
    # TODO(delesley): think about generators
    elem_block = self.element_block
    return [elem_block._evaluate(eval_ctx, xi) for xi in x]


class Fold(Block):
  """Left-fold a two-argument block over a sequence or tuple."""

  def __init__(self, combine_block, start_block, name=None):
    self._start_block = convert_to_block(start_block)
    # Make sure start_block has void input.
    self.start_block.set_input_type(tdt.VoidType())
    self._combine_block = convert_to_block(combine_block)
    super(Fold, self).__init__(
        children=[self._start_block, self._combine_block], name=name)

  def _repr_kwargs(self):
    return dict(combine_block=self.combine_block, start_block=self.start_block)

  @property
  def start_block(self):
    return self._start_block

  @property
  def combine_block(self):
    return self._combine_block

  _expected_input_types = (tdt.SequenceType, tdt.TupleType, tdt.PyObjectType)

  def _update_input_type(self):
    if isinstance(self.input_type, tdt.BroadcastSequenceType):
      raise TypeError('cannot fold over an infinite sequence: %s' %
                      self.input_type)
    self._update_output_type()

  def _update_output_type(self):
    self.start_block.set_output_type(self.output_type)
    self.combine_block.set_output_type(self.output_type)
    if self.input_type is not None:
      elem_type = _infer_element_type(self.input_type)
      self.combine_block.set_input_type((self.output_type, elem_type))

  def _propagate_types_from_child(self, _):
    self.set_output_type(self.start_block.output_type)
    self.set_output_type(self.combine_block.output_type)
    cb_input_type = self.combine_block.input_type
    if cb_input_type is None: return
    if not isinstance(cb_input_type, tdt.TupleType):
      raise TypeError('Fold: combine_block has non-tuple input_type.')
    if len(cb_input_type) != 2:
      raise TypeError('Fold: combine_block input_type must be a 2-tuple.')
    self.set_output_type(cb_input_type[0])
    self._update_output_type()

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    res = self._start_block._evaluate(eval_ctx, None)
    for xi in x:
      res = self._combine_block._evaluate(eval_ctx, (res, xi))
    return res


class _RNN(Block):
  """Process a sequence with an RNN."""

  def __init__(self, rnn_cell_block, name=None):
    """Create an RNN block.

    Typically rnn_cell_block is Function(rnn_cell).

    Args:
      rnn_cell_block: A block that takes input of type (input_element, state)
        and produces output of type (output_element, state).
      name: An optional string name for the block.
    """
    self._rnn_cell_block = convert_to_block(rnn_cell_block)
    super(_RNN, self).__init__(children=[self._rnn_cell_block], name=name)
    self.set_constructor_name('RNN')

  def _repr_kwargs(self):
    return dict(rnn_cell_block=self.rnn_cell_block)

  @property
  def rnn_cell_block(self):
    return self._rnn_cell_block

  _expected_input_types = (tdt.TupleType,)
  _expected_output_types = (tdt.TupleType,)

  def _update_input_type(self):
    if len(self.input_type) != 2:
      raise TypeError('Expected a two-tuple of (input_sequence, state), saw: %s'
                      % (self.input_type,))
    in_seq_ty, in_st_ty = self.input_type
    if not isinstance(in_seq_ty, tdt.SequenceType):
      raise TypeError('First RNN input must be a sequence: %s' % (in_seq_ty,))
    if isinstance(in_seq_ty, tdt.BroadcastSequenceType):
      raise TypeError('cannot run an RNN on an infinite sequence: %s' %
                      (in_seq_ty,))
    self.rnn_cell_block.set_input_type((in_seq_ty.element_type, in_st_ty))
    self._check_state_type()

  def _update_output_type(self):
    if len(self.output_type) != 2:
      raise TypeError('Expected a two-tuple of (output_sequence, state), saw: '
                      '%s' % (self.output_type,))
    out_seq_ty, out_st_ty = self.output_type
    if not isinstance(out_seq_ty, tdt.SequenceType):
      raise TypeError('First RNN output must be a sequence: %s' % (out_seq_ty,))
    self.rnn_cell_block.set_output_type((out_seq_ty.element_type, out_st_ty))
    self._check_state_type()

  def _check_state_type(self):
    if self.input_type is None or self.output_type is None: return
    if self.input_type[1] != self.output_type[1]:
      raise TypeError('RNN cell input and output state types don\'t match:'
                      '%s vs. %s' % (self.input_type[1], self.output_type[1]))

  def _propagate_types_from_child(self, _):
    # Infer input_type from rnn_cell_block.
    in_ty = self._rnn_cell_block.input_type
    if in_ty:
      if not isinstance(in_ty, tdt.TupleType):
        raise TypeError('RNN cell must take a tuple as input.')
      if len(in_ty) != 2:
        raise TypeError('Expected a two-tuple of (input_elem, state), saw: %s' %
                        (in_ty,))
      in_el_ty, in_st_ty = in_ty
      self.set_input_type((tdt.SequenceType(in_el_ty), in_st_ty))
    # Infer output_type from rnn_cell_block.
    out_ty = self._rnn_cell_block.output_type
    if out_ty:
      if not isinstance(out_ty, tdt.TupleType):
        raise TypeError('RNN cell must produce a tuple as output.')
      out_el_ty, out_st_ty = out_ty
      if len(out_ty) != 2:
        raise TypeError('Expected a two-tuple of (output_elem, state), saw: %s'
                        % (out_ty,))
      self.set_output_type((tdt.SequenceType(out_el_ty), out_st_ty))

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    (xs, state) = x
    if not xs:
      return ([], state)
    outputs = []
    for xi in xs:
      (yi, state) = self._rnn_cell_block._evaluate(eval_ctx, (xi, state))
      outputs.append(yi)
    return (outputs, state)


def RNN(cell, initial_state=None,             # pylint: disable=invalid-name
        initial_state_from_input=False, name=None):
  """Create an RNN block.

  An RNN takes a tuple of (input sequence, initial state) as input, and
  returns a tuple of (output sequence, final state) as output.  It can be used
  to implement sequence-to-sequence RNN models, such as LSTMs.

  If `initial_state_from_input` is False (the default), then the output of
  `initial_state` will be used for the initial state instead, and the input to
  the RNN block is just the input sequence, rather than a (sequence, state)
  tuple.  If `initial_state` is None (the default), then a block of the form
  `td.Zeros(cell.output_type[1])` will be created. This requires
  that cell has an output type set (which it will if it is e.g. a
  `td.ScopedLayer` wrapping a tf rnn cell). For example:

  ```python
  cell = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=16), 'mygru')
  model = td.Map(td.Vector(8)) >> td.RNN(gru_cell)
  ```

  Args:
    cell: a block or layer that takes (input_elem, state) as input and
          produces (output_elem, state) as output.
    initial_state: an (optional) tensor or block to use for the initial state.
    initial_state_from_input:  if True, pass the initial state as an input
                               to the RNN block, otherwise use initial_state.
    name: An optional string name.

  Raises:
    ValueError: if initial_state_from_input == True and initial_state != None

  Returns:
    a block.
  """
  cell = convert_to_block(cell)

  if initial_state_from_input:
    if initial_state is not None:
      raise ValueError('Cannot specify initial_state if '
                       'initial_state_from_input is True.')
    return _RNN(cell, name=name)

  # Otherwise create a composition to wire in initial_state.
  if initial_state is None:
    if cell.output_type is None:
      raise TypeError('cannot infer initial_state type from cell')
    if not isinstance(cell.output_type, tdt.TupleType):
      raise TypeError('RNN cell must have tuple output type: %s' %
                      cell.output_type)
    _, state_type = cell.output_type
    initial_state = Zeros(state_type)
  else:
    initial_state = convert_to_block(initial_state)

  c = Composition(name=name).set_constructor_name('RNN')
  with c.scope():
    rnn = _RNN(cell, name=name).reads(c.input, initial_state)
    c.output.reads(rnn)
  return c


class Reduce(Block):
  """Reduce a two-argument block over a sequence or tuple."""

  def __init__(self, combine_block, default_block=None, name=None):
    self._combine_block = convert_to_block(combine_block)
    super(Reduce, self).__init__(name=name)
    self._add_child(self._combine_block)
    self._default_block = None
    if default_block is not None: self._set_default(default_block)

  def _repr_kwargs(self):
    return dict(combine_block=self.combine_block)

  def _set_default(self, block_like):
    self._default_block = convert_to_block(block_like)
    self._add_child(self._default_block.set_input_type(tdt.VoidType()))

  @property
  def combine_block(self):
    return self._combine_block

  @property
  def default_block(self):
    return self._default_block

  _expected_input_types = (tdt.TupleType, tdt.SequenceType)

  def _update_input_type(self):
    if isinstance(self.input_type, tdt.BroadcastSequenceType):
      raise TypeError('cannot reduce an infinite sequence: %s' %
                      (self.input_type,))
    elem_type = _infer_element_type(self.input_type)
    self._combine_block.set_input_type(tdt.TupleType(elem_type, elem_type))
    self.set_output_type(elem_type)

  def _update_output_type(self):
    self._combine_block.set_output_type(self.output_type)
    if self._default_block is None: self._set_default(Zeros(self.output_type))
    self.default_block.set_output_type(self.output_type)

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    if not x:
      return self._default_block._evaluate(eval_ctx, None)
    while len(x) > 1:
      evens = x[::2]
      odds = x[1::2]
      new_x = [self._combine_block._evaluate(eval_ctx, pair)
               for pair in zip(evens, odds)]
      if len(evens) != len(odds): new_x.append(x[-1])
      x = new_x
    return x[0]


def Sum(name=None):  # pylint: disable=invalid-name
  """Sums its inputs."""
  return Reduce(Function(tf.add), name=name).set_constructor_name('Sum')


def Min(name=None):  # pylint: disable=invalid-name
  """Takes the minimum of its inputs.  Zero on no inputs."""
  return Reduce(Function(tf.minimum), name=name).set_constructor_name('Min')


def Max(name=None):  # pylint: disable=invalid-name
  """Takes the maximum of its inputs.  Zero on no inputs."""
  return Reduce(Function(tf.maximum), name=name).set_constructor_name('Max')


def _tf_safe_reciprocal(x):
  return tf.reciprocal(x + tf.cast(tf.equal(x, 0), x.dtype))


def _tf_batch_scalar_mul(scalars, tensor_stack):
  for _ in xrange(len(tensor_stack.get_shape()) - 1):
    scalars = tf.expand_dims(scalars, -1)
  return tf.multiply(scalars, tensor_stack)


def _tf_batch_safe_scalar_division(tensor_stack, scalars):
  return _tf_batch_scalar_mul(_tf_safe_reciprocal(scalars), tensor_stack)


def Mean(name=None):  # pylint: disable=invalid-name
  """Takes the average of its inputs.  Zero on no inputs."""
  c = Composition(name=name)
  with c.scope():
    c.output.reads(Function(_tf_batch_safe_scalar_division).reads(
        Sum().reads(c.input), Length().reads(c.input)))
  return c.set_constructor_name('Mean')


class OneOf(Block):
  """A block that dispatches its input to one of its children.

  Can be used to dynamically dispatch on the type of its input, or emulate an
  'if' or 'switch' statement.

  ```python
  case_blocks = {'a': a_block, 'b': b_block}
  block = OneOf(GetItem('key'), case_blocks)

  inp1 = {'key': 'a', ...}
  inp2 = {'key': 'b', ...}
  block.eval(inp1) => a_block.eval(inp1)
  block.eval(inp2) => b_block.eval(inp2)
  ```

  ```python
  case_blocks = (block0, block1, block2)
  block = OneOf(GetItem('index'), case_blocks)

  inp1 = {'index': 0, ...}
  inp2 = {'index': -1, ...}
  block.eval(inp1) => block0.eval(inp1)
  block.eval(inp2) => block2.eval(inp2)
  ```
  """

  def __init__(self, key_fn, case_blocks, pre_block=None, name=None):
    """Creates the OneOf block.

    Args:
      key_fn: A python function or a block with `PyObject` output type,
        which returns a key, when given an input.  The key will be used to
        look up a child in `case_blocks` for dispatch.
      case_blocks: A non-empty Python dict, list of (key, block) pairs, or tuple
        of blocks (in which case the keys are 0, 1, 2, ...), where each block
        has the same input type `T` and the same output type.
      pre_block: An optional block with output type `T`.  If specified,
        pre_block will be used to pre-process the
        input before the input is handed to one of `case_blocks`.
      name: An optional string name for the block.

    Raises:
      ValueError: If `case_blocks` is empty.
    """
    try:
      # Try converting to a block first so that layers (which are callable)
      # get handled properly.
      self._key_block = convert_to_block(key_fn).set_output_type(
          tdt.PyObjectType())
    except TypeError:
      if not callable(key_fn):
        raise TypeError('key_fn is not callable: %s' % str(key_fn))
      self._key_block = InputTransform(key_fn)
    if not case_blocks: raise ValueError('case_blocks must be non-empty')
    named_cases = _get_sorted_list(case_blocks)
    children = [b for _, b in named_cases] + [self._key_block]
    self._case_blocks = dict(named_cases)
    if pre_block is not None:
      pre_block = convert_to_block(pre_block)
      children.append(pre_block)
    self._pre_block = pre_block
    super(OneOf, self).__init__(children=children, name=name)

  def _update_input_type(self):
    # We should have the same input type as key_block.
    self._key_block.set_input_type(self.input_type)
    if self._pre_block:
      # If pre_block exists, we should have the same input type as it, also.
      self._pre_block.set_input_type(self.input_type)
    else:
      # Otherwise, we should have the same input type as all the cases.
      for b in six.itervalues(self._case_blocks):
        b.set_input_type(self.input_type)

  def _update_output_type(self):
    for b in six.itervalues(self._case_blocks):
      b.set_output_type(self.output_type)

  def _propagate_types_from_child(self, _):
    self.set_input_type(self._key_block.input_type)
    if self._pre_block:
      for b in six.itervalues(self._case_blocks):
        b.set_input_type(self._pre_block.output_type)
    for b in six.itervalues(self._case_blocks):
      self.set_output_type(b.output_type)

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    key = self._key_block._evaluate(eval_ctx, x)
    if self._pre_block:
      x = self._pre_block._evaluate(eval_ctx, x)
    return self._case_blocks[key]._evaluate(eval_ctx, x)


class Optional(Block):
  """Dispatches its input based on whether the input exists, or is None.

  Similar to `OneOf(lambda x: x is None, {True: none_block, False: some_block})`
  except that `none_block` has `input_type` `VoidType`.
  """

  def __init__(self, some_case, none_case=None, name=None):
    """Creates an Optional block.

    Args:
      some_case: The block to evaluate on x if x exists.
      none_case: The block to evaluate if x is None -- defaults to zeros for
        tensor types, and an empty sequence for sequence types.
      name: An optional string name for the block.
    """
    self._some_case = convert_to_block(some_case)
    self._some_case.set_input_type(tdt.PyObjectType())
    children = [self._some_case]
    if none_case is None:
      self._none_case = None
    else:
      self._none_case = convert_to_block(none_case)
      children.append(self._none_case.set_input_type(tdt.VoidType()))
    super(Optional, self).__init__(
        children=children, input_type=tdt.PyObjectType(), name=name)

  def _repr_kwargs(self):
    return dict(some_case_block=self._some_case)

  def _update_output_type(self):
    self._some_case.set_output_type(self.output_type)
    if self._none_case is None:
      self._none_case = Zeros(self.output_type).set_input_type(tdt.VoidType())
      self._add_child(self._none_case)
    self._none_case.set_output_type(self.output_type)

  def _propagate_types_from_child(self, _):
    self.set_output_type(self._some_case.output_type)
    if self._none_case is not None:
      self.set_output_type(self._none_case.output_type)

  def _evaluate(self, eval_ctx, x):
    # pylint: disable=protected-access
    if x is None: return self._none_case._evaluate(eval_ctx, x)
    return self._some_case._evaluate(eval_ctx, x)


class Concat(Function):
  """Concatenates a non-empty tuple of tensors into a single tensor."""

  def __init__(self, concat_dim=0, flatten=False, name=None):
    """Create a Concat block.

    Args:
      concat_dim: The dimension to concatenate along (not counting the batch
        dimension).
      flatten: Whether or not to recursively concatenate nested tuples of
        tensors. Default is False, in which case we throw on nested tuples.
      name: An optional string name for the block. If present, must be a valid
        name for a TensorFlow scope.
    """
    self._concat_dim = concat_dim
    self._flatten = flatten
    def tf_concat(*inputs):
      inputs = list(inputs)
      for i in self._scalar_indices:
        inputs[i] = tf.expand_dims(inputs[i], 1)
      return tf.concat(inputs, concat_dim + 1)  # first dimension is batch
    super(Concat, self).__init__(
        tf_fn=tf_concat, name=name, infer_output_type=False)

  def _repr_kwargs(self):
    return dict(concat_dim=self._concat_dim, flatten=self._flatten)

  _expected_input_types = (tdt.TupleType,)
  _expected_output_types = (tdt.TensorType,)

  def _update_input_type(self):
    if self.input_type.size is None:
      raise TypeError('Concat inputs must be tensors: %s' % self.input_type)
    if (not self._flatten and
        any(isinstance(t, tdt.TupleType) for t in self.input_type)):
      raise TypeError('input type %s contains nested tuples, expected a flat '
                      'tuple of tensors; set flatten=True in the constructor' %
                      self.input_type)

    size = 0
    shape = None
    dtype = None
    self._scalar_indices = []
    for (i, ty) in enumerate(self.input_type.terminal_types()):
      tyshape = list(ty.shape)         # clone original shape
      tyrank = len(tyshape)
      if tyrank == 0:
        tyshape = [1]                  # upgrade scalars to vectors
        tyrank = 1
        self._scalar_indices.append(i)
      if tyrank <= self._concat_dim:
        raise TypeError('Concat argument %d of type %s has rank less than %d.' %
                        (i, ty, self._concat_dim+1))
      size += tyshape[self._concat_dim]
      tyshape[self._concat_dim] = None  # for shape matching

      if not shape:
        shape = tyshape
      elif shape != tyshape:
        raise TypeError('Shapes for concat don\'t match: %s vs. %s'
                        % (shape, tyshape))
      if not dtype:
        dtype = ty.dtype
      elif ty.dtype != dtype:
        raise TypeError('Cannot concat tensors of different dtypes: %s vs. %s'
                        % (dtype, ty.dtype))
    if not dtype:
      raise TypeError('Concat requires at least one tensor as input')
    shape[self._concat_dim] = size
    self.set_output_type(tdt.TensorType(shape, dtype=dtype))

  def _compile(self, compiler_ctx):
    # Pass the loom op a flattened version of our input type because
    # we don't want it to be unflattened when we call tf_concat.
    flat_input_type = tdt.TupleType(self.input_type.terminal_types())
    fop = loom_ops.FuncallOp(self._tf_fn, flat_input_type, self.output_type)
    self._loom_op_id = compiler_ctx.register_op(fop, self.name)


class Broadcast(Block):
  """Block that creates an infinite sequence of the same element.

  This is useful in conjunction with `Zip` and `Map`, for example:

  ```python
  def center_seq(seq_block):
    return (seq_block >> AllOf(Identity(), Mean() >> Broadcast()) >> Zip() >>
            Map(Function(tf.sub)))
  ```
  """

  def __init__(self, name=None):
    super(Broadcast, self).__init__(name=name)

  _expected_output_types = (tdt.BroadcastSequenceType,)

  def _update_input_type(self):
    self.set_output_type(tdt.BroadcastSequenceType(self.input_type))

  def _evaluate(self, _, x):
    return itertools.repeat(x)


class Zip(_SeqToSeqBlock):
  """Converts a tuple of sequences to a sequence of tuples.

  The output sequence is truncated in length to the length of the
  shortest input sequence.
  """

  def __init__(self, name=None):
    super(Zip, self).__init__(name=name)

  _expected_input_types = (tdt.TupleType,)

  def _update_input_type(self):
    if not self.input_type:
      raise TypeError('zip requires at least one input sequence')
    for item_type in self.input_type:
      if not isinstance(item_type, (tdt.SequenceType, tdt.PyObjectType)):
        raise TypeError('item types must be sequences: %s' % str(item_type))
    elem_type = tdt.TupleType(_infer_element_type(t) for t in self.input_type)
    self._infer_output_type(elem_type)

  def _evaluate_infinite(self, _, x):
    return itertools.repeat(tuple(next(y) for y in x))

  def _evaluate_finite(self, _, x):
    return list(zip(*x))


def ZipWith(elem_block, name=None):   # pylint: disable=invalid-name
  """A Zip followed by a Map.

  ```python
  ZipWith(elem_block) => Zip() >> Map(elem_block)
  ```

  Args:
    elem_block: A block with a tuple input type.
    name: An optional string name for the block.

  Returns:
    A block zips its input then maps over it with `elem_block`.
  """
  return Zip() >> Map(elem_block, name=name)


class NGrams(_SeqToSeqBlock):
  """Computes tuples of n-grams over a sequence.

  ```python
  (Map(Scalar()) >> NGrams(2)).eval([1, 2, 3]) => [(1, 2), (2, 3)]
  ```
  """

  def __init__(self, n, name=None):
    if n <= 0: raise ValueError('n must be positive: %s' % str(n))
    self._n = n
    super(NGrams, self).__init__(name=name)

  def _repr_kwargs(self):
    return dict(n=self.n)

  @property
  def n(self):
    return self._n

  _expected_input_types = (tdt.SequenceType,)

  def _update_input_type(self):
    self._infer_output_type(tdt.TupleType(
        *([self.input_type.element_type] * self.n)))

  def _evaluate_finite(self, _, x):
    n = self.n
    return [tuple(x[i:i+n]) for i in xrange(len(x) - n + 1)]

  def _evaluate_infinite(self, _, x):
    return itertools.repeat((next(x),) * self.n)


class OneHot(Block):
  """A block that converts PyObject input to a one-hot encoding.

  Will raise an `KeyError` if the block is applied to an out-of-range input.
  """

  def __init__(self, start, stop=None, dtype='float32', name=None):
    """Initializes the block.

    Args:
      start: The start of the input range.
      stop: Upper limit (exclusive) on the input range. If stop is `None`, the
        range is `[0, start)`, like the Python range function.
      dtype: The dtype for the output array.
      name: An optional string name for the block.

    Raises:
      IndexError: If the range is empty.
    """
    if stop:
      n = stop - start
    else:
      n = start
      start = 0
    if n <= 0:
      raise IndexError('range [%d, %d) is empty.' % (start, start + n))
    self._start = start
    super(OneHot, self).__init__(name=name, input_type=tdt.PyObjectType(),
                                 output_type=tdt.TensorType([n], dtype))

  def _repr_kwargs(self):
    return dict(dtype=self.output_type.dtype, start=self._start,
                stop=self._start + self.output_type.shape[0])

  def _compile(self, compiler_ctx):
    array = np.identity(self.output_type.shape[0], self.output_type.dtype)
    self._tensor_names = {
        index: compiler_ctx.register_tensor(tf.constant(row), self.name)
        for index, row in enumerate(array, self._start)}

  def _evaluate(self, eval_ctx, x):
    return eval_ctx.named_tensor(self._tensor_names[x])


def OneHotFromList(elements, dtype='float32', strict=True, name=None):  # pylint: disable=invalid-name
  """A block that converts PyObject input to a one-hot encoding.

  Differs from `OneHot` in that the user specifies the elements covered by the
  one-hot encoding rather than assuming they are consecutive integers.

  Args:
    elements: The list of elements to be given one-hot encodings.
    dtype: The type of the block's return value.
    strict: Whether the block should throw a KeyError if it encounters an input
      which wasn't in elements.  Default: True.
    name: An optional string name for the block.

  Raises:
    AssertionError: if any of the `elements` given are equal.

  Returns:
    A Block that takes a PyObject and returns a tensor of type `dtype` and shape
    `[len(elements)]`.  If passed any member of `elements` the block will return
    a basis vector corresponding to the position of the element in the list.  If
    passed anything else the block will throw a KeyError if `strict` was set to
    True, and return the zero vector if `strict` was set to False.
  """
  dimension = len(elements)

  tensors = {}
  for idx, basis_vector in enumerate(np.identity(dimension, dtype)):
    tensors[idx] = FromTensor(basis_vector)

  indices = {elt: idx for idx, elt in enumerate(elements)}
  assert len(indices) == dimension, (
      'OneHotFromList was passed duplicate elements.')

  if strict:
    key_fn = lambda x: indices[x]
  else:
    tensors[-1] = Zeros([dimension, dtype])
    key_fn = lambda x: indices.get(x, -1)

  return OneOf(key_fn, tensors, pre_block=Void(),
               name=name).set_constructor_name('OneHotFromList')


class Nth(Block):
  """Extracts the Nth element of a sequence, where N is a PyObject.

  ```python
  block = (Map(Scalar()), Identity()) >> Nth()
  block.eval((list, n)) => list[n]
  ```
  """

  def __init__(self, name=None):
    super(Nth, self).__init__(name=name)

  _expected_input_types = (tdt.TupleType,)

  def _update_input_type(self):
    if len(self.input_type) != 2:
      raise TypeError('Nth block takes 2 inputs %s' % list(self.input_type))
    seq_type, n_type = self.input_type
    if not isinstance(seq_type, tdt.SequenceType):
      raise TypeError('first input to Nth must be a sequence: %s' % seq_type)
    if isinstance(seq_type, tdt.BroadcastSequenceType):
      raise TypeError('cannot call Nth on an infinite sequence: %s' % seq_type)
    if not isinstance(n_type, tdt.PyObjectType):
      raise TypeError('second input to Nth must be a PyObject: %s' % n_type)
    self.set_output_type(seq_type.element_type)

  def _evaluate(self, _, x):
    return x[0][x[1]]


def Zeros(output_type, name=None):  # pylint: disable=invalid-name
  """A block of zeros, voids, and empty sequences of `output_type`.

  If `output_type` is a tensor type, the output is `tf.zeros` of this
  type. If it is a tuple type, the output is a tuple of `Zeros` of the
  corresponding item types. If it is void, the output is void. If it
  is a sequence type, the output is an empty sequence of this type.

  Args:
    output_type: A type. May not contain pyobject types.
    name: An optional string name for the block.

  Returns:
    A block.

  Raises:
    TypeError: If `output_type` contains pyobject types.
  """
  pp_output_type = output_type   # for pretty printing
  output_type = tdt.convert_to_type(output_type)
  if not all(isinstance(t, tdt.TensorType)
             for t in output_type.terminal_types()):
    raise TypeError('all terminal types must be tensors: %s' % output_type)
  if isinstance(output_type, tdt.TensorType):
    result = FromTensor(np.zeros_like(output_type), name=name)
  elif isinstance(output_type, tdt.TupleType):
    result = AllOf(*[Zeros(itype) for itype in output_type], name=name)
  elif isinstance(output_type, tdt.BroadcastSequenceType):
    raise TypeError('cannot create Zeros for an infinite sequence type: %s' %
                    output_type)
  elif isinstance(output_type, tdt.VoidType):
    result = Void(name=name)
  else:
    assert isinstance(output_type, tdt.SequenceType)
    result = _EmptySequence(input_type=tdt.VoidType(), output_type=output_type,
                            name=name)
  result.set_constructor_args(pp_output_type)
  return result.set_constructor_name('Zeros')


def Void(name=None):  # pylint: disable=invalid-name
  """A block with void output type that accepts any input type."""
  return Composition(name=name).set_output_type(
      tdt.VoidType()).set_constructor_name('Void')


def convert_to_block(block_like):
  """Converts `block_like` to a block.

  The conversion rules are as follows:

  |type of `block_like`                   | result                   |
  |-------------------------------------- | -------------------------|
  |`Block`                                | `block_like`             |
  |`Layer`                                | `Function(block_like)`   |
  |`(tf.Tensor, tf.Variable, np.ndarray)` | `FromTensor(block_like)` |
  |`(dict, list, tuple)`                  | `Record(block_like)`     |

  Args:
    block_like: Described above.

  Returns:
    A block.

  Raises:
    TypeError: If `block_like` cannot be converted to a block.
  """
  if isinstance(block_like, Block): return block_like
  if _is_layer(block_like): return Function(block_like)
  if isinstance(block_like, (tf.Tensor, tf.Variable, np.ndarray)):
    return FromTensor(block_like)
  if isinstance(block_like, (dict, list, tuple)): return Record(block_like)
  raise TypeError('%s cannot be converted to a block' % str(block_like))


# Misc. internal helpers go here.


def _infer_element_type(result_type):
  if result_type is None: return None
  if isinstance(result_type, tdt.PyObjectType):
    return tdt.PyObjectType()
  elif isinstance(result_type, tdt.SequenceType):
    return result_type.element_type
  elif isinstance(result_type, tdt.TupleType):
    if not result_type: raise TypeError('expected a non-empty tuple')
    if len(set(result_type)) != 1:
      raise TypeError('tuple item types must all be equal: %s' % result_type)
    return result_type[0]
  else:
    raise TypeError('Expected python object or sequence or tuple: %s' %
                    str(result_type))


def _get_sorted_list(named_children):
  """Convert named_children to a sorted list."""
  if isinstance(named_children, tuple):
    named_children = enumerate(named_children)
  elif isinstance(named_children, collections.OrderedDict):
    named_children = six.iteritems(named_children)
  elif isinstance(named_children, dict):
    named_children = sorted(six.iteritems(named_children))
  elif not isinstance(named_children, list):
    raise TypeError('expected a list or tuple or dictionary of child blocks: %s'
                    % str(named_children))
  return [(k, convert_to_block(v)) for k, v in named_children]


def _infer_tf_output_type_from_input_type(fn, input_type):
  assert input_type.size is not None
  with tf.name_scope('tensorflow_fold_output_type_inference'):
    inputs = [tf.placeholder(t.dtype, (None,) + t.shape)
              for t in input_type.terminal_types()]
    if not isinstance(input_type, tdt.TensorType):
      inputs = input_type.unflatten(iter(inputs), None)
    return _type_from_outputs(fn(*inputs))


def _type_from_outputs(outputs):
  if isinstance(outputs, collections.Sequence):
    return tdt.TupleType(_type_from_outputs(o) for o in outputs)
  return _type_from_batch_tensor(outputs)


def _type_from_batch_tensor(tensor):
  """Computes instance type from a tensor representing a batch of instances."""
  if not isinstance(tensor, (tf.Tensor, tf.Variable)):
    raise TypeError('%s is not a TF tensor' % str(tensor))
  shape = tensor.get_shape()
  if shape.ndims is None: raise TypeError('%s has unspecified rank' % tensor)
  if not shape.ndims:
    raise TypeError('expected a batch tensor, saw a scalar: %s', tensor)
  if shape[0].value is not None:
    raise TypeError('leading (batch) dimension should be None: %s' % tensor)
  shape = shape[1:]  # drop the batch dimension
  if not shape.is_fully_defined():
    raise TypeError('instance shape is not fully defined: %s' % tensor)
  return tdt.TensorType(shape.as_list(), tensor.dtype)


class _EmptySequence(Block):

  def _evaluate(self, unused_eval_ctx, unused_x):
    return []


def _is_layer(x):
  return isinstance(
      x, tensorflow_fold.blocks.layers.Layer)
