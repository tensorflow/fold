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
"""Neural net layers for TensorFlow Fold.

Layers are a convenience rather than an integral part of Fold.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import itertools
# import google3
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_fold.blocks.blocks
import tensorflow_fold.blocks.result_types as tdt


class Layer(tdt.IOBase):
  """A callable that accepts and returns nests of batched of tensors."""

  def __init__(self, input_type=None, output_type=None, name_or_scope=None):
    """Creates the layer.

    Args:
      input_type: A type.
      output_type: A type.
      name_or_scope: A string or variable scope. If a string, a new variable
        scope will be created by calling
        [`create_variable_scope`](#create_variable_scope), with defaults
        inherited from the current variable scope. If no caching device is set,
        it will be set to `lambda op: op.device`. This is because `tf.while` can
        be very inefficient if the variables it uses are not cached locally.
    """
    if name_or_scope is None: name_or_scope = type(self).__name__
    if isinstance(name_or_scope, tf.VariableScope):
      self._vscope = name_or_scope
      name = str(self._vscope.name)
    elif isinstance(name_or_scope, six.string_types):
      self._vscope = create_variable_scope(name_or_scope)
      name = name_or_scope
    else:
      raise TypeError('name_or_scope must be a tf.VariableScope or a string: '
                      '%s' % (name_or_scope,))
    if self._vscope.caching_device is None:
      self._vscope.set_caching_device(lambda op: op.device)
    super(Layer, self).__init__(input_type, output_type, name)

  def __rshift__(self, rhs):
    return tensorflow_fold.blocks.blocks.Pipe(
        self, rhs)

  def __rrshift__(self, lhs):
    return tensorflow_fold.blocks.blocks.Pipe(
        lhs, self)


def create_variable_scope(name):
  """Creates a new variable scope based on `name`, nested in the current scope.

  If `name` ends with a `/` then the new scope will be created exactly as if
  you called `tf.variable_scope(name)`.  Otherwise, `name` will be
  made globally unique, in the context of the current graph (e.g.
  `foo` will become `foo_1` if a `foo` variable scope already exists).

  Args:
    name: A non-empty string.

  Returns:
    A variable scope.

  Raises:
    TypeError: if `name` is not a string.
    ValueError: if `name` is empty.

  """
  if not isinstance(name, six.string_types):
    raise TypeError('name must be a string: %s' % (name,))
  if not name: raise ValueError('name must be non-empty')
  if name.endswith('/'):
    with tf.variable_scope(name) as scope:
      return scope
  current_scope_name = tf.get_variable_scope().name
  if current_scope_name:
    full_name = '%s/%s' % (current_scope_name, name)
  else:
    full_name = name
  # We rely on the fact that every variable scope has a name scope
  # with the exact same name, so a unique name scope is by
  # implication also a unique name for a variable scope.
  with tf.name_scope(None):  # enter the root name scope
    with tf.name_scope(full_name) as unique_name:
      pass
  if current_scope_name: unique_name = unique_name[len(current_scope_name)+1:]
  with tf.variable_scope(unique_name[:-1]) as scope:
    return scope


@six.add_metaclass(abc.ABCMeta)
class TensorToTensorLayer(Layer):
  """A set of TF variables and an associated Tensor -> Tensor function."""

  def __init__(self, *args, **kwargs):
    self._created_variables = False
    super(TensorToTensorLayer, self).__init__(*args, **kwargs)

  @abc.abstractmethod
  def _create_variables(self):
    """Creates the variables associated with this layer.

    Guaranteed to be called at most once, either when the layer's call operator
    is invoked for the first time, in which case the input type will have been
    set, or when the public method create_variables is called for the first
    time. Scope will be set to this layer's vscope.

    Raises:
      TypeError: If `input_type` is invalid for this layer or isn't set.
    """
    pass

  @abc.abstractmethod
  def _process_batch(self, batch):
    """Processes a batch of inputs using this layer; called in its vscope.

    Args:
      batch: A batch tensor for this layer's input type.

    Returns:
      A tensor of this layer's output type.
    """
    pass

  def __call__(self, batch):
    """Calls the function associated with this layer on a batch of inputs.

    Creates the variables for this layer if they don't already exist.

    Args:
      batch: A batch tensor.

    Returns:
      A tensor of this layer's output type.

    Raises:
      ValueError: If the layer was previously called with a batch of a different
        dtype or shape (not considering the leading dimension).
    """
    self.set_input_type(
        tdt.TensorType(batch.get_shape().as_list()[1:], batch.dtype))
    self.create_variables()
    with tf.variable_scope(self._vscope):
      return self._process_batch(batch)

  def create_variables(self):
    """Creates the variables for this layer if they don't already exist.

    If the variables are created by this method rather than by calling the
    layer, the input type may need to be set manually.

    Raises:
      TypeError: If the input type is invalid or unset.
    """
    self._check_input_type()
    with tf.variable_scope(self._vscope):
      if not self._created_variables:
        self._create_variables()
        self._created_variables = True


class FC(TensorToTensorLayer):
  """A fully connected network layer.

  Fully connected layers require a `float32` vector (i.e. 1D tensor) as input,
  and build `float32` vector outputs. Layers can be applied to multiple inputs,
  provided they all have the same shape.

  For example, to apply the same hidden layer to two different input fields:
  ```python
  layer = FC(100)
  in = {'a': Vector(10), 'b': Vector(10)}
  hidden = [in['a'] >> Call(layer), in['b'] >> Call(layer)] >> Concat()
  out = hidden >> Call(FC(10, activation=None))
  ```
  """

  def __init__(self, num_units_out, activation=tf.nn.relu, initializer=None,
               input_keep_prob=None, output_keep_prob=None, name=None):
    """Initializes the layer.

    Args:
      num_units_out: The number of output units in the layer.
      activation: The activation function. Default is ReLU. Use `None` to get a
        linear layer.
      initializer: The initializer for the weights. Defaults to uniform unit
        scaling with factor derived in <http://arxiv.org/pdf/1412.6558v3.pdf>
        if activation is ReLU, ReLU6, tanh, or linear. Otherwise defaults to
        truncated normal initialization with a standard deviation of 0.01.
      input_keep_prob: Optional scalar float32 tensor for dropout on input.
        Feed 1.0 at serving to disable dropout.
      output_keep_prob: Optional scalar float32 tensor for dropout on output.
        Feed 1.0 at serving to disable dropout.
      name: An optional string name. Defaults to `FC_%d % num_units_out`. Used
        to name the variable scope where the variables for the layer live.
    """
    if not initializer:
      # TODO(SamEisenstat): This constant is calibrated for ReLU, something else
      # might be better for ReLU6.
      if activation in [tf.nn.relu, tf.nn.relu6]:
        initializer = tf.uniform_unit_scaling_initializer(1.43)
      elif activation == tf.tanh:
        initializer = tf.uniform_unit_scaling_initializer(1.15)
      elif not activation:
        initializer = tf.uniform_unit_scaling_initializer(1.0)
      else:
        initializer = tf.truncated_normal_initializer(stddev=0.01)
    self._activation = activation
    self._initializer = initializer
    self._input_keep_prob = input_keep_prob
    self._output_keep_prob = output_keep_prob
    if name is None: name = 'FC_%d' % num_units_out
    super(FC, self).__init__(
        output_type=tdt.TensorType([num_units_out]), name_or_scope=name)

  @property
  def output_size(self):
    return self.output_type.shape[0]

  def _create_variables(self):
    if self.input_type.dtype != 'float32':
      raise TypeError('FC input dtype must be float32: %s' %
                      self.input_type.dtype)
    if self.input_type.ndim != 1:
      raise TypeError('FC input shape must be 1D: %s' %
                      str(self.input_type.shape))
    self._bias = tf.get_variable(
        'bias', self.output_type.shape, initializer=tf.constant_initializer(0))
    self._weights = tf.get_variable(
        'weights', [self.input_type.shape[0], self.output_type.shape[0]],
        initializer=self._initializer)

  def _process_batch(self, batch):
    if self._input_keep_prob is not None:
      batch = tf.nn.dropout(batch, self._input_keep_prob)
    y = tf.nn.xw_plus_b(batch, self._weights, self._bias)
    if self._activation is not None: y = self._activation(y)
    if self._output_keep_prob is not None:
      y = tf.nn.dropout(y, self._output_keep_prob)
    return y


class Embedding(TensorToTensorLayer):
  """An embedding for integers.

  Embeddings require integer scalars as input, and build `float32` vector
  outputs. Embeddings can be applied to multiple inputs. `Embedding` doesn't
  do any hashing on its own, it just takes its inputs mod `num_buckets`
  to determine which embedding(s) to return.

  Implementation detail: `tf.gather` currently only supports `int32`
  and `int64`. If the input type is smaller than 32 bits it will be
  cast to `tf.int32`. Since all currently defined TF dtypes other than
  `int32` and `int64` have less than 32 bits, this means that we
  support all current integer dtypes.
  """

  def __init__(self, num_buckets, num_units_out, initializer=None, name=None,
               trainable=True, mod_inputs=True):
    """Initializes the layer.

    Args:
      num_buckets: How many buckets the embedding has.
      num_units_out: The number of output units in the layer.
      initializer: the initializer for the weights. Defaults to uniform unit
        scaling. The initializer can also be a Tensor or numpy array, in which
        case the weights are initialized to this value and shape. Note that in
        this case the weights will still be trainable unless you also pass
        `trainable=False`.
      name: An optional string name. Defaults to
        `Embedding_%d_%d % (num_buckets, num_units_out)`. Used to name the
        variable scope where the variables for the layer live.
      trainable: Whether or not to make the weights trainable.
      mod_inputs: Whether or not to mod the input by the number of buckets.

    Raises:
      ValueError: If the shape of `weights` is not
        `(num_buckets, num_units_out)`.
    """
    self._weights_shape = (num_buckets, num_units_out)
    if name is None: name = 'Embedding_%d_%d' % self._weights_shape
    if initializer is None:
      initializer = tf.uniform_unit_scaling_initializer(1.0)
    elif isinstance(initializer, np.ndarray):
      initializer = tf.convert_to_tensor(initializer)
    if isinstance(initializer, tf.Tensor):
      initializer.set_shape(self._weights_shape)
      self._weights_shape = None  # otherwise get_variable barfs
    self._initializer = initializer
    self._num_buckets = num_buckets
    self._num_units_out = num_units_out
    self._trainable = trainable
    self._mod_inputs = bool(mod_inputs)
    super(Embedding, self).__init__(
        output_type=tdt.TensorType([num_units_out]), name_or_scope=name)

  def _create_variables(self):
    if self.input_type.ndim != 0:
      raise TypeError('Embeddings take scalar inputs.')
    dtype = tf.as_dtype(self.input_type.dtype)
    if not dtype.is_integer: raise TypeError('Embeddings take integer inputs.')
    if dtype not in (tf.int32, tf.int64):  # only dtypes supported by tf.gather
      if np.iinfo(dtype.as_numpy_dtype).max > 2147483647:
         # pedantic future-proofing to handle hypothetical tf.uint64
        raise TypeError('cannot gather or upcast dtype %s' % dtype)
      self._cast = True
    else:
      self._cast = False
    self._weights = tf.get_variable(
        'weights', self._weights_shape, initializer=self._initializer,
        trainable=self._trainable)

  @property
  def weights(self):
    if not self._created_variables:
      raise RuntimeError('weights have not been created; call the layer first')
    return self._weights

  @property
  def num_buckets(self):
    return self._num_buckets

  @property
  def num_units_out(self):
    return self._num_units_out

  def _process_batch(self, batch):
    # We have to call tf.abs before calling tf.mod, because tf.mod gives
    # native outputs when given negative inputs.
    if self._cast: batch = tf.cast(batch, tf.int32)
    if self._mod_inputs: batch = tf.mod(tf.abs(batch), self._num_buckets)
    return tf.gather(self._weights, batch)


def _binary_sequences_of_at_most(n):
  return itertools.chain.from_iterable(
      itertools.product((0, 1), repeat=i) for i in xrange(n+1))


class FractalNet(TensorToTensorLayer):
  """An implementation of FractalNet.

  See https://arxiv.org/abs/1605.07648 for details.
  """

  # Choices for drop-path (names describe which paths are kept.)
  _BOTH = 0
  _JUST_BASE = 1
  _JUST_RECURSE = 2

  def __init__(self, num_fractal_blocks, fractal_block_depth,
               base_layer_builder, mixer=None, drop_path=False,
               p_local_drop_path=0.5, p_drop_base_case=0.25,
               p_drop_recursive_case=0.25, name=None):
    """Initializes the FractalNet.

    Args:
      num_fractal_blocks: The number of fractal blocks the net is made from.
        This variable is named `B` in the FractalNet paper.  This argument uses
        the word `block` in the sense that the FractalNet paper uses it.
      fractal_block_depth: How deeply nested the blocks are.  This variable is
        `C-1` in the paper.
      base_layer_builder: A callable that takes a name and returns a `Layer`
        object.  We would pass in a convolutional layer to reproduce the results
        in the paper.
      mixer: The join operation in the paper.  Assumed to have two arguments.
        Defaults to element-wise averaging.  Mixing doesn't occur if either path
        gets dropped.
      drop_path: A boolean, whether or not to do drop-path.  Defaults to False.
        If selected, we do drop path as described in the paper (unless drop-path
        choices is provided in which case how drop path is done can be further
        customized by the user.
      p_local_drop_path: A probability between 0.0 and 1.0.  0.0 means always do
        global drop path.  1.0 means always do local drop path.  Default: 0.5,
        as in the paper.
      p_drop_base_case: The probability, when doing local drop path, to drop the
        base case.
      p_drop_recursive_case: The probability, when doing local drop path, to
        drop the recusrive case. (Requires: `p_drop_base_case +
        p_drop_recursive_case < 1`)
      name: An optional string name.
    """
    if mixer is None:
      mixer = lambda a, b: tf.add(a, b)/2.0
    self._num_fractal_blocks = num_fractal_blocks
    self._fractal_block_depth = fractal_block_depth
    self._mixer = mixer
    self._drop_path = drop_path
    self._p_local_drop_path = p_local_drop_path
    self._p_drop_base_case = p_drop_base_case
    self._p_drop_recursive_case = p_drop_recursive_case
    self._drop_path_choices = None

    super(FractalNet, self).__init__(name_or_scope=name)
    self._children = {}
    self._choice_id = {}
    self._choices = []
    with tf.variable_scope(self._vscope):
      for block_idx in xrange(num_fractal_blocks):
        for binary_seq in _binary_sequences_of_at_most(fractal_block_depth):
          child_name = 'block_' + '_'.join(
              [str(block_idx)] + [str(b) for b in binary_seq])
          self._children[block_idx, binary_seq] = base_layer_builder(
              name=child_name)
          if len(binary_seq) < fractal_block_depth:
            self._choice_id[(block_idx, binary_seq)] = len(self._choices)
            self._choices.append((block_idx, binary_seq))
    self._propagate_types()

  def _create_drop_path_choices(self):
    if not self._drop_path:  # Drop path was turned off.
      return np.zeros(shape=[len(self._choices)], dtype='int32')
    elif np.random.uniform() < self._p_local_drop_path:
      # Local drop-path (make each choice independantly at random.)
      choices = np.random.uniform(size=[len(self._choices)])
      drop_base = choices < self._p_drop_base_case
      drop_recursive = np.logical_and(
          choices < (self._p_drop_base_case + self._p_drop_recursive_case),
          np.logical_not(drop_base))
      return (np.int32(drop_base)*self._JUST_RECURSE +
              np.int32(drop_recursive)*self._JUST_BASE)
    else:
      # Global (pick a single column.)
      column = np.random.randint(self._fractal_block_depth)
      return np.array(
          [self._JUST_RECURSE if len(binary_seq) < column else self._JUST_BASE
           for _, binary_seq in self._choices],
          dtype='int32')

  @property
  def drop_path(self):
    return self._drop_path

  @drop_path.setter
  def drop_path(self, value):
    self._drop_path = bool(value)

  def _propagate_types(self):
    for _ in xrange(2):
      for child in six.itervalues(self._children):
        self.set_io_types(child)

  def _create_variables(self):
    if self._drop_path_choices is None:
      self._drop_path_choices, = tf.py_func(
          self._create_drop_path_choices, [], [tf.int32],
          stateful=True, name='calculate_drop_path')

  def _instantiate_subnet(self, batch, block_idx, seq_prefix):
    def zeros_fn():
      return tf.zeros_like(batch)
    def base_case_fn():
      return self._children[block_idx, seq_prefix](batch)
    def recursive_case_fn():
      first_subnet = self._instantiate_subnet(
          batch, block_idx, seq_prefix + (0,))
      return self._instantiate_subnet(
          first_subnet, block_idx, seq_prefix + (1,))
    if len(seq_prefix) == self._fractal_block_depth:
      return base_case_fn()
    else:
      choice = self._drop_path_choices[self._choice_id[(block_idx, seq_prefix)]]
      base_case = tf.cond(
          tf.not_equal(choice, self._JUST_RECURSE), base_case_fn, zeros_fn)
      base_case.set_shape(batch.get_shape())
      recursive_case = tf.cond(
          tf.not_equal(choice, self._JUST_BASE), recursive_case_fn, zeros_fn)
      recursive_case.set_shape(batch.get_shape())
      cases = [
          (tf.equal(choice, self._BOTH),
           lambda: self._mixer(base_case, recursive_case)),
          (tf.equal(choice, self._JUST_BASE), lambda: base_case),
          (tf.equal(choice, self._JUST_RECURSE), lambda: recursive_case)]
      result = tf.case(cases, lambda: base_case)
      result.set_shape(batch.get_shape())
      return result

  def _process_batch(self, batch):
    for block_idx in xrange(self._num_fractal_blocks):
      batch = self._instantiate_subnet(batch, block_idx, ())
    return batch


class ScopedLayer(Layer):
  """Create a Fold Layer that wraps a TensorFlow layer or RNN cell.

  The default TensorFlow mechanism for weight sharing is to use
  tf.variable_scope, but this requires that a scope parameter be passed
  whenever the layer is invoked.  ScopedLayer stores a TensorFlow layer,
  along with its variable scope, and passes the scope appropriately.
  For example:

  ```
  gru_cell1 = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=16), 'gru1')
  ... td.RNN(gru_cell1) ...
  ```
  """

  def __init__(self, layer_fn, name_or_scope=None):
    """Wrap a TensorFlow layer.

    Args:
      layer_fn: A callable that accepts and returns nests of batched tensors. A
        nest of tensors is either a tensor or a sequence of nests of tensors.
        Must also accept a `scope` keyword argument. For example, may be an
        instance of `tf.contrib.rnn.RNNCell`.
      name_or_scope: A variable scope or a string to use as the scope name.
    """
    if name_or_scope is None:
      if hasattr(layer_fn, '__name__'):
        name_or_scope = layer_fn.__name__
      elif hasattr(layer_fn, 'func') and hasattr(layer_fn.func, '__name__'):
        # If layer_fn is e.g. a functools.partial.
        name_or_scope = layer_fn.func.__name__
    super(ScopedLayer, self).__init__(name_or_scope=name_or_scope)
    self._layer_fn = layer_fn
    if isinstance(layer_fn, tf.contrib.rnn.RNNCell):
      self.set_output_type((layer_fn.output_size, layer_fn.state_size))

  @property
  def state_size(self):
    return self._layer_fn.state_size

  @property
  def output_size(self):
    return self._layer_fn.output_size

  def __call__(self, *args):
    result = self._layer_fn(*args, scope=self._vscope)
    self._vscope.reuse_variables()   # Reuse scope on subsequent calls
    return result
