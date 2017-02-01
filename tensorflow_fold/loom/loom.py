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
"""Loom, the low-level API for TensorFlow Fold.

This file contains the code that sets up the TensorFlow graph, and the
Python Weaver API for scheduling operations on that graph.  The
scheduling code itself lives in weaver.cc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import collections
import functools
import numbers
import re

# import google3
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow_fold.loom import deserializing_weaver_op
from tensorflow_fold.loom import loom_pb2
from tensorflow_fold.loom import pywrapweaver


TENSOR_IDX_T = tf.int32

_TypeShape = collections.namedtuple('TypeShape', ['dtype', 'shape', 'tag'])


@functools.total_ordering
class TypeShape(_TypeShape):
  """A type and shape defining a kind of tensor.

  TypeShapes are used to specify the argument types and return types of
  operations on the emulated graph.

  Attributes:
    dtype: What type the cells of the tensor should have. Should be a
      `tf.DType`, or stringified version thereof (e.g. 'int64').
    shape: A tuple of integers for the shape of the tensor.
    tag: A name for the type.  Two TypeShapes with the same dtype and
      shape, but with different names, will be treated as different types.
  """

  def __new__(cls, dtype=None, shape=None, tag='', tensor=None):
    if tensor is not None:
      if dtype is not None:
        raise TypeError('Specify only one of tensor and dtype.')
      if shape is not None:
        raise TypeError('Specify only one of tensor and shape.')
      dtype = tensor.dtype
      shape = tensor.get_shape().as_list()
    elif not (isinstance(dtype, tf.DType) or
              isinstance(dtype, six.string_types)):
      raise TypeError('%r is not a tf.DType or string' % (dtype,))
    dtype = tf.as_dtype(dtype).base_dtype.name
    if not all(isinstance(s, numbers.Integral) and s >= 0 for s in shape):
      raise TypeError('shape must be non-negative integers: %s' % shape)
    shape = tuple(int(s) for s in shape)
    if not isinstance(tag, six.string_types):
      raise TypeError('A TypeShape tag must be a string; type of %r is %s' %
                      (tag, type(tag)))
    return _TypeShape.__new__(cls, dtype, shape, tag)

  @property
  def dtype_enum(self):
    """The dtype of this TypeShape as an enum."""
    return tf.as_dtype(self.dtype).as_datatype_enum

  def tensor_flow_name(self):
    """Makes a name for the TypeShape usuable as a TF op name."""
    # Just to make sure we don't include characters that aren't allowed in TF
    # opnames, we strip all the non-word characters from the tag.
    tag = re.sub(r'\W+', '', self.tag)
    if tag: tag = '_'+tag
    return '%s_%s%s' % (self.dtype, '_'.join(str(x) for x in self.shape), tag)

  def __str__(self):
    """Stringifies all the fields of the typeshape for pretty printing."""
    return 'TypeShape{dtype:%s shape:%s tag:"%s"}' % (
        self.dtype, self.shape, self.tag)


@six.add_metaclass(abc.ABCMeta)
class LoomOp(object):
  """Class to be subclassed for defining Loom operations."""

  def __init__(self, input_type_shapes, output_type_shapes):
    """Setup the input and output typeshapes for the LoomOp, with checking."""
    if not input_type_shapes:
      raise TypeError('Every LoomOp must have at least one input.')
    if not output_type_shapes:
      raise TypeError('Every LoomOp must have at least one output.')
    if not all(isinstance(ts, TypeShape) for ts in input_type_shapes):
      raise TypeError('All of input_type_shapes must be TypeShapes.')
    if not all(isinstance(ts, TypeShape) for ts in output_type_shapes):
      raise TypeError('All of output_type_shapes must be TypeShapes.')
    self._input_type_shapes = input_type_shapes
    self._output_type_shapes = output_type_shapes

  @property
  def input_type_shapes(self):
    """A list of TypeShapes of the arguments to this LoomOp."""
    return self._input_type_shapes

  @property
  def output_type_shapes(self):
    """A list of TypeShapes of the return values of this LoomOp."""
    return self._output_type_shapes

  @abc.abstractmethod
  def instantiate_batch(self, inputs):
    """Instantiate the TensorFlow ops for running this loom Op.

    Args:
      inputs: The types of the inputs should match up with the types
        of the input_type_shapes. The shapes of the inputs should also match
        with the shapes of the TypeShapes with one additional dimension at the
        start for the batch size.

    Returns:
      A list of TensorFlow objects which will contain tensors whose dimensions
      match up with those of the corresponding output_type_shapes (with an
      additional dimension at the start for the batch size as before.)
    """
    _ = inputs
    raise NotImplementedError(
        'LoomOp needs a definition for instantiate_batch.')


class PassThroughLoomOp(LoomOp):
  """Op which leaves its input alone.

  This Op exists to allow Loom objects to pass values to higher depths.

  For example, suppose we wanted to run an expression like "(a+b)+c" where a, b,
  and c are constants which have depth 0. The first "+" would take place at
  depth 1, and the second "+" would take place at depth 2. However, "(a+b)" is
  now a level above "c". To fix this, c needs to be passed through depth 1
  so it can be an argument to the second "+" at depth 2.
  """

  def __init__(self, type_shape):
    super(PassThroughLoomOp, self).__init__([type_shape], [type_shape])

  def instantiate_batch(self, inputs):
    """Leaves the inputs alone."""
    return inputs


class Loom(object):
  """A Loom lets TensorFlow quickly run dynamic graphs, with back-prop.

  A Loom builds a TensorFlow graph which can be made to emulate any
  graph made from the `LoomOps` passed in via the `named_ops` argument, tensors
  passed in via the `named_tensors`, and constants (created at runtime, along
  with the graphs the Loom is to emulate.)

  After creating a Loom object (see `Loom.__init__`), you can connect its
  outputs to a larger TensorFlow graph by getting its `output_tensor` to get the
  batch tensor containing all outputs of a certain shape from Loom, and then
  call arbitrary further TensorFlow ops on that (Loom is also fully compatible
  with gradients privided the LoomOps it was constructed with are.)

  At runtime, you can construct an emulated graph by calling the
  `Loom.make_weaver` method and using the resulting Weaver object's API to
  build up a graph. To run the emulated graph you can then call the Weaver's
  build_feed_dict method to get a dictionary that can be passed to
  TensorFlow's `Tensor.eval` or `Session.run` methods.

  ##### Batch input mode

    In batch input mode, Loom reads inputs from a batched input tensor (which
    will contain a batch of elements of the appropriate TypeShape) instead of
    from constants passed in via Weaver.  The main advantage is that is that
    inputs fed in via batch input mode can be backpropagated through, unlike
    constants.  Batch input mode can be turned on on a per TypeShape basis,
    given a tensor which will contain a batch of elements of the right
    TypeShape.  If a TypeShape is put into batch input mode, then, when using
    the Weaver, instead of manually declaring constants you can simply refer to
    elemements of the batch by index.  Additional advantages of using batch
    input mode are that bulky constants need not bloat the weaver messages and
    they may be changed at runtime.

  ##### Bypass modes

    Loom can bypass python graph specification should it become a bottle-neck.
    If the `Loom` is constructed with an `input_tensor`, that input tensor
    should contain serialized `WeaverMessages` (which specify a graph or
    schedule for the `Loom` to execute.  See loom.proto for the definition.)
    This `input_tensor` might, for example, read pre-computed `WeaverMessages`
    from a TF-Record file, or pull them from a Tensorflow queue (which would
    allow the computation of schedules to be parallelized across multiple
    processes.)

    Alternatively, if you want all scheduling to happen at runtime in C++ you
    can subclass `WeaverOpBase` (see `weaver_op_base.h`).  The Loom's
    constructor should then be given a `weaver_op`, a function which will be
    passed Loom's metadata and instantiate the user's custom op.

  ##### `direct_feed_dict` mode
    Loom can also be run in direct_feed_dict mode.  This bypasses
    DeserializingWeaverOp (a TensorFlow op which turns serialized
    `WeaverMessages` in a string tensor into the sorts of tensors that can drive
    the Loom), copying constant tensors directly into the feed_dict, and
    extracting the wiring vectors directly into the feed_dict.  This mode is
    useful for evaluating the cost of DeserializingWeaverOp.  `direct_feed_dict`
    mode is not compatible with the bypass modes above because it avoids putting
    any of its schedules into a string tensor as a `WeaverMessage` entirely.

  ##### Implementation details

    ![wiring diagram](wiring.png)

    A Loom object wraps a sequence of loom layers each of which can be directed
    to perform a variety of operations. In order to carry a variety of types at
    runtime, the connections from one loom layer to the next include one state
    Tensor containing a batch of values of each supported TypeShape. (E.g. if
    float32 vectors of length 3 are supported, there will be a float32 Tensor
    with dimensions (*, 3) to carry 3-vectors from each layer to the next.)

    Each layer has three sub-layers: (1) an input demultiplexer that extracts
    arguments from the input state tensors and routes them to sub-layer 2 (using
    tf.gather.)  (See weaver.h for the scheduling code that transforms a graph
    of loom operations into lists of indices for these gather ops.) (2) a
    collection of TensorFlow subnetworks, one for each operation type supported
    by the Loom.  These are instantiated with LoomOp.instantiate_batch using the
    inputs from sub-layer 1.  (3) an output multiplexer that combines the
    outputs of the LoomOps (using tf.concat) to form the state tensors to be
    passed on to sub-layer 1 of the next Unit.

    (See `Loom._construct_loom_layer` for the implementation.)

    By default, `max_depth` is unspecified and Loom wraps a single loom layer in
    a `tf.while_loop`.  If max_depth is specified, loom unrolls the loop by
    instantiating `max_depth` copies the loom layer.
  """

  def __init__(self, max_depth=None, named_tensors=None, named_ops=None,
               batch_inputs=None, extra_type_shapes=None, dry_run=False,
               parallel_iterations=None, back_prop=None, swap_memory=None,
               direct_feed_dict=False, loom_input_tensor=None, weaver_op=None):
    """Constructs a Loom.

    While this constructor has many arguments, the only arguments most users
    will care about are `named_ops`, `named_tensors`, `dry_run`,
    `loom_input_tensor` and possibly `weaver_op`.

    To create a Loom object, the only mandatory argument is `named_ops` (a
    dictionary mapping strings to `LoomOps`) specifying the collection of
    operations the Loom should support.

    Specifiying `named_tensors` allows the `Loom` to construct graphs that refer
    to the provided TensorFlow tensors.  The advantage of using a named tensor
    instead of a Loom constant is that the named tensor can be backpropped
    through.

    Specifying `loom_input_tensor` causes the `Loom` to read its schedules
    (`WeaverMessages`) from external sources.  Specifying `weaver_op` allows
    `Loom` to compute them on the fly in C++.  See the class docstring section
    named "Bypass Modes" for the motivation for this feature.

    Specifying `dry_run` creates the Loom without constructing the associated
    TensorFlow graph.  This is useful when the loom is only going to be used to
    construct `WeaverMessages` to drive another instance of the same loom.

    Args:
      max_depth: An optional integer depth to unroll the generic network to.  If
        absent, Loom uses a `tf.while_loop`.  `max_depth` is provided for
        compatibility with old versions of TensorFlow with bad support for
        `tf.while_loop` and for debugging purposes.
      named_tensors: An optional dictionary mapping strings to Tensors. (Named
        tensors are effectively zero argument LoomOps.)  Each value of
        `named_tensors` must be either a tf.Tensor or a tuple of the form
        (tf.Tensor, str) with the string specifying a TypeShape tag.
      named_ops: A mandatory dictionary mapping strings to LoomOp objects (the
        set of operations the Loom should support.)
      batch_inputs: An optional dictionary mapping TypeShapes to Tensors.  Each
        Tensor in the dictionary should have the type and shape to contain a
        batch of things of that TypeShape stacked along dimension 0.
      extra_type_shapes: An optional iterable containing extra TypeShapes that
        may not be inputs or outputs of LoomOps but that the Loom should support
        anyway.
      dry_run: Boolean. If true, don't build the TensorFlow graph (and make the
        output tensors be dummy constants.)  This is useful for rapid testing in
        situtions where building the TensorFlow graph is expensive (eg. large
        max_depth) or when the objective is to construct schedules and serialize
        them as `WeaverMessages` for later use.
      parallel_iterations: Integer. tf.while_loop's parallel_iterations option,
        which caps the number of different depths at which ops could run in
        parallel.  Only applies when max_depth=None. Default: 10.
      back_prop: Boolean. tf.while_loop's back_prop option, which enables
        gradients. Only applies when max_depth=None.  Default: True.
      swap_memory: Boolean. Whether to use tf.while_loop's swap_memory option,
        which enables swapping memory between GPU and CPU at the possible
        expense of some performance. Only applies when max_depth=None. Default:
        False.
      direct_feed_dict: Boolean. If true, this loom doesn't create a loom_input
        tensor for WeaverMessages, and instead creates placeholders for the
        wiring diagrams. Default: False.
      loom_input_tensor: An optional string Tensor from which to read
        WeaverMessages which specify how to wire the loom.  If more than one is
        present they will be merged (auto-merge is provided so that
        WeaverMessages for individual inputs can be cached in advance while
        still using random mini-batches at run-time.)  Mutally exclusive with
        `weaver_op`.
      weaver_op: An optional callable which constructs a TensorFlow op to
        produce inputs for the loom.  Mutually exclusive with
        `loom_input_tensor`.  If absent, the loom acts as though `weaver_op`
        were a function creating a `deserializing_weaver` op which consumes
        `WeaverMessages` from `loom_input_tensor`.  The callable will be called
        with three keyword arguments named `metadata`, `constant_types`, and
        `num_type_shapes` (because these are the three attributes any op
        descending from `WeaverOpBase` requires to be instantiated.)

    Raises:
      TypeError: If `named_ops` is not provided.
      TypeError: If more than one tagged TypeShape has the same tag.
    """
    if named_ops is None:
      raise TypeError('named_ops is a mandatory argument.')

    # max_depth is going to be put into the LoomMetadata proto which uses the
    # special value -1 to indicate that Loom's TensorFlow graph will be
    # constructed using a while loop (and therefore have no fixed maximum
    # depth.)
    if max_depth is None: max_depth = -1

    # _max_depth: the maximum operation depth supported by the loom (or -1 for
    # while loop.)
    #
    # If _max_depth is not -1, it is the maximum nesting depth for the graph
    # loom can emulate. For example f(f(c)) (where c is a constant and f is an
    # loom operation) would be allowed if _max_depth is 2 but not if _max_depth
    # is 1.
    self._max_depth = max_depth

    if named_tensors is None: named_tensors = {}
    if batch_inputs is None: batch_inputs = {}
    if parallel_iterations is None: parallel_iterations = 10
    if back_prop is None: back_prop = True
    if swap_memory is None: swap_memory = False

    # _batch_inputs: a dictionary mapping typeshapes to tensors containing
    # batches of that typeshape to be used as inputs.
    self._batch_inputs = batch_inputs

    # _dry_run: if true don't build the TF graph (all output tensors get
    # replaced with one set of zeros.)
    self._dry_run = dry_run

    # _parallel_iterations, _back_prop, _swap_memory: options for tf.while_loop.
    self._parallel_iterations = parallel_iterations
    self._back_prop = back_prop
    self._swap_memory = swap_memory

    # _direct_feed_dict: a bool specifying whether to construct a graph which
    # bypasses the deserializing_weaver_op.
    self._direct_feed_dict = direct_feed_dict

    if direct_feed_dict:
      if loom_input_tensor is not None:
        raise TypeError(
            'direct_feed_dict and loom_input_tensor are incompatible.')
      if weaver_op is not None:
        raise TypeError('direct_feed_dict and weaver_op are incompatible.')

    # _loom_input_tensor: a tensor which ought to hold a single serialized
    # WeaverMessage specifying the loom's wiring diagram.
    if not direct_feed_dict:
      if weaver_op is None:
        if loom_input_tensor is None:
          loom_input_tensor = tf.placeholder(
              'string', name='LoomInput')
        def weaver_from_input_tensor(**kwargs):
          return deserializing_weaver_op.deserializing_weaver(
              self._loom_input_tensor, **kwargs)
        weaver_op = weaver_from_input_tensor
      else:
        if loom_input_tensor is not None:
          raise TypeError('You can specify at most one of loom_input_tensor '
                          'or weaver_op.')
    self._loom_input_tensor = loom_input_tensor
    self._weaver_op = weaver_op

    self._setup_type_shapes(named_ops, extra_type_shapes)

    self._setup_named_tensors(named_tensors)

    self._setup_loom_ops(named_ops)

    self._setup_metadata()

    self._setup_network()

  def _setup_type_shapes(self, named_ops, extra_type_shapes):
    """Setup fields to keep track of our typeshapes."""
    type_shape_set = set()
    for op in six.itervalues(named_ops):
      type_shape_set.update(op.input_type_shapes)
      type_shape_set.update(op.output_type_shapes)
    if extra_type_shapes is not None:
      type_shape_set.update(extra_type_shapes)

    # _type_shapes: a list of all the typeshapes this loom object supports.
    self._type_shapes = sorted(type_shape_set)

    # Enforce uniqueness for non-empty TypeShape tags.
    non_empty_tags = set()
    for ts in self._type_shapes:
      if ts.tag:
        if ts.tag in non_empty_tags:
          raise TypeError('Tags on tagged TypeShapes must be unique; '
                          '%s occured more than once.' % (ts.tag,))
        else:
          non_empty_tags.add(ts.tag)

    # _type_shape_to_idx: a dict mapping TypeShape objects to their indices in
    #  '_type_shapes'.
    self._type_shape_to_idx = {ts: idx for idx, ts in
                               enumerate(self._type_shapes)}

  def _setup_named_tensors(self, named_tensors):
    """Setup fields to track the named tensors."""
    # _ts_idx_to_named_tensors: a list containing all the tensors in
    # 'named_tensors' separated out by their TypeShape.
    self._ts_idx_to_named_tensors = [[] for _ in self._type_shapes]

    # _ts_idx_to_tensor_names: a list containing all the names in
    # 'named_tensors' separated out by their TypeShape.
    self._ts_idx_to_tensor_names = [[] for _ in self._type_shapes]

    for name, tensor in sorted(six.iteritems(named_tensors)):
      if isinstance(tensor, tuple):
        tensor, tag = tensor
        ts = TypeShape(tensor=tensor, tag=tag)
      else:
        ts = TypeShape(tensor=tensor)

      ts_idx = self._type_shape_to_idx[ts]
      self._ts_idx_to_named_tensors[ts_idx].append(tensor)
      self._ts_idx_to_tensor_names[ts_idx].append(name)

  def _pass_through_name(self, ts):
    return '_pass_through_' + ts.tensor_flow_name()

  def _setup_loom_ops(self, named_ops):
    """Sets up mappings between loom ops, loom op ids, and loom op names."""
    # Make a PassThroughOp for each TypeShape. Then set up mappings between
    # names, op indices and ops with the PassThrough ops having the same
    # indices as their type_shapes and no names.
    pass_through_ops = [PassThroughLoomOp(ts) for ts in self._type_shapes]

    non_passthrough_op_names = sorted(six.iterkeys(named_ops))

    # _loom_op_names: a list of names for all the ops (including autogenerated
    # names for the PassThroughLoomOps for debugging purposes.)
    #
    # The first len(self._type_shapes) ops are forced to be the passthrough ops
    # for the corresponding TypeShape.  This is enforced in VerifyLoomMetadata.
    self._loom_op_names = (
        [self._pass_through_name(ts) for ts in self._type_shapes] +
        non_passthrough_op_names)

    # _loom_ops: a list of the supported LoomOps (including autogenerated
    # PassThroughOps)
    self._loom_ops = (
        pass_through_ops + [named_ops[k] for k in non_passthrough_op_names])

    # _loom_total_args: The sum of the number of arguments across all loom ops.
    self._loom_total_args = sum(
        len(op.input_type_shapes) for op in self._loom_ops)

    # _loom_op_name_to_idx: a dict mapping the names in '_op_names' back to into
    # indices usable with '_ops' and '_op_names'.
    self._loom_op_name_to_idx = {
        name: idx for idx, name in enumerate(self._loom_op_names)}

  def _setup_metadata(self):
    """Construct the serialized metadata about this loom for the scheduler."""
    # loom_metadata is what we use to pass all the information about
    # the loom (max_depth, which typeshapes are supported, and the signatures of
    # the LoomOps) to scheduler.cc
    loom_metadata = loom_pb2.LoomMetadata()
    loom_metadata.max_depth = self._max_depth
    for ts, tensor_names in zip(
        self._type_shapes, self._ts_idx_to_tensor_names):
      type_shape_metadata = loom_metadata.type_shape_metadata.add()
      type_shape_metadata.dtype = ts.dtype_enum
      type_shape_metadata.shape.extend(ts.shape)
      type_shape_metadata.tag = ts.tag
      type_shape_metadata.name = str(ts)  # Debug string.
      type_shape_metadata.tensor_names.extend(tensor_names)
      type_shape_metadata.is_batch_input = (
          (ts in self._batch_inputs) or self._direct_feed_dict)

    for op_name, op in zip(self._loom_op_names, self._loom_ops):
      op_metadata = loom_metadata.op_metadata.add()
      op_metadata.name = op_name
      op_metadata.input_ts_idx.extend(
          self._type_shape_to_idx[ts] for ts in op.input_type_shapes)
      op_metadata.output_ts_idx.extend(
          self._type_shape_to_idx[ts] for ts in op.output_type_shapes)

    self._loom_metadata_str = (
        loom_metadata.SerializeToString())

  def _setup_network(self):
    """Build the TensorFlow network that can emulate Loom graphs."""
    if self._dry_run:
      self._output = [tf.constant(np.zeros((1,)+ts.shape, dtype=ts.dtype))
                      for ts in self._type_shapes]
      return

    if self._direct_feed_dict:
      self._arg_wiring_concat = tf.placeholder(
          TENSOR_IDX_T, name='arg_wiring_concat')
      self._arg_wiring_slice_starts = tf.placeholder(
          TENSOR_IDX_T, name='arg_wiring_slice_starts')
      self._arg_wiring_slice_sizes = tf.placeholder(
          TENSOR_IDX_T, name='arg_wiring_slice_sizes')
      self._output_wirings = [
          tf.placeholder(TENSOR_IDX_T, name='output_wirings_%d' % ts_idx)
          for ts_idx in xrange(len(self._type_shapes))]
      self._constants = [
          tf.placeholder(ts.dtype, name='constants_%d' % ts_idx)
          for ts_idx, ts in enumerate(self._type_shapes)]
    else:
      # See REGISTER_WEAVER_OP in weaver_op_base.h for the definitions of the
      # outputs in the destructuring assignment below.
      (self._arg_wiring_concat,
       self._arg_wiring_slice_starts,
       self._arg_wiring_slice_sizes,
       self._output_wirings,
       self._constants) = self._weaver_op(
           metadata=self._loom_metadata_str,
           constant_types=[tf.as_dtype(ts.dtype) for ts in self._type_shapes],
           num_type_shapes=len(self._type_shapes))
    # _arg_wiring_concat: an integer vector Tensor containing all the wirings
    # for the current schedule concatenated together.  They are sorted
    # lexically, by (depth, op_idx, arg_idx).  This means that
    # _arg_wiring_concat consists of max_depth*self._loom_total_args, vectors
    # concatenated together.  (Here max_depth refers to the final max_depth of
    # the emulated graph, not -1 in the event that the Loom was instantiated
    # with a while_loop.)
    #
    # _arg_wiring_slice_starts and _arg_wiring_slice_sizes: these are integer
    # vector Tensors of length max_depth*self._loom_total_args that specify how
    # to split _arg_wiring_concat back apart into wirings for each (depth,
    # op_idx, arg_idx).
    #
    # The rationale for concatenating all the wiring diagrams together
    # like this is that in order to support tf.while_loop, we need to create a
    # tensor which produces the appropriate wiring diagram in a way that depends
    # on the current depth (this is accomplished using tf.slice in
    # _construct_loom_layer.)
    #
    # _output_wirings: A list of integer vector Tensors, one for each TypeShape.
    # These vectors select which elements of the final state tensor end up in
    # the Loom's `output_tensor`s.
    #
    # _constants: A list of Tensors, one for each TypeShape.  Each of these
    # Tensors should have the dtype of the corresponding TypeShape.  The
    # contents should be the stacked set of constants declared for that
    # TypeShape.

    # For each TypeShape, if it's in batched input mode, we use the user
    # provided tensor as the input.  Otherwise, we take the constants from the
    # weaver.
    inputs = self._constants
    for ts_idx, ts in enumerate(self._type_shapes):
      if ts in self._batch_inputs:
        inputs[ts_idx] = self._batch_inputs[ts]

    # iteration of building up the graph, state will contain tensors
    # whose rows will be the objects passed from each depth to the next one of
    # the appropriate shapes.
    state = []
    for inputs_tensor, named_tensors in (
        zip(inputs, self._ts_idx_to_named_tensors)):
      if not named_tensors:
        state.append(inputs_tensor)
      else:
        state.append(tf.concat([tf.stack(named_tensors), inputs_tensor], 0))

    # This block builds up the static graph that consumes Loom's wiring
    # diagrams and emulates the dynamic network.
    #
    # Note: the code that computes wiring diagrams lives in scheduler.cc for
    # efficiency reasons.
    if self._max_depth == -1:  # For dynamic max_depth we use tf.while.
      current_max_depth = (
          tf.size(self._arg_wiring_slice_starts) // self._loom_total_args)
      def loop_conditional(depth, *unused_state):
        return tf.less_equal(depth, current_max_depth)
      def loop_body(depth, *state):
        new_depth = tf.add(depth, 1, name='increment_depth')
        new_state = self._construct_loom_layer(depth, state)
        return [new_depth] + new_state
      initial_depth = tf.constant(1, name='initial_depth')
      state = tf.while_loop(loop_conditional, loop_body,
                            [initial_depth] + state,
                            parallel_iterations=self._parallel_iterations,
                            back_prop=self._back_prop,
                            swap_memory=self._swap_memory)[1:]
    else:  # For explicit max_depth we unroll the loop.
      for depth in xrange(1, self._max_depth+1):
        with tf.name_scope('loom_depth_%03d' % depth):
          state = self._construct_loom_layer(depth, state)

    # _output: The output tensors of the loom, indexed by TypeShape.
    with tf.name_scope('output_gathers'):
      self._output = [
          tf.gather(s, w, name=self._type_shapes[ts_idx].tensor_flow_name())
          for ts_idx, (s, w) in enumerate(zip(state, self._output_wirings))]

    # Make sure the output tensors know what shape they're supposed to be.
    for type_shape, output in zip(self._type_shapes, self._output):
      output.set_shape((None,) + type_shape.shape)

  def _construct_loom_layer(self, depth, state):
    """Builds one unit of the loom's graph.

    A Loom unit is a TensorFlow graph that performs all the operations scheduled
    on the Loom at a given depth.

    Args:
      depth: An integer or integer tensor containing the current depth.
      state: A list of tensors (one for each TypeShape) which will contain
      batches of things of that TypeShape.

    Returns:
      A list of tensors (one for each TypeShape) which will contain batches of
      things of that TypeShape. (The input to the next loom layer.)

    Raises:
      ValueError: If a LoomOp's instantiate_batch method returns Tensors of the
        wrong DataType or shape.
    """
    # Segments to be concatenated together to form the output state (indexed by
    # TypeShape ID.)
    new_state_segments = [[] for _ in state]

    # Note: `start_wire_pos` might be a tensor or an integer.
    start_wire_pos = (depth - 1) * self._loom_total_args

    wire_pos_offset = 0  # `wire_pos_offset` is an integer.

    for op_idx, op in enumerate(self._loom_ops):
      with tf.name_scope(self._loom_op_names[op_idx]):
        arg_inputs = []
        for arg_idx, arg_ts in enumerate(op.input_type_shapes):
          with tf.name_scope('arg_%d' % arg_idx):
            # wire_pos: a tensor or integer specifying which argument's wiring
            # diagram we wish to extract from `arg_wiring_concat`
            wire_pos = start_wire_pos + wire_pos_offset
            wire_pos_offset += 1

            # slice_start: a vector of length 1 containing the starting postion
            # starting postion of this argument's wiring in arg_wiring_concat.
            slice_start = tf.slice(
                self._arg_wiring_slice_starts, [wire_pos], [1])

            # slice_size: a vector of length 1 containing the starting postion
            # starting postion of this argument's wiring in arg_wiring_concat.
            slice_size = tf.slice(
                self._arg_wiring_slice_sizes, [wire_pos], [1])

            # arg_wiring: a tensor specifying the indices the of several tensors
            # (within the state vector corresponding to the TypeShape of arg).
            # This batch of tensors get will be passed to argument `arg_idx` of
            # op `op` at depth `depth`.
            #
            # The contents of this tensor will be the same as the vector
            # computed by Weaver::GetWiring(depth, op_idx, arg_idx) in C++.
            arg_wiring = tf.slice(
                self._arg_wiring_concat, slice_start, slice_size)

            arg_ts_idx = self._type_shape_to_idx[arg_ts]

            # This tf.gather constructs sub-layer (1) of the loom layer.
            # (See the class doc-string section on Implementation Details)
            #
            # This gather selects which batch of tensors get passed to argument
            # `arg_idx` of op `op` at depth `depth`.
            arg_input = tf.gather(state[arg_ts_idx], arg_wiring)

            # We sure the inputs are tagged with the correct shape before
            # attempting to concatenate.
            arg_input.set_shape((None,) + arg_ts.shape)
            arg_inputs.append(arg_input)

        # This call to op.instantiate_batch constructs sub-layer (2) of the loom
        # layer.
        op_outputs = op.instantiate_batch(arg_inputs)

        for output_idx, (output, output_ts) in enumerate(
            zip(op_outputs, op.output_type_shapes)):
          # Do some sanity checking to make sure instantiate_batch output
          # Tensors of the right type and shape.
          if not isinstance(output, tf.Tensor):
            raise TypeError('Op %s returns non-Tensor output %r' %
                            (self._loom_op_names[op_idx], output))
          try:
            output.set_shape((None,) + output_ts.shape)  # Check shape.
          except ValueError as e:
            raise ValueError('Op %s output %d: %s' % (
                self._loom_op_names[op_idx], output_idx, e))
          if output.dtype.base_dtype.name != output_ts.dtype:
            raise ValueError('Op %s output %d: expected dtype %s got %s' % (
                self._loom_op_names[op_idx], output_idx,
                output_ts.dtype, output.dtype.base_dtype.name))

          # Append this output of the arg to the list of segments of the
          # appropriate typeshape.
          #
          # Note: The segments of a given typeshape will end up sorted lexically
          # by (op_idx, op_output_idx).  weaver.cc depends on this fact when
          # computing offsets in order to transform its graph into a wiring
          # diagram (See Weaver::Finalize)
          output_ts_idx = self._type_shape_to_idx[output_ts]
          new_state_segments[output_ts_idx].append(output)

    with tf.name_scope('concat'):
      # This concat constructs sub-layer (3) of the loom layer.
      #
      # We need to concatenate all the outputs of the same type-shape
      # together so that the next layer can gather over them.
      # This allows any LoomOp with an input of some type_shape to get its
      # input from any output of any LoomOp (provided it is of the same
      # TypeShape.)
      return [
          tf.concat(
              s, 0, name=self._type_shapes[ts_idx].tensor_flow_name())
          for ts_idx, s in enumerate(new_state_segments)
      ]

  def output_tensor(self, type_shape):
    """Return the output Tensor for the given TypeShape.

    Returns:
      An output Tensor has one more dimension than the type_shape (the first
      dimension is the one along which all the values of that TypeShape have
      been concatenated. For example if the TypeShape were `('float32', (3, 5))`
      we'd return a `float32` tensor whose dimensions are `(*, 3, 5)` where the
      `*` can be any number. The Tensor will contains a batch of all 3x5 matrix
      results passed to Weaver's `build_feed_dict`.

    Args:
      type_shape: The TypeShape we want to look up.
    """
    return self._output[self._type_shape_to_idx[type_shape]]

  @property
  def input_tensor(self):
    """The input tensor for this loom.

    Returns:
      The Loom's input tensor.

    Raises: TypeError if `direct_feed_dict` mode was enabled when the loom was
      constructed.
    """
    if self._direct_feed_dict:
      raise TypeError('This loom has direct_feed_dict set, '
                      'so it has no input tensor')
    return self._loom_input_tensor

  @property
  def type_shapes(self):
    """The list of TypeShapes used by this loom."""
    return self._type_shapes

  def make_weaver(self):
    """Constructs a Weaver object for the current loom."""
    return Weaver(self)

  def deserialize_weaver(self, serialized_weaver):
    """Turn a serialized `WeaverMessage` proto into an Python Weaver object."""
    deserialized = self.make_weaver()
    # pylint: disable=protected-access
    deserialized._deserialize(serialized_weaver)
    # pylint: enable=protected-access
    return deserialized


# Weaver is a friend of Loom and needs to be able to read its fields.
# pylint: disable=protected-access
class Weaver(object):
  """A (partially constructed) wiring diagram or schedule for a Loom object.

  This object is a user-friendly wrapper for the `tensorflow::fold::Weaver` C++
  object.

  The `build_feed_dict` method uses the Weaver to construct a dict directing the
  Loom to behave like the diagram.

  Alternatively, the user can call the `serialize` method to serialize the
  schedule to a string in order to eventually pass it into a Loom's
  `input_tensor`.
  """

  def __init__(self, loom):
    """Sets up the Weaver Object.

    Args:
      loom: The Loom object backing this Weaver.

    Raises:
      TypeError: If loom is not a Loom object.
      AssertionError: If the Weaver object cannot be constructed or if any of
        its named tensors cannot be retrieved.
    """
    if not isinstance(loom, Loom):
      raise TypeError('A weaver must be passed a Loom on construction.')
    self._loom = loom

    self._weaver = pywrapweaver.Weaver(self._loom._loom_metadata_str)
    if self._weaver.error_string():
      raise AssertionError('Failed to create weaver: ',
                           self._weaver.error_string())

    self._constants = [[] for _ in self._loom._type_shapes]

    self._tensor_name_to_result = {}

    for ts_idx, names in enumerate(self._loom._ts_idx_to_tensor_names):
      for name_idx, name in enumerate(names):
        named_tensor = self._weaver.GetNamedTensor(ts_idx, name_idx)
        if named_tensor == -1:
          raise AssertionError(
              'Failed to GetNamedTensor in Weaver wrapper for %s error: %s.' %
              (name, self._weaver.error_string()))
        self._tensor_name_to_result[name] = named_tensor

    # Set up the syntactic sugar:
    for name, result in six.iteritems(self._tensor_name_to_result):
      self._safe_set_attr(name, result)

    for op_idx, op_name in enumerate(self._loom._loom_op_names):
      self._safe_set_attr(op_name,
                          functools.partial(self._call_op_sugar, op_idx))

  def _safe_set_attr(self, name, value):
    if hasattr(self, name):
      print('Warning: op or named tensor has the same name as a Weaver',
            'attribute:', name)
    else:
      setattr(self, name, value)

  def _call_op_sugar(self, op_idx, *args):
    """Used to create .op_name syntactic sugar methods."""
    if not all(isinstance(a, six.integer_types) for a in args):
      raise TypeError('All args passed to call_op must be integers '
                      '(LoomResult ids.)  Did you forget to call constant?')
    result = self._weaver.CallOp(op_idx, args)
    if not result:
      raise AssertionError('Weaver op call failed: %s' %
                           self._weaver.error_string())
    if len(result) == 1:
      return result[0]
    return result

  def _deserialize(self, weaver_message_str):
    if not self._weaver.Deserialize(weaver_message_str):
      raise AssertionError(
          'Weaver Deserialization failed: %s' % self._weaver.error_string())

  def serialize(self):
    """Turn this Weaver into a serialized `WeaverMessage` proto.

    Returns:
      A string (the serialization of the Weaver.)

    Raises:
      AssertionError: if the serialization fails.
    """
    serialization = self._weaver.Serialize()
    if not serialization:
      raise AssertionError(
          'Weaver Serialization failed: %s' % self._weaver.error_string())
    return serialization

  @property
  def deepest(self):
    """The maximum depth of any LoomResult created by this input."""
    return self._weaver.Deepest()

  def depth(self, result):
    """Returns the depth of a given itermediate loom result.

    Constants have depth `0`, and (the outputs of) loom ops whose arguments
    have maximum depth `n-1` have depth `n`

    Args:
      result: A loom result whose depth is to be calculated.

    Returns:
      The depth of the result.
    """
    depth = self._weaver.Depth(result)
    if depth == -1:
      raise AssertionError('Invalid LoomResult ID passed to depth.')
    return depth

  def get_type_shape(self, result):
    """Returns the TypeShape of the tensor represented by `result`."""
    ts_idx = self._weaver.GetTypeShape(result)
    if ts_idx == -1:
      raise AssertionError('Invalid LoomResult ID passed to get_type_shape.')
    return self._loom._type_shapes[ts_idx]

  def named_tensor(self, name):
    """Return a LoomResult which stands in for the named Tensor input."""
    return self._tensor_name_to_result[name]

  def constant(self, value, tag=''):
    """Return a LoomResult which stands in a constant value.

    Args:
      value: A NumPy object containing the constant.
      tag: What tag the value's TypeShape ought to have.

    Returns:
      A LoomResult which stands for a constant value.

    Raises:
      TypeError: Raised if the constant does not have one of the TypeShapes
        supported by the loom.
      AssertionError: If an internal error occurs in creating the constant.
    """
    # Constructing a _TypeShape rather than a TypeShape here makes
    # calls to Consant() ~4x faster overall. It is OK to not validate
    # or canonicalize the dtype and shape because are coming from a
    # NumPy array. It is OK to use a _TypeShape instead of a TypeShape
    # for dictionary lookup because they have the same __hash__ and __eq__.
    type_shape = _TypeShape(value.dtype.name, value.shape, tag)
    try:
      ts_idx = self._loom._type_shape_to_idx[type_shape]
    except KeyError:
      raise TypeError('Constant is not of a recognized TypeShape: %s' %
                      str(type_shape))
    if self._loom._direct_feed_dict:
      constant = self._weaver.BatchInput(ts_idx, len(self._constants[ts_idx]))
      self._constants[ts_idx].append(value)
    else:
      # Send raw bytes rather than TensorProto because make_tensor_proto
      # is very slow, taking e.g. 35 usec for a vector of 10 float32s.
      constant = self._weaver.MakeConstantSerialized(ts_idx, value.tobytes())
    if constant == -1:
      raise AssertionError('Weaver Constant creation failed: %s' %
                           self._weaver.error_string())
    return constant

  def batch_input(self, type_shape, batch_idx):
    """Return a LoomResult which stands for en element of a batch_input tensor.

    Args:
      type_shape: Which typeshape the input is from.
      batch_idx: Which element of the batch this input is.

    Returns:
      A LoomResult which stands for an element of a batch_input tensor.

    Raises:
      TypeError: Raised if `type_shape` is not a recognized TypeShape.
      AssertionError: If an internal error occurs in creating the batch input.
    """
    try:
      ts_idx = self._loom._type_shape_to_idx[type_shape]
    except KeyError:
      raise TypeError('Constant is not of a recognized TypeShape: %s' %
                      str(type_shape))
    batch_input = self._weaver.BatchInput(ts_idx, batch_idx)
    if batch_input == -1:
      raise AssertionError('Weaver Batch Input creation failed: %s' %
                           self._weaver.error_string())
    return batch_input

  def op(self, op_name, args):
    """Creates a LoomResult representing the invocation of a LoomOp.

    Args:
      op_name: Which operation to call.
      args: A list of LoomResult objects representing the arguments of the op.

    Returns:
      A list of loom result objects.

    Raises:
      KeyError: Raised if op_name is not the name of a LoomOp for this Loom.
      TypeError: Raised if any of 'args' is not an integer.
      AssertionError: If an internal error occurs calling the op.  Raised if the
        LoomResult arguments are of the wrong TypeShape or if the user attempts
        to create a graph deeper than the Loom's max_depth.
    """
    try:
      op_idx = self._loom._loom_op_name_to_idx[op_name]
    except KeyError:
      raise NameError('Loom Op not found: %s' % str(op_name))

    if not all(isinstance(a, six.integer_types) for a in args):
      raise TypeError('All args passed to call_op must be integers '
                      '(LoomResult ids.)  Did you forget to call constant?')

    op_result = self._weaver.CallOp(op_idx, args)
    if not op_result:
      raise AssertionError('Weaver op call failed: %s' %
                           self._weaver.error_string())
    return op_result

  def __call__(self, val, tag=''):
    """diagram(...) is syntactic sugar for diagram.constant(...)."""
    return self.constant(val, tag=tag)

  def add_output(self, result):
    """Mark 'result' as an output of the loom."""
    if not isinstance(result, six.integer_types):
      raise TypeError('add_output must be called with an integer '
                      '(LoomResult id.)  Did you forget to call constant?')
    if not self._weaver.AddOutput(result):
      raise AssertionError('Weaver AddOutput failed: %s' %
                           self._weaver.error_string())

  def build_feed_dict(self, outputs=None):
    """Turn this diagram into a dictionary for feed_dict.

    Warning: No changes made to this Weaver will be reflected in the
    results of `build_feed_dict` after the first time it is called
    because `build_feed_dict` calls `Weaver::Finalize`, which freezes
    the Weaver's output wirings.

    Returns:
     A dictionary which can be passed as a `feed_dict` argument to
     `tf.Session.run()` t which will cause this Weaver's Loom to behave like
    the diagram.

    Args:
      outputs: Additional nodes which should be sent to the output tensors
        (these can also be set using `add_output`.)
    """

    if self._loom._dry_run:
      return {}

    if outputs is not None:
      for output in outputs:
        self.add_output(output)

    if self._loom._direct_feed_dict:
      self._weaver.Finalize()
      arg_wiring_concat = []
      arg_wiring_slice_starts = []
      arg_wiring_slice_sizes = []
      for depth in xrange(1, self._weaver.MaxDepth()+1):
        for op_idx, op in enumerate(self._loom._loom_ops):
          for arg_idx in xrange(len(op.input_type_shapes)):
            arg_wiring_slice_starts.append(len(arg_wiring_concat))
            wiring = list(self._weaver.GetWiring(depth, op_idx, arg_idx))
            arg_wiring_slice_sizes.append(len(wiring))
            arg_wiring_concat.extend(wiring)
      feed_dict = {
          self._loom._arg_wiring_concat: arg_wiring_concat,
          self._loom._arg_wiring_slice_starts: arg_wiring_slice_starts,
          self._loom._arg_wiring_slice_sizes: arg_wiring_slice_sizes}
      for ts_idx, output_wiring_ph in enumerate(self._loom._output_wirings):
        feed_dict[output_wiring_ph] = self._weaver.GetOutputWiring(ts_idx)
      for ts_idx, constant_ph in enumerate(self._loom._constants):
        constants = np.array(self._constants[ts_idx],
                             dtype=self._loom._type_shapes[ts_idx].dtype)
        if not self._constants[ts_idx]:
          constants = np.reshape(
              constants, (0,) + self._loom._type_shapes[ts_idx].shape)
        feed_dict[constant_ph] = constants
      return feed_dict
    else:
      if self._loom._loom_input_tensor is None:
        raise TypeError('You cannot call build_feed_dict on a LoomInput if '
                        'its Loom has a custom weaver op.')

      return {self._loom._loom_input_tensor: self.serialize()}

# pylint: enable=protected-access
