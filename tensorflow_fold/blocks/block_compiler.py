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
"""Compiler class for a TensorFlow Fold block."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import contextlib
import functools
import multiprocessing
# import google3
import numpy as np
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import loom_ops
from tensorflow_fold.blocks import result_types as tdt
from tensorflow_fold.blocks import util
import tensorflow_fold.blocks.blocks
from tensorflow_fold.public import loom


class _TaggedLoom(loom.Loom):
  """A wrapper around a Loom with a tagging op."""

  def __init__(self, tagging_op, tagging_op_name, *args, **kwargs):
    self.tagging_op = tagging_op
    self.tagging_op_name = tagging_op_name
    super(_TaggedLoom, self).__init__(*args, **kwargs)


class _CompilerContext(object):
  """An encapsulation of the state associated with compilation of a block.

  A compilation context is created by a Compiler in order to
  incrementally build up the state necessary for creating a Loom
  object; dictionaries of named ops and tensors, and a set of all type
  shapes that the loom is required to support.
  """

  def __init__(self):
    self._tensor_to_name = {}
    self._op_to_name = {}
    self._current_gensym = 0
    self._extra_type_shapes = set()
    self._metric_ops = {}

  def register_op(self, op, name):
    """Generates a unique name for an op.

    Args:
      op: A LoomOp.
      name: A string; this will become part of the unique name.

    Returns:
      A string name for the op.
    """
    return self._op_to_name.setdefault(op, self._gensym(name))

  def register_tensor(self, tensor, name):
    """Like register_op but for loom named tensors."""
    return self._tensor_to_name.setdefault(tensor, self._gensym(name))

  def register_tensor_type(self, tensor_type):
    """Ensures that loom can support tensor_type, even if it is not in an op.

    Loom internally indexes tensors by TypeShape. By default the set
    of TypeShapes that get indexed over is the union of all input and
    output types from loom ops. This is generally sufficient, but
    fails in the edge case of a constant having a unique TypeShape
    that is used as an output, because there will be no op having such
    a TypeShape. We handle this by explicitly registering all tensor
    types as part of the block compilation process, and passing them
    to the Loom constructor.

    Args:
      tensor_type: An instance of `td.TensorType`.
    """
    type_shape = loom.TypeShape(tensor_type.dtype, tensor_type.shape)
    self._extra_type_shapes.add(type_shape)

  def _gensym(self, name):
    self._current_gensym += 1
    return '%s_%d' % (name, self._current_gensym)

  @property
  def metric_ops(self):
    """Returns a dictionary mapping metric names to loom ops."""
    return self._metric_ops

  def register_metric_op(self, name, typ):
    """Creates a loom op for metric `name` with `typ` if not already present."""
    if name not in self.metric_ops:
      op = loom_ops.TaggingPassThroughOp([typ], [name])
      self._metric_ops[name] = op
      self._op_to_name[op] = name
    elif self.metric_ops[name].passthrough_types[0] != typ:
      raise TypeError('Metric %s has incompatible types: %s vs %s' %
                      (name, self.metric_ops[name].passthrough_types[0], typ))

  def create_tagged_loom(self, passthrough_types, max_depth=None,
                         loom_input_tensor=None, dry_run=False,
                         parallel_iterations=None, back_prop=None,
                         swap_memory=None):
    """Creates a loom with a tagged passthrough op."""
    named_tensors = {n: t for t, n in six.iteritems(self._tensor_to_name)}
    named_ops = {name: op for (op, name) in six.iteritems(self._op_to_name)}

    if passthrough_types:
      tags = ['_TensorFlowFoldOutputTag_%d' % i
              for i in xrange(len(passthrough_types))]
      tagging_op = loom_ops.TaggingPassThroughOp(passthrough_types, tags)
      tagging_op_name = self._gensym('TaggingPassThrough')
      named_ops[tagging_op_name] = tagging_op
    else:  # we can't create an op with no types but we can create the loom
      tagging_op = None
      tagging_op_name = None

    # We need to add one to max_depth here because the tagging op
    # increases the depth of the tree by one.
    if max_depth is not None: max_depth += 1
    return _TaggedLoom(tagging_op, tagging_op_name, max_depth=max_depth,
                       named_tensors=named_tensors, named_ops=named_ops,
                       extra_type_shapes=self._extra_type_shapes,
                       parallel_iterations=parallel_iterations,
                       back_prop=back_prop, swap_memory=swap_memory,
                       dry_run=dry_run, loom_input_tensor=loom_input_tensor)


class _EvalContext(object):
  """An encapsulation of the state associated with evaluation of a block.

  Attributes:
    op: A function taking an op name and list of loom result arguments
      that returns a list of loom results representing the invocation of
      the op on the arguments.
    constant: A function that takes a NumPy array as input and returns a
      corresponding loom result.
    named_tensor: A function taking a string tensor name as input that
      returns the corresponding loom result.
    add_output: A function taking a loom result as input that marks it
     as an output of the weaver.
    metric_labels: A dict mapping metric names (strings) to lists of
      associated labels (arbitrary python objects) used to identify the
      corresponding metric outputs; see [`td.Metric`](#td.Metric) for details.
  """

  def __init__(self, weaver, metric_names):
    self._constant_cache = {}
    self.set_weaver(weaver)
    self.metric_labels = {name: [] for name in metric_names}

  def set_weaver(self, weaver):
    self.op = weaver.op
    self.constant = weaver.constant
    self.named_tensor = weaver.named_tensor
    self.add_output = weaver.add_output
    self._constant_cache.clear()

  def memoize_constant(self, block, key, value):
    """Creates a constant with memoiziation.

    Args:
      block: The block creating the constant.
      key: Hashable key; `(block, key)` must be globally unique.
      value: The NumPy array to create a constant for.

    Returns:
      A LoomResult.
    """
    try:
      return self._constant_cache[block, key]
    except KeyError:
      self._constant_cache[block, key] = result = self.constant(value)
      return result


class Compiler(object):
  """A compiler for TensorFlow Fold blocks."""

  def __init__(self):
    """Creates a Compiler.

    Most users will want to use the `Compiler.create` factory, like so:

    ```python
    compiler = td.Compiler.create(root_block_like)
    ```

    Which is simply a short-hand for:

    ```python
    compiler = td.Compiler()
    compiler.compile(root_block_like)
    compiler.init_loom()
    ```
    """
    self._root = None
    self._input_tensor = None
    self._dry = None
    self._wet = None
    self._pools = []

  @classmethod
  def create(cls, root_block_like, max_depth=None, loom_input_tensor=None,
             input_tensor=None, parallel_iterations=None, back_prop=None,
             swap_memory=None):
    """Creates a Compiler, compiles a block, and initializes loom.

    Args:
      root_block_like: A block or an object that can be converted to a block by
        [`td.convert_to_block`](#td.convert_to_block). Must have at least one
        output or metric tensor. The output type may not contain any
        Sequence or PyObject types.
      max_depth: Optional. The maximum nesting depth that the encapsulated loom
        ought to support. This is dependent on the topology of the block graph
        and on the shape of the data being passed in. May be calculated by
        calling [`Compiler.max_depth`](#td.Compiler.max_depth). If unspecified,
        a `tf.while_loop` will be used to dynamically calculate `max_depth` on
        a per-batch basis.
      loom_input_tensor: An optional string tensor of loom inputs for the
        compiler to read from. Mutually exclusive with `input_tensor'.
      input_tensor: an optional string tensor for the block to read inputs from.
        If an input_tensor is supplied the user can just evaluate the compiler's
        output tensors without needing to create a feed dict via
        'build_feed_dict'. Mutually exclusive with `loom_input_tensor'.
      parallel_iterations: tf.while_loop's parallel_iterations option, which
        caps the number of different depths at which ops could run in parallel.
        Only applies when max_depth=None. Default: 10.
      back_prop: tf.while_loop's back_prop option, which enables gradients. Only
        applies when max_depth=None.  Default: True.
      swap_memory: Whether to use tf.while_loop's swap_memory option, which
        enables swapping memory between GPU and CPU at the possible expense of
        some performance. Only applies when max_depth=None. Default: False.

    Returns:
      A fully initialized Compiler.

    Raises:
      TypeError: If `root_block_like` cannot be converted to a block.
      TypeError: If `root_block_like` fails to compile.
      TypeError: If `root_block_like` has no output or metric tensors.
      TypeError: If `root_block_like` has an invalid output type.
      ValueError: If both `loom_input_tensor` and `input_tensor` are provided.
    """
    self = cls().compile(root_block_like)
    self.init_loom(max_depth, loom_input_tensor, input_tensor,
                   parallel_iterations, back_prop, swap_memory)
    return self

  @classmethod
  def _interactive(cls, root_block):
    # pylint: disable=protected-access
    return cls()._setup(root_block, interactive_mode=True)
    # pylint: enable=protected-access

  @property
  def root(self):
    """Returns the root block, or None if `compile()` has not been called."""
    return self._root

  def compile(self, root_block_like):
    """Compiles a block, and sets it to the root.

    Args:
      root_block_like: A block or an object that can be converted to a block by
        [`td.convert_to_block`](#td.convert_to_block). Must have at least one
        output or metric tensor. The output type may not contain any
        Sequence or PyObject types.

    Returns:
      `self`

    Raises:
      RuntimeError: If `init_loom()` has already been called.
      TypeError: If `root_block_like` cannot be converted to a block.
      TypeError: If `root_block_like` fails to compile.
      TypeError: If `root_block_like` has no output or metric tensors.
      TypeError: If `root_block_like` has an invalid output type.
    """
    if self.is_loom_initialized:
      raise RuntimeError('Loom has already been initialized.')
    return self._setup(root_block_like, interactive_mode=False)

  def _setup(self, root_block_like, interactive_mode):
    """Sets up the compiler."""
    self._root = (
        tensorflow_fold.blocks.blocks.convert_to_block(
            root_block_like))
    self._ctx = _CompilerContext()
    self._compile_blocks()

    out_type = self.root.output_type
    terminal_ts = list(out_type.terminal_types())
    tensor_ts = [t for t in terminal_ts if isinstance(t, tdt.TensorType)]

    if not interactive_mode:
      if any(isinstance(t, tdt.PyObjectType) for t in terminal_ts):
        raise TypeError('root outputs must all be tensors: %s' % out_type)
      if not (tensor_ts or self._ctx.metric_ops):
        raise TypeError('root must have at least one output or metric tensor: '
                        '%s' % out_type)
      subtypes = [out_type]
      while subtypes:
        subtype = subtypes.pop()
        if isinstance(subtype, tdt.SequenceType):
          raise TypeError('root output may not contain sequences %s' % out_type)
        if isinstance(subtype, tdt.TupleType):
          subtypes.extend(subtype)

      flatten = out_type.flatten
      if not tensor_ts:  # no tensor outputs
        self._extract_tensor_outputs = None
      else:  # all outputs will be tensors
        self._extract_tensor_outputs = flatten
    self._dry = self._ctx.create_tagged_loom(tensor_ts, dry_run=True)
    return self

  def _compile_blocks(self):
    """Compiles all blocks reachable via root."""
    # Input type must be PyObject or VoidType.
    if not isinstance(self.root.input_type, tdt.VoidType):
      self.root.set_input_type(tdt.PyObjectType())

    visited = set()
    stack = [self.root]

    def traverse_blocks(block):
      """Returns blocks and its descendants in post order."""
      # Check for cycles as an internal consistency check; it should
      # not be possible to create a cycle via the public API.
      if block in visited:
        raise ValueError('internal error; block %s forms a cycle; please file '
                         'a bug' % (block,))
      visited.add(block)
      post_order_blocks = []
      # These blocks appear in post-order (not counting links created by
      # forward declarations).
      for child in block.children:
        post_order_blocks += traverse_blocks(child)
      post_order_blocks.append(block)
      if block.is_forward_declaration_ref: stack.append(block.target_block)
      return post_order_blocks

    while stack:
      block = stack.pop()
      if block not in visited:
        block._validate(self._ctx)  # pylint: disable=protected-access
        for b in traverse_blocks(block):
          b._compile(self._ctx)  # pylint: disable=protected-access
          # Ensure that loom recognizes all necessary type shapes.
          for t in b.output_type.terminal_types():
            if isinstance(t, tdt.TensorType): self._ctx.register_tensor_type(t)

  def _eval_ctx(self, weaver):
    return _EvalContext(weaver, six.iterkeys(self._ctx.metric_ops))

  def init_loom(self, max_depth=None, loom_input_tensor=None, input_tensor=None,
                parallel_iterations=None, back_prop=None, swap_memory=None):
    """Intializes the loom object, which is used to run on tensorflow.

    Args:
      max_depth: Optional. The maximum nesting depth that the encapsulated loom
        ought to support. This is dependent on the topology of the block graph
        and on the shape of the data being passed in. May be calculated by
        calling [`Compiler.max_depth`](#td.Compiler.max_depth). If unspecified,
        a `tf.while_loop` will be used to dynamically calculate `max_depth` on
        a per-batch basis.
      loom_input_tensor: An optional string tensor of loom inputs for the
        compiler to read from. Mutually exclusive with `input_tensor'.
      input_tensor: an optional string tensor for the block to read inputs from.
        If an input_tensor is supplied the user can just evaluate the compiler's
        output tensors without needing to create a feed dict via
        'build_feed_dict'. Mutually exclusive with `loom_input_tensor'.
      parallel_iterations: tf.while_loop's parallel_iterations option, which
        caps the number of different depths at which ops could run in parallel.
        Only applies when max_depth=None. Default: 10.
      back_prop: tf.while_loop's back_prop option, which enables gradients. Only
        applies when max_depth=None.  Default: True.
      swap_memory: Whether to use tf.while_loop's swap_memory option, which
        enables swapping memory between GPU and CPU at the possible expense of
        some performance. Only applies when max_depth=None. Default: False.

    Raises:
      RuntimeError: If `compile()` has not been called.
      RuntimeError: If the loom has already been initialized.
      ValueError: If both `loom_input_tensor` and `input_tensor` are provided.
    """
    if self.root is None: raise RuntimeError('compile() has not been called')
    if self.is_loom_initialized:
      raise RuntimeError('Loom has already been initialized.')
    # TODO(delesley): remove unused ops and tensors
    if input_tensor is not None:
      if loom_input_tensor is not None:
        raise ValueError('cannot specify both input_tensor and '
                         'loom_input_tensor for the same compiler')
      self._input_tensor = input_tensor
      loom_input_tensor = tf.py_func(
          self.build_loom_input_batched,
          [self._input_tensor], [tf.string], name='Scheduler')
    passthrough_types = (self._dry.tagging_op.passthrough_types
                         if self._dry.tagging_op else None)
    self._init_loom(passthrough_types, max_depth, loom_input_tensor,
                    parallel_iterations, back_prop, swap_memory)

  def _init_loom(self, passthrough_types, max_depth, loom_input_tensor=None,
                 parallel_iterations=None, back_prop=None, swap_memory=None):
    """Sets up the wet loom and its output tensors and metric tensors."""
    self._wet = self._ctx.create_tagged_loom(
        passthrough_types, max_depth, loom_input_tensor,
        parallel_iterations=parallel_iterations,
        back_prop=back_prop, swap_memory=swap_memory)
    if passthrough_types:
      self._output_tensors = [self._wet.output_tensor(ts)
                              for ts in self._wet.tagging_op.output_type_shapes]
    else:
      self._output_tensors = []
    self._metric_tensors = collections.OrderedDict(
        (name, self._wet.output_tensor(op.output_type_shapes[0]))
        for name, op in sorted(six.iteritems(self._ctx.metric_ops)))

  @property
  def is_loom_initialized(self):
    return bool(self._wet)

  def _check_loom_initialized(self):
    if not self.is_loom_initialized:
      raise RuntimeError('Loom has not been initialized.')

  def build_feed_dict(self, examples, batch_size=None, metric_labels=False,
                      ordered=False):
    """Turns a batch of examples into a dictionary for feed_dict.

    If an input_tensor was supplied when the Compiler was constructed, the user
    can just evaluate the compiler's output tensors without needing to create a
    feed_dict via 'build_feed_dict'.

    This is a convenience method equivalent to
    `{compiler.loom_input_tensor:
     compiler.build_loom_input_batched(examples, batch_size, ordered)}`
    when `metric_labels=False`.

    The result is computed lazily (e.g. when passed as a feed_dict to
    `Session.run()`), and thus does not block when using
    multiprocessing. The exception is when metric_labels=True, in
    which case we need to block in order to aggregate the labels
    across chunks of work.

    Args:
      examples: A non-empty iterable of examples to be built into tensors.
      batch_size: The maximum number of examples to compile into each loom
        input. Defaults to 100. If multiprocessing then this will also be the
        chunk size for each unit of work.
      metric_labels: Whether or not to return metric labels.
      ordered: Whether or not to preserve ordering when multiprocessing,
        otherwise has not effect (and order is always preserved).

    Returns:
      A feed dictionary which can be passed to TensorFlow `run()`/`eval()`. If
      `metric_labels` is True, a `(feed_dict, metric_labels)` tuple.

    Raises:
      TypeError: If `examples` is not an iterable.
      RuntimeError: If [`init_loom()`](#td.Compiler.init_loom) has not been
        called.

    """
    self._check_build('build_feed_dict', examples)
    results = self.build_loom_input_batched(
        examples, batch_size, metric_labels, ordered)
    if not metric_labels: return {self.loom_input_tensor: results}
    # We need to block here to compute metric labels.
    first_batch, metric_labels = next(results)
    batches = [first_batch]
    for batch, batch_metric_labels in results:
      batches.append(batch)
      for k, v in six.iteritems(batch_metric_labels):
        metric_labels[k].extend(v)
    return {self.loom_input_tensor: batches}, metric_labels

  def build_loom_input_batched(self, examples, batch_size=None,
                               metric_labels=False, ordered=False):
    """Turns examples into a feed value for `self.loom_input_tensor`.

    The result is an iterator; work doesn't happen until you call
    e.g. `next()` or `list()` on it.

    Args:
      examples: A non-empty iterable of examples to be built into tensors.
      batch_size: The maximum number of examples to compile into each loom
        input. Defaults to 100. If multiprocessing then this will also be
        the chunk size for each unit of work.
      metric_labels: Whether or not to return metric labels.
      ordered: Whether or not to preserve ordering when multiprocessing,
        otherwise has not effect (and order is always preserved).

    Returns:
      Feed value(s) corresponding to `examples` grouped into batches. The result
      itself can be fed directly to `self.loom_input_tensor`, or be iterated
      over to feed values batch-by-batch. If `metric_labels` is True, an
      iterable of `(batch_feed_value, metric_labels)` tuples.

    Raises:
      TypeError: If `examples` is not an iterable.
      RuntimeError: If [`init_loom()`](#td.Compiler.init_loom) has not been
        called.

    """
    self._check_build('build_loom_input_batched', examples)
    if batch_size is None: batch_size = 100
    batches = util.group_by_batches(examples, batch_size)
    results = _map_maybe_parallel(
        self.pool, _subprocess_build_batch, self._build_batch, batches, ordered,
        chunk_size=1, metric_labels=metric_labels)
    # If metric_labels is false, use an edible iterator so that
    # `results` can be fed.
    if not metric_labels: results = util.EdibleIterator(results)
    return results

  def _build_batch(self, batch, metric_labels):
    weaver = self._wet.make_weaver()
    eval_ctx = self._eval_ctx(weaver)
    for example in batch:
      self._add_outputs(eval_ctx, example)
    return ((weaver.serialize(), eval_ctx.metric_labels)
            if metric_labels else weaver.serialize())

  def build_loom_inputs(self, examples, metric_labels=False,
                        chunk_size=100, ordered=False):
    """Turns examples into feed values for `self.loom_input_tensor`.

    The result is an iterator; work doesn't happen until you call
    e.g. `next()` or `list()` on it.

    Args:
      examples: An iterable of example to be built into tensors.
      metric_labels: Whether or not to return metric labels.
      chunk_size: If multiprocessing then the size of each unit of work.
        Defaults to 100. If not multiprocessing then this has no effect.
      ordered: Whether or not to preserve ordering when multiprocessing. If
        not multiprocessing then this has no effect (order is always preserved).

    Returns:
      An iterable of strings (morally bytes) that can be fed to
      `self.loom_input_tensor`. If `metric_labels` is True, an iterable of
      `(string, metric_labels)` tuples.

    Raises:
      TypeError: If `examples` is not an iterable.
      RuntimeError: If [`init_loom()`](#td.Compiler.init_loom) has not been
        called.
    """
    self._check_build('build_loom_inputs', examples)
    return _map_maybe_parallel(
        self.pool, _subprocess_build_single, self._build_single, examples,
        ordered, chunk_size, metric_labels=bool(metric_labels))

  def _build_single(self, example, metric_labels):
    weaver = self._wet.make_weaver()
    eval_ctx = self._eval_ctx(weaver)
    self._add_outputs(eval_ctx, example)
    return ((weaver.serialize(), eval_ctx.metric_labels)
            if metric_labels else weaver.serialize())

  def _add_outputs(self, eval_ctx, x):
    # pylint: disable=protected-access
    result = self._root._evaluate(eval_ctx, x)
    if self._extract_tensor_outputs:
      for y in eval_ctx.op(
          self._wet.tagging_op_name, self._extract_tensor_outputs(result)):
        eval_ctx.add_output(y)

  def _check_build(self, fn_name, examples):
    self._check_loom_initialized()
    if not isinstance(examples, collections.Iterable):
      raise TypeError('td.Compiler.%s takes a batch of examples, '
                      'not a single example.' % fn_name)

  @contextlib.contextmanager
  def multiprocessing_pool(self, processes=None):
    """Creates a context for use with the Python `with` statement.

    Entering this context creates a pool of subprocesses for building
    loom inputs in parallel with this compiler. When the context exits
    the pool is closed, blocking until all work is completed.

    Args:
      processes: The number of worker processes to use. Defaults to the
      cpu count (`multiprocessing.cpu_count()`).

    Yields:
      Nothing.

    Raises:
      RuntimeError: If [`init_loom()`](#td.Compiler.init_loom) has not been
        called.
    """
    self._check_loom_initialized()
    pool = multiprocessing.Pool(processes, _subprocess_init, (self,))
    self._pools.append(pool)
    try:
      yield
    finally:
      pool.close()
      pool.join()
      if self.pool != pool:
        raise AssertionError('multiprocessing_pool nesting violation')
      self._pools.pop()

  @property
  def pool(self):
    """Returns the current multiprocessing pool if it exists, else None."""
    return self._pools[-1] if self._pools else None

  @property
  def input_tensor(self):
    """Returns input tensor that can feed data to this compiler."""
    if not self._input_tensor:
      raise RuntimeError('Compiler has no input tensor.')
    return self._input_tensor

  @property
  def output_tensors(self):
    """Returns a flattened list of all output tensors."""
    self._check_loom_initialized()
    return self._output_tensors

  @property
  def loom_input_tensor(self):
    """Returns the loom input tensor, used for building feed dictionaries.

    May be fed a single result or a sequence of results from
    `Compiler.build_loom_inputs()` or `Compiler.build_loom_input_batched()`.

    Returns:
      A string tensor.

    Raises:
      RuntimeError: If `Compiler.init_loom()` has not been called.
    """
    self._check_loom_initialized()
    return self._wet.input_tensor

  @property
  def metric_tensors(self):
    """Returns a ordered dictionary of tensors for output metrics."""
    self._check_loom_initialized()
    return self._metric_tensors

  def max_depth(self, inp):
    """Returns the loom `max_depth` needed to evaluate `inp`."""
    if not self._dry: return 0  # handles PyObject, Void, and () output
    return self._dry_run(inp)[1]

  def _dry_run(self, inp):
    """Does a dry run on `inp`, returns (result, max_depth)."""
    # pylint: disable=protected-access
    weaver = self._dry.make_weaver()
    result = self.root._evaluate(self._eval_ctx(weaver), inp)
    return result, weaver.deepest

  def _eval(self, inp, feed_dict, session, tolist, use_while_loop):
    """Implements block.eval()."""
    out_type = self.root.output_type

    # Do a dry run to calculate max depth and the structure of the result.
    # We need to create a pass-through op that takes all tensor types in
    # out_type as inputs, to ensure that loom recognizes their type-shapes.
    dry_result, max_depth = self._dry_run(inp)
    if use_while_loop: max_depth = None

    # We map the loom results 'dry_result' to their corresponding tensor types.
    terminal_ts = []
    out_type.for_each_terminal(lambda t, _: terminal_ts.append(t), dry_result)
    tensor_ts = [t for t in terminal_ts if isinstance(t, tdt.TensorType)]

    # If there are no tensor types or metrics then we can't do a wet
    # run. Fortunately we don't need to.
    if not (tensor_ts or self._ctx.metric_ops): return dry_result

    # Now we can do a wet run to get the actual loom results.
    # Create a custom passthrough op that handles the exact shape of dry_result.
    self._init_loom(tensor_ts, max_depth)
    weaver = self._wet.make_weaver()
    wet_result = self.root._evaluate(  # pylint: disable=protected-access
        self._eval_ctx(weaver), inp)

    # We need to flatten 'wet_result' so we can use it with the passthrough op.
    tensor_results = [
        r for r, t in zip(out_type.flatten(wet_result), terminal_ts)
        if isinstance(t, tdt.TensorType)]
    if tensor_results:  # only create the op if we have tensor outputs
      tensor_results = weaver.op(self._wet.tagging_op_name, tensor_results)
    fd = weaver.build_feed_dict(tensor_results)
    if feed_dict is not None: fd.update(feed_dict)

    # Now we can evaluate the loom results, assuming we have a session.
    if session is None:
      session = tf.get_default_session()
      if session is None:
        raise ValueError('No default session is registered. Use `with '
                         'sess.as_default()` or pass a session to eval()')
    _init_uninitialized(session)
    tensor_outputs_and_metrics = session.run(
        self.output_tensors + list(self.metric_tensors.values()), feed_dict=fd)

    # Handle the tensor outputs.
    tensor_outputs = tensor_outputs_and_metrics[:len(self.output_tensors)]

    # Drop the first (batch) dimension.
    tensor_outputs = (np.squeeze(array, 0) for array in tensor_outputs)

    # If requested, convert arrays to lists.
    if tolist: tensor_outputs = (array.tolist() for array in tensor_outputs)

    # Merge with non-tensor outputs.
    outputs = (next(tensor_outputs) if isinstance(t, tdt.TensorType) else dry
               for t, dry in zip(terminal_ts, out_type.flatten(dry_result)))

    # Unflatten to create a result with the correct shape.
    result = out_type.unflatten(outputs, dry_result)

    # Handle metrics.
    metrics = tensor_outputs_and_metrics[len(self.output_tensors):]
    metrics = {name: value.tolist() if tolist else value
               for name, value in zip(self.metric_tensors, metrics)}

    return (result, metrics) if metrics else result


def _init_uninitialized(sess):
  """Initializes all uninitialized variables and returns them as a list."""
  variables = tf.global_variables()
  if not variables: return []  # sess.run() barfs on empty list
  is_initialized = sess.run([tf.is_variable_initialized(v) for v in variables])
  needs_init = [v for v, i in zip(variables, is_initialized) if not i]
  if not needs_init: return []
  sess.run(tf.variables_initializer(needs_init))
  return needs_init


_subprocess_compiler = None


def _subprocess_init(compiler):
  global _subprocess_compiler
  if _subprocess_compiler is not None:
    raise ValueError('internal error; subprocess initialization was called '
                     'twice; please file a bug')
  _subprocess_compiler = compiler


def _subprocess_build_single(example, metric_labels):
  # pylint: disable=protected-access
  return _subprocess_compiler._build_single(example, metric_labels)


def _subprocess_build_batch(examples, metric_labels):
  # pylint: disable=protected-access
  return _subprocess_compiler._build_batch(examples, metric_labels)


def _map_maybe_parallel(pool, parallel_fn, serial_fn, items,
                        ordered, chunk_size, *fn_args, **fn_kwargs):
  """Lazy map over `items`; parallel if `pool` is true, otherwise serial."""
  if pool:
    mapper = pool.imap if ordered else pool.imap_unordered
    mapper = functools.partial(mapper, chunksize=chunk_size)
    fn = parallel_fn
  else:
    mapper = map
    fn = serial_fn
  if fn_args or fn_kwargs: fn = functools.partial(fn, *fn_args, **fn_kwargs)
  return mapper(fn, items)
