<!-- This file is machine generated: DO NOT EDIT! -->

# TensorFlow Fold Python Blocks API

Note: Functions taking `Block` arguments can also take anything accepted by
[`td.convert_to_block`](#td.convert_to_block). Functions taking `ResultType`
arguments can can also take anything accepted by
[`td.convert_to_type`](#td.convert_to_type).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [`module tensorflow_fold.public.blocks` (`td`)](#module-tensorflow_foldpublicblocks-td)
- [Compiler](#compiler)
  - [`class td.Compiler`](#class-tdcompiler)
- [Blocks for input](#blocks-for-input)
  - [`class td.Tensor`](#class-tdtensor)
  - [`td.Scalar(dtype='float32', name=None)`](#tdscalardtypefloat32-namenone)
  - [`td.Vector(size, dtype='float32', name=None)`](#tdvectorsize-dtypefloat32-namenone)
  - [`class td.InputTransform`](#class-tdinputtransform)
  - [`td.SerializedMessageToTree(message_type_name)`](#tdserializedmessagetotreemessage_type_name)
  - [`class td.OneHot`](#class-tdonehot)
  - [`td.OneHotFromList(elements, dtype='float32', strict=True, name=None)`](#tdonehotfromlistelements-dtypefloat32-stricttrue-namenone)
  - [`class td.Optional`](#class-tdoptional)
- [Blocks for composition](#blocks-for-composition)
  - [`class td.Composition`](#class-tdcomposition)
  - [`td.Pipe(*blocks, **kwargs)`](#tdpipeblocks-kwargs)
  - [`class td.Record`](#class-tdrecord)
  - [`td.AllOf(*blocks, **kwargs)`](#tdallofblocks-kwargs)
- [Blocks for tensors](#blocks-for-tensors)
  - [`class td.FromTensor`](#class-tdfromtensor)
  - [`class td.Function`](#class-tdfunction)
  - [`class td.Concat`](#class-tdconcat)
  - [`td.Zeros(output_type, name=None)`](#tdzerosoutput_type-namenone)
- [Blocks for sequences](#blocks-for-sequences)
  - [`class td.Map`](#class-tdmap)
  - [`class td.Fold`](#class-tdfold)
  - [`td.RNN(cell, initial_state=None, initial_state_from_input=False, name=None)`](#tdrnncell-initial_statenone-initial_state_from_inputfalse-namenone)
  - [`class td.Reduce`](#class-tdreduce)
  - [`td.Sum(name=None)`](#tdsumnamenone)
  - [`td.Min(name=None)`](#tdminnamenone)
  - [`td.Max(name=None)`](#tdmaxnamenone)
  - [`td.Mean(name=None)`](#tdmeannamenone)
  - [`class td.Broadcast`](#class-tdbroadcast)
  - [`class td.Zip`](#class-tdzip)
  - [`td.ZipWith(elem_block, name=None)`](#tdzipwithelem_block-namenone)
  - [`class td.NGrams`](#class-tdngrams)
  - [`class td.Nth`](#class-tdnth)
  - [`class td.GetItem`](#class-tdgetitem)
  - [`class td.Length`](#class-tdlength)
  - [`td.Slice(*args, **kwargs)`](#tdsliceargs-kwargs)
- [Other blocks](#other-blocks)
  - [`class td.ForwardDeclaration`](#class-tdforwarddeclaration)
  - [`class td.OneOf`](#class-tdoneof)
  - [`class td.Metric`](#class-tdmetric)
  - [`class td.Identity`](#class-tdidentity)
  - [`td.Void(name=None)`](#tdvoidnamenone)
- [Layers](#layers)
  - [`class td.FC`](#class-tdfc)
  - [`class td.Embedding`](#class-tdembedding)
  - [`class td.FractalNet`](#class-tdfractalnet)
  - [`class td.ScopedLayer`](#class-tdscopedlayer)
- [Types](#types)
  - [`class td.TensorType`](#class-tdtensortype)
  - [`class td.VoidType`](#class-tdvoidtype)
  - [`class td.PyObjectType`](#class-tdpyobjecttype)
  - [`class td.TupleType`](#class-tdtupletype)
  - [`class td.SequenceType`](#class-tdsequencetype)
  - [`class td.BroadcastSequenceType`](#class-tdbroadcastsequencetype)
- [Plans](#plans)
  - [`class td.Plan`](#class-tdplan)
  - [`class td.TrainPlan`](#class-tdtrainplan)
  - [`class td.EvalPlan`](#class-tdevalplan)
  - [`class td.InferPlan`](#class-tdinferplan)
  - [`td.define_plan_flags(default_plan_name='plan', blacklist=None)`](#tddefine_plan_flagsdefault_plan_nameplan-blacklistnone)
  - [`td.plan_default_params()`](#tdplan_default_params)
- [Conversion functions](#conversion-functions)
  - [`td.convert_to_block(block_like)`](#tdconvert_to_blockblock_like)
  - [`td.convert_to_type(type_like)`](#tdconvert_to_typetype_like)
  - [`td.canonicalize_type(type_like)`](#tdcanonicalize_typetype_like)
- [Utilities](#utilities)
  - [`class td.EdibleIterator`](#class-tdedibleiterator)
  - [`td.group_by_batches(iterable, batch_size, truncate=False)`](#tdgroup_by_batchesiterable-batch_size-truncatefalse)
  - [`td.epochs(items, n=None, shuffle=True, prng=None)`](#tdepochsitems-nnone-shuffletrue-prngnone)
  - [`td.parse_spec(spec)`](#tdparse_specspec)
  - [`td.build_optimizer_from_params(optimizer='adam', global_step=None, learning_rate_decay_params=None, **kwargs)`](#tdbuild_optimizer_from_paramsoptimizeradam-global_stepnone-learning_rate_decay_paramsnone-kwargs)
  - [`td.create_variable_scope(name)`](#tdcreate_variable_scopename)
- [Abstract classes](#abstract-classes)
  - [`class td.IOBase`](#class-tdiobase)
  - [`class td.Block`](#class-tdblock)
  - [`class td.Layer`](#class-tdlayer)
  - [`class td.ResultType`](#class-tdresulttype)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

- - -
## `module tensorflow_fold.public.blocks` (`td`)
High-level Blocks API for [TensorFlow Fold](../index.md).

## Compiler

- - -

<a name="td.Compiler"></a>
### `class td.Compiler`

A compiler for TensorFlow Fold blocks.
- - -

<a name="td.Compiler.__init__"></a>
#### `td.Compiler.__init__()`

Creates a Compiler.

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


- - -

<a name="td.Compiler.build_feed_dict"></a>
#### `td.Compiler.build_feed_dict(examples, batch_size=None, metric_labels=False, ordered=False)`

Turns a batch of examples into a dictionary for feed_dict.

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

##### Args:


*  <b>`examples`</b>: A non-empty iterable of examples to be built into tensors.
*  <b>`batch_size`</b>: The maximum number of examples to compile into each loom
    input. Defaults to 100. If multiprocessing then this will also be the
    chunk size for each unit of work.
*  <b>`metric_labels`</b>: Whether or not to return metric labels.
*  <b>`ordered`</b>: Whether or not to preserve ordering when multiprocessing,
    otherwise has not effect (and order is always preserved).

##### Returns:

  A feed dictionary which can be passed to TensorFlow `run()`/`eval()`. If
  `metric_labels` is True, a `(feed_dict, metric_labels)` tuple.

##### Raises:


*  <b>`TypeError`</b>: If `examples` is not an iterable.
*  <b>`RuntimeError`</b>: If [`init_loom()`](#td.Compiler.init_loom) has not been
    called.


- - -

<a name="td.Compiler.build_loom_input_batched"></a>
#### `td.Compiler.build_loom_input_batched(examples, batch_size=None, metric_labels=False, ordered=False)`

Turns examples into a feed value for `self.loom_input_tensor`.

The result is an iterator; work doesn't happen until you call
e.g. `next()` or `list()` on it.

##### Args:


*  <b>`examples`</b>: A non-empty iterable of examples to be built into tensors.
*  <b>`batch_size`</b>: The maximum number of examples to compile into each loom
    input. Defaults to 100. If multiprocessing then this will also be
    the chunk size for each unit of work.
*  <b>`metric_labels`</b>: Whether or not to return metric labels.
*  <b>`ordered`</b>: Whether or not to preserve ordering when multiprocessing,
    otherwise has not effect (and order is always preserved).

##### Returns:

  Feed value(s) corresponding to `examples` grouped into batches. The result
  itself can be fed directly to `self.loom_input_tensor`, or be iterated
  over to feed values batch-by-batch. If `metric_labels` is True, an
  iterable of `(batch_feed_value, metric_labels)` tuples.

##### Raises:


*  <b>`TypeError`</b>: If `examples` is not an iterable.
*  <b>`RuntimeError`</b>: If [`init_loom()`](#td.Compiler.init_loom) has not been
    called.


- - -

<a name="td.Compiler.build_loom_inputs"></a>
#### `td.Compiler.build_loom_inputs(examples, metric_labels=False, chunk_size=100, ordered=False)`

Turns examples into feed values for `self.loom_input_tensor`.

The result is an iterator; work doesn't happen until you call
e.g. `next()` or `list()` on it.

##### Args:


*  <b>`examples`</b>: An iterable of example to be built into tensors.
*  <b>`metric_labels`</b>: Whether or not to return metric labels.
*  <b>`chunk_size`</b>: If multiprocessing then the size of each unit of work.
    Defaults to 100. If not multiprocessing then this has no effect.
*  <b>`ordered`</b>: Whether or not to preserve ordering when multiprocessing. If
    not multiprocessing then this has no effect (order is always preserved).

##### Returns:

  An iterable of strings (morally bytes) that can be fed to
  `self.loom_input_tensor`. If `metric_labels` is True, an iterable of
  `(string, metric_labels)` tuples.

##### Raises:


*  <b>`TypeError`</b>: If `examples` is not an iterable.
*  <b>`RuntimeError`</b>: If [`init_loom()`](#td.Compiler.init_loom) has not been
    called.


- - -

<a name="td.Compiler.compile"></a>
#### `td.Compiler.compile(root_block_like)`

Compiles a block, and sets it to the root.

##### Args:


*  <b>`root_block_like`</b>: A block or an object that can be converted to a block by
    [`td.convert_to_block`](#td.convert_to_block). Must have at least one
    output or metric tensor. The output type may not contain any
    Sequence or PyObject types.

##### Returns:

  `self`

##### Raises:


*  <b>`RuntimeError`</b>: If `init_loom()` has already been called.
*  <b>`TypeError`</b>: If `root_block_like` cannot be converted to a block.
*  <b>`TypeError`</b>: If `root_block_like` fails to compile.
*  <b>`TypeError`</b>: If `root_block_like` has no output or metric tensors.
*  <b>`TypeError`</b>: If `root_block_like` has an invalid output type.


- - -

<a name="td.Compiler.create"></a>
#### `td.Compiler.create(cls, root_block_like, max_depth=None, loom_input_tensor=None, input_tensor=None, parallel_iterations=None, back_prop=None, swap_memory=None)`

Creates a Compiler, compiles a block, and initializes loom.

##### Args:


*  <b>`root_block_like`</b>: A block or an object that can be converted to a block by
    [`td.convert_to_block`](#td.convert_to_block). Must have at least one
    output or metric tensor. The output type may not contain any
    Sequence or PyObject types.
*  <b>`max_depth`</b>: Optional. The maximum nesting depth that the encapsulated loom
    ought to support. This is dependent on the topology of the block graph
    and on the shape of the data being passed in. May be calculated by
    calling [`Compiler.max_depth`](#td.Compiler.max_depth). If unspecified,
    a `tf.while_loop` will be used to dynamically calculate `max_depth` on
    a per-batch basis.
*  <b>`loom_input_tensor`</b>: An optional string tensor of loom inputs for the
    compiler to read from. Mutually exclusive with `input_tensor'.
*  <b>`input_tensor`</b>: an optional string tensor for the block to read inputs from.
    If an input_tensor is supplied the user can just evaluate the compiler's
    output tensors without needing to create a feed dict via
    'build_feed_dict'. Mutually exclusive with `loom_input_tensor'.
*  <b>`parallel_iterations`</b>: tf.while_loop's parallel_iterations option, which
    caps the number of different depths at which ops could run in parallel.
    Only applies when max_depth=None. Default: 10.
*  <b>`back_prop`</b>: tf.while_loop's back_prop option, which enables gradients. Only
    applies when max_depth=None.  Default: True.
*  <b>`swap_memory`</b>: Whether to use tf.while_loop's swap_memory option, which
    enables swapping memory between GPU and CPU at the possible expense of
    some performance. Only applies when max_depth=None. Default: False.

##### Returns:

  A fully initialized Compiler.

##### Raises:


*  <b>`TypeError`</b>: If `root_block_like` cannot be converted to a block.
*  <b>`TypeError`</b>: If `root_block_like` fails to compile.
*  <b>`TypeError`</b>: If `root_block_like` has no output or metric tensors.
*  <b>`TypeError`</b>: If `root_block_like` has an invalid output type.
*  <b>`ValueError`</b>: If both `loom_input_tensor` and `input_tensor` are provided.


- - -

<a name="td.Compiler.init_loom"></a>
#### `td.Compiler.init_loom(max_depth=None, loom_input_tensor=None, input_tensor=None, parallel_iterations=None, back_prop=None, swap_memory=None)`

Intializes the loom object, which is used to run on tensorflow.

##### Args:


*  <b>`max_depth`</b>: Optional. The maximum nesting depth that the encapsulated loom
    ought to support. This is dependent on the topology of the block graph
    and on the shape of the data being passed in. May be calculated by
    calling [`Compiler.max_depth`](#td.Compiler.max_depth). If unspecified,
    a `tf.while_loop` will be used to dynamically calculate `max_depth` on
    a per-batch basis.
*  <b>`loom_input_tensor`</b>: An optional string tensor of loom inputs for the
    compiler to read from. Mutually exclusive with `input_tensor'.
*  <b>`input_tensor`</b>: an optional string tensor for the block to read inputs from.
    If an input_tensor is supplied the user can just evaluate the compiler's
    output tensors without needing to create a feed dict via
    'build_feed_dict'. Mutually exclusive with `loom_input_tensor'.
*  <b>`parallel_iterations`</b>: tf.while_loop's parallel_iterations option, which
    caps the number of different depths at which ops could run in parallel.
    Only applies when max_depth=None. Default: 10.
*  <b>`back_prop`</b>: tf.while_loop's back_prop option, which enables gradients. Only
    applies when max_depth=None.  Default: True.
*  <b>`swap_memory`</b>: Whether to use tf.while_loop's swap_memory option, which
    enables swapping memory between GPU and CPU at the possible expense of
    some performance. Only applies when max_depth=None. Default: False.

##### Raises:


*  <b>`RuntimeError`</b>: If `compile()` has not been called.
*  <b>`RuntimeError`</b>: If the loom has already been initialized.
*  <b>`ValueError`</b>: If both `loom_input_tensor` and `input_tensor` are provided.


- - -

<a name="td.Compiler.input_tensor"></a>
#### `td.Compiler.input_tensor`

Returns input tensor that can feed data to this compiler.


- - -

<a name="td.Compiler.is_loom_initialized"></a>
#### `td.Compiler.is_loom_initialized`




- - -

<a name="td.Compiler.loom_input_tensor"></a>
#### `td.Compiler.loom_input_tensor`

Returns the loom input tensor, used for building feed dictionaries.

May be fed a single result or a sequence of results from
`Compiler.build_loom_inputs()` or `Compiler.build_loom_input_batched()`.

##### Returns:

  A string tensor.

##### Raises:


*  <b>`RuntimeError`</b>: If `Compiler.init_loom()` has not been called.


- - -

<a name="td.Compiler.max_depth"></a>
#### `td.Compiler.max_depth(inp)`

Returns the loom `max_depth` needed to evaluate `inp`.


- - -

<a name="td.Compiler.metric_tensors"></a>
#### `td.Compiler.metric_tensors`

Returns a ordered dictionary of tensors for output metrics.


- - -

<a name="td.Compiler.multiprocessing_pool"></a>
#### `td.Compiler.multiprocessing_pool(processes=None)`

Creates a context for use with the Python `with` statement.

Entering this context creates a pool of subprocesses for building
loom inputs in parallel with this compiler. When the context exits
the pool is closed, blocking until all work is completed.

##### Args:


*  <b>`processes`</b>: The number of worker processes to use. Defaults to the
  cpu count (`multiprocessing.cpu_count()`).

##### Yields:

  Nothing.

##### Raises:


*  <b>`RuntimeError`</b>: If [`init_loom()`](#td.Compiler.init_loom) has not been
    called.


- - -

<a name="td.Compiler.output_tensors"></a>
#### `td.Compiler.output_tensors`

Returns a flattened list of all output tensors.


- - -

<a name="td.Compiler.pool"></a>
#### `td.Compiler.pool`

Returns the current multiprocessing pool if it exists, else None.


- - -

<a name="td.Compiler.root"></a>
#### `td.Compiler.root`

Returns the root block, or None if `compile()` has not been called.




## Blocks for input

- - -

<a name="td.Tensor"></a>
### `class td.Tensor`

A block that converts its input from a python object to a tensor.
- - -

<a name="td.Tensor.__init__"></a>
#### `td.Tensor.__init__(shape, dtype='float32', name=None)`





- - -

<a name="td.Scalar"></a>
### `td.Scalar(dtype='float32', name=None)`

A block that converts its input to a scalar.


- - -

<a name="td.Vector"></a>
### `td.Vector(size, dtype='float32', name=None)`

A block that converts its input to a vector.


- - -

<a name="td.InputTransform"></a>
### `class td.InputTransform`

A Python function, lifted to a block.
- - -

<a name="td.InputTransform.__init__"></a>
#### `td.InputTransform.__init__(py_fn, name=None)`




- - -

<a name="td.InputTransform.py_fn"></a>
#### `td.InputTransform.py_fn`





- - -

<a name="td.SerializedMessageToTree"></a>
### `td.SerializedMessageToTree(message_type_name)`

A block that turns serialized protobufs into nested Python dicts and lists.

The block's input and output types are both `PyObjectType`.

##### Args:


*  <b>`message_type_name`</b>: A string; the full name of the expected message type.

##### Returns:

  A dictionary of the message's values by fieldname, where the
  function renders repeated fields as lists, submessages via
  recursion, and enums as dictionaries whose keys are `name`,
  `index`, and `number`. Missing optional fields are rendered as
  `None`. Scalar field values are rendered as themselves.

##### Raises:


*  <b>`TypeError`</b>: If `message_type_name` is not a string.


- - -

<a name="td.OneHot"></a>
### `class td.OneHot`

A block that converts PyObject input to a one-hot encoding.

Will raise an `KeyError` if the block is applied to an out-of-range input.
- - -

<a name="td.OneHot.__init__"></a>
#### `td.OneHot.__init__(start, stop=None, dtype='float32', name=None)`

Initializes the block.

##### Args:


*  <b>`start`</b>: The start of the input range.
*  <b>`stop`</b>: Upper limit (exclusive) on the input range. If stop is `None`, the
    range is `[0, start)`, like the Python range function.
*  <b>`dtype`</b>: The dtype for the output array.
*  <b>`name`</b>: An optional string name for the block.

##### Raises:


*  <b>`IndexError`</b>: If the range is empty.



- - -

<a name="td.OneHotFromList"></a>
### `td.OneHotFromList(elements, dtype='float32', strict=True, name=None)`

A block that converts PyObject input to a one-hot encoding.

Differs from `OneHot` in that the user specifies the elements covered by the
one-hot encoding rather than assuming they are consecutive integers.

##### Args:


*  <b>`elements`</b>: The list of elements to be given one-hot encodings.
*  <b>`dtype`</b>: The type of the block's return value.
*  <b>`strict`</b>: Whether the block should throw a KeyError if it encounters an input
    which wasn't in elements.  Default: True.
*  <b>`name`</b>: An optional string name for the block.

##### Raises:


*  <b>`AssertionError`</b>: if any of the `elements` given are equal.

##### Returns:

  A Block that takes a PyObject and returns a tensor of type `dtype` and shape
  `[len(elements)]`.  If passed any member of `elements` the block will return
  a basis vector corresponding to the position of the element in the list.  If
  passed anything else the block will throw a KeyError if `strict` was set to
  True, and return the zero vector if `strict` was set to False.


- - -

<a name="td.Optional"></a>
### `class td.Optional`

Dispatches its input based on whether the input exists, or is None.

Similar to `OneOf(lambda x: x is None, {True: none_block, False: some_block})`
except that `none_block` has `input_type` `VoidType`.
- - -

<a name="td.Optional.__init__"></a>
#### `td.Optional.__init__(some_case, none_case=None, name=None)`

Creates an Optional block.

##### Args:


*  <b>`some_case`</b>: The block to evaluate on x if x exists.
*  <b>`none_case`</b>: The block to evaluate if x is None -- defaults to zeros for
    tensor types, and an empty sequence for sequence types.
*  <b>`name`</b>: An optional string name for the block.




## Blocks for composition

- - -

<a name="td.Composition"></a>
### `class td.Composition`

A composition of blocks, which are connected in a DAG.
- - -

<a name="td.Composition.__init__"></a>
#### `td.Composition.__init__(children=None, name=None)`




- - -

<a name="td.Composition.connect"></a>
#### `td.Composition.connect(a, b)`

Connect `a` to the input of `b`.

The argument `a` can be either:

* A block, in which case the output of `a` is fed into the input of `b`.

* The i^th output of a block, obtained from `a[i]`.

* A tuple or list of blocks or block outputs.

##### Args:


*  <b>`a`</b>: Inputs to the block (see above).
*  <b>`b`</b>: The block to connect the inputs to.

##### Raises:


*  <b>`ValueError`</b>: if `a` includes the output of the composition.
*  <b>`ValueError`</b>: if `b` is the input of the composition.
*  <b>`ValueError`</b>: if the input of `b` is already connected.


- - -

<a name="td.Composition.input"></a>
#### `td.Composition.input`

Return a placeholder whose output is the input to the composition.


- - -

<a name="td.Composition.output"></a>
#### `td.Composition.output`

Return a placeholder whose input is the output of the composition.


- - -

<a name="td.Composition.scope"></a>
#### `td.Composition.scope()`

Creates a context for use with the python `with` statement.

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



- - -

<a name="td.Pipe"></a>
### `td.Pipe(*blocks, **kwargs)`

Creates a composition which pipes each block into the next one.

`Pipe(a, b, c)` is equivalent to `a >> b >> c`.

```python
Pipe(a, b, c).eval(x) => c(b(a(x)))
```

##### Args:


*  <b>`*blocks`</b>: A tuple of blocks.
*  <b>`**kwargs`</b>: Optional keyword arguments.  Accepts name='block_name'.

##### Returns:

  A block.


- - -

<a name="td.Record"></a>
### `class td.Record`

Dispatch each element of a dict, list, or tuple to child blocks.

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
- - -

<a name="td.Record.__init__"></a>
#### `td.Record.__init__(named_children, name=None)`

Create a Record Block.

If named_children is list or tuple or ordered dict, then the
output tuple of the Record will preserve child order, otherwise
the output tuple will be ordered by key.

##### Args:


*  <b>`named_children`</b>: A dictionary, list of (key, block) pairs, or a
    tuple of blocks (in which case the keys are 0, 1, 2, ...).
*  <b>`name`</b>: An optional string name for the block.



- - -

<a name="td.AllOf"></a>
### `td.AllOf(*blocks, **kwargs)`

A block that runs all of its children (conceptually) in parallel.

```python
AllOf().eval(inp) => None
AllOf(a).eval(inp) => (a.eval(inp),)
AllOf(a, b, c).eval(inp) => (a.eval(inp), b.eval(inp), c.eval(inp))
```

##### Args:


*  <b>`*blocks`</b>: Blocks.
*  <b>`**kwargs`</b>: {name: name_string} or {}.

##### Returns:

  See above.



## Blocks for tensors

- - -

<a name="td.FromTensor"></a>
### `class td.FromTensor`

A block that returns a particular TF tensor or NumPy array.
- - -

<a name="td.FromTensor.__init__"></a>
#### `td.FromTensor.__init__(tensor, name=None)`

Creates the block.

##### Args:


*  <b>`tensor`</b>: A TF tensor or variable with a complete shape, or a NumPy array.
*  <b>`name`</b>: A string. Defaults to the name of `tensor` if it has one.

##### Raises:


*  <b>`TypeError`</b>: If `tensor` is not a TF tensor or variable or NumPy array.
*  <b>`TypeError`</b>: If  `tensor` does not have a complete shape.


- - -

<a name="td.FromTensor.tensor"></a>
#### `td.FromTensor.tensor`





- - -

<a name="td.Function"></a>
### `class td.Function`

A TensorFlow function, wrapped in a block.

The TensorFlow function that is passed into a `Function` block must be a batch
version of the operation you want.  This doesn't matter for things like
element-wise addition `td.Function(tf.add)`, but if you, for example, want a
`Function` block that multiplies matrices, you need to call
`td.Function(tf.batch_matmul)`.  This is done for efficiency reasons, so that
calls to the same function can be batched together naturally and take
advantage of TensorFlow's parallelism.
- - -

<a name="td.Function.__init__"></a>
#### `td.Function.__init__(tf_fn, name=None, infer_output_type=True)`

Creates a `Function` block.

##### Args:


*  <b>`tf_fn`</b>: The batch version of the TensorFlow function to be evaluated.
*  <b>`name`</b>: An optional string name for the block. If present, must be a valid
    name for a TensorFlow scope.
*  <b>`infer_output_type`</b>: A bool; whether or not to infer the output type of
    of the block by invoking `tf_fn` once on dummy placeholder. If False,
    you will probably need to call `set_output_type()` explicitly.


- - -

<a name="td.Function.tf_fn"></a>
#### `td.Function.tf_fn`





- - -

<a name="td.Concat"></a>
### `class td.Concat`

Concatenates a non-empty tuple of tensors into a single tensor.
- - -

<a name="td.Concat.__init__"></a>
#### `td.Concat.__init__(concat_dim=0, flatten=False, name=None)`

Create a Concat block.

##### Args:


*  <b>`concat_dim`</b>: The dimension to concatenate along (not counting the batch
    dimension).
*  <b>`flatten`</b>: Whether or not to recursively concatenate nested tuples of
    tensors. Default is False, in which case we throw on nested tuples.
*  <b>`name`</b>: An optional string name for the block. If present, must be a valid
    name for a TensorFlow scope.



- - -

<a name="td.Zeros"></a>
### `td.Zeros(output_type, name=None)`

A block of zeros, voids, and empty sequences of `output_type`.

If `output_type` is a tensor type, the output is `tf.zeros` of this
type. If it is a tuple type, the output is a tuple of `Zeros` of the
corresponding item types. If it is void, the output is void. If it
is a sequence type, the output is an empty sequence of this type.

##### Args:


*  <b>`output_type`</b>: A type. May not contain pyobject types.
*  <b>`name`</b>: An optional string name for the block.

##### Returns:

  A block.

##### Raises:


*  <b>`TypeError`</b>: If `output_type` contains pyobject types.



## Blocks for sequences

- - -

<a name="td.Map"></a>
### `class td.Map`

Map a block over a sequence or tuple.
- - -

<a name="td.Map.__init__"></a>
#### `td.Map.__init__(elem_block, name=None)`




- - -

<a name="td.Map.element_block"></a>
#### `td.Map.element_block`





- - -

<a name="td.Fold"></a>
### `class td.Fold`

Left-fold a two-argument block over a sequence or tuple.
- - -

<a name="td.Fold.__init__"></a>
#### `td.Fold.__init__(combine_block, start_block, name=None)`




- - -

<a name="td.Fold.combine_block"></a>
#### `td.Fold.combine_block`




- - -

<a name="td.Fold.start_block"></a>
#### `td.Fold.start_block`





- - -

<a name="td.RNN"></a>
### `td.RNN(cell, initial_state=None, initial_state_from_input=False, name=None)`

Create an RNN block.

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

##### Args:


*  <b>`cell`</b>: a block or layer that takes (input_elem, state) as input and
        produces (output_elem, state) as output.
*  <b>`initial_state`</b>: an (optional) tensor or block to use for the initial state.
*  <b>`initial_state_from_input`</b>: if True, pass the initial state as an input
                             to the RNN block, otherwise use initial_state.
*  <b>`name`</b>: An optional string name.

##### Raises:


*  <b>`ValueError`</b>: if initial_state_from_input == True and initial_state != None

##### Returns:

  a block.


- - -

<a name="td.Reduce"></a>
### `class td.Reduce`

Reduce a two-argument block over a sequence or tuple.
- - -

<a name="td.Reduce.__init__"></a>
#### `td.Reduce.__init__(combine_block, default_block=None, name=None)`




- - -

<a name="td.Reduce.combine_block"></a>
#### `td.Reduce.combine_block`




- - -

<a name="td.Reduce.default_block"></a>
#### `td.Reduce.default_block`





- - -

<a name="td.Sum"></a>
### `td.Sum(name=None)`

Sums its inputs.


- - -

<a name="td.Min"></a>
### `td.Min(name=None)`

Takes the minimum of its inputs.  Zero on no inputs.


- - -

<a name="td.Max"></a>
### `td.Max(name=None)`

Takes the maximum of its inputs.  Zero on no inputs.


- - -

<a name="td.Mean"></a>
### `td.Mean(name=None)`

Takes the average of its inputs.  Zero on no inputs.


- - -

<a name="td.Broadcast"></a>
### `class td.Broadcast`

Block that creates an infinite sequence of the same element.

This is useful in conjunction with `Zip` and `Map`, for example:

```python
def center_seq(seq_block):
  return (seq_block >> AllOf(Identity(), Mean() >> Broadcast()) >> Zip() >>
          Map(Function(tf.sub)))
```
- - -

<a name="td.Broadcast.__init__"></a>
#### `td.Broadcast.__init__(name=None)`





- - -

<a name="td.Zip"></a>
### `class td.Zip`

Converts a tuple of sequences to a sequence of tuples.

The output sequence is truncated in length to the length of the
shortest input sequence.
- - -

<a name="td.Zip.__init__"></a>
#### `td.Zip.__init__(name=None)`





- - -

<a name="td.ZipWith"></a>
### `td.ZipWith(elem_block, name=None)`

A Zip followed by a Map.

```python
ZipWith(elem_block) => Zip() >> Map(elem_block)
```

##### Args:


*  <b>`elem_block`</b>: A block with a tuple input type.
*  <b>`name`</b>: An optional string name for the block.

##### Returns:

  A block zips its input then maps over it with `elem_block`.


- - -

<a name="td.NGrams"></a>
### `class td.NGrams`

Computes tuples of n-grams over a sequence.

```python
(Map(Scalar()) >> NGrams(2)).eval([1, 2, 3]) => [(1, 2), (2, 3)]
```
- - -

<a name="td.NGrams.__init__"></a>
#### `td.NGrams.__init__(n, name=None)`




- - -

<a name="td.NGrams.n"></a>
#### `td.NGrams.n`





- - -

<a name="td.Nth"></a>
### `class td.Nth`

Extracts the Nth element of a sequence, where N is a PyObject.

```python
block = (Map(Scalar()), Identity()) >> Nth()
block.eval((list, n)) => list[n]
```
- - -

<a name="td.Nth.__init__"></a>
#### `td.Nth.__init__(name=None)`





- - -

<a name="td.GetItem"></a>
### `class td.GetItem`

A block that calls Pythons getitem operator (i.e. [] syntax) on its input.

The input type may be a PyObject, a Tuple, or a finite Sequence.

```python
(GetItem(key) >> block).eval(inp) => block.eval(inp[key])
```

Will raise a `KeyError` if applied to an input where the key cannot be found.
- - -

<a name="td.GetItem.__init__"></a>
#### `td.GetItem.__init__(key, name=None)`




- - -

<a name="td.GetItem.key"></a>
#### `td.GetItem.key`





- - -

<a name="td.Length"></a>
### `class td.Length`

A block that returns the length of its input.
- - -

<a name="td.Length.__init__"></a>
#### `td.Length.__init__(dtype='float32', name=None)`





- - -

<a name="td.Slice"></a>
### `td.Slice(*args, **kwargs)`

A block which applies Python slicing to a PyObject, Tuple, or Sequence.

For example, to reverse a sequence:
```python
(Map(Scalar()) >> Slice(step=-1)).eval(range(5)) => [4, 3, 2, 1, 0]
```

Positional arguments are not accepted in order to avoid the ambiguity
of slice(start=N) vs. slice(stop=N).

##### Args:


*  <b>`*args`</b>: Positional arguments; must be empty (see above).
*  <b>`**kwargs`</b>: Keyword arguments; `start=None, stop=None, step=None, name=None`.

##### Returns:

  The block.



## Other blocks

- - -

<a name="td.ForwardDeclaration"></a>
### `class td.ForwardDeclaration`

A ForwardDeclaration is used to define Blocks recursively.

Usage:

```python
fwd = ForwardDeclaration(in_type, out_type)  # declare type of block
block = ... fwd() ... fwd() ...              # define block recursively
fwd.resolve_to(block)                        # resolve forward declaration
```
- - -

<a name="td.ForwardDeclaration.__init__"></a>
#### `td.ForwardDeclaration.__init__(input_type=None, output_type=None, name=None)`




- - -

<a name="td.ForwardDeclaration.resolve_to"></a>
#### `td.ForwardDeclaration.resolve_to(target_block)`

Resolve the forward declaration by setting it to the given block.



- - -

<a name="td.OneOf"></a>
### `class td.OneOf`

A block that dispatches its input to one of its children.

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
- - -

<a name="td.OneOf.__init__"></a>
#### `td.OneOf.__init__(key_fn, case_blocks, pre_block=None, name=None)`

Creates the OneOf block.

##### Args:


*  <b>`key_fn`</b>: A python function or a block with `PyObject` output type,
    which returns a key, when given an input.  The key will be used to
    look up a child in `case_blocks` for dispatch.
*  <b>`case_blocks`</b>: A non-empty Python dict, list of (key, block) pairs, or tuple
    of blocks (in which case the keys are 0, 1, 2, ...), where each block
    has the same input type `T` and the same output type.
*  <b>`pre_block`</b>: An optional block with output type `T`.  If specified,
    pre_block will be used to pre-process the
    input before the input is handed to one of `case_blocks`.
*  <b>`name`</b>: An optional string name for the block.

##### Raises:


*  <b>`ValueError`</b>: If `case_blocks` is empty.



- - -

<a name="td.Metric"></a>
### `class td.Metric`

A block that computes a metric.

Metrics are used in Fold when the size of a model's output is not
fixed, but varies as a function of the input data. They are also
handy for accumulating results across sequential and recursive
computations without having the thread them through explicitly as
return values.

For example, to create a block `y` that takes a (label, prediction)
as input, adds an L2 `'loss'` metric, and returns the prediction as
its output, you could say:

```python
y = Composition()
with y.scope():
  label = y.input[0]
  prediction = y.input[1]
  l2 = (Function(tf.sub) >> Function(tf.nn.l2_loss)).reads(label, prediction)
  Metric('loss').reads(l2)
  y.output.reads(prediction)
```

The input type of the block must be a `TensorType`, or a
`(TensorType, PyObjectType)` tuple.
The output type is always `VoidType`. In the tuple input case, the
second item of the tuple becomes a label for the tensor value, which
can be used to identify where the value came from in a nested data
structure and/or batch of inputs.

For example:

```python
sess = tf.InteractiveSession()
# We pipe Map() to Void() because blocks with sequence output types
# cannot be compiled.
block = td.Map(td.Scalar() >> td.Metric('foo')) >> td.Void()
compiler = td.Compiler.create(block)
sess.run(compiler.metric_tensors['foo'],
         compiler.build_feed_dict([range(3), range(4)])) =>
  array([ 0.,  1.,  2.,  0.,  1.,  2.,  3.], dtype=float32)
```

Or with labels:

```python
sess = tf.InteractiveSession()
block = td.Map((td.Scalar(), td.Identity()) >> td.Metric('bar')) >> td.Void()
compiler = td.Compiler.create(block)
feed_dict, metric_labels = compiler.build_feed_dict(
    [[(0, 'zero'), (1, 'one')], [(2, 'two')]],
    metric_labels=True)
metric_labels  =>  {'bar': ['zero', 'one', 'two']}
sess.run(compiler.metric_tensors['bar'], feed_dict)  =>
    array([ 0.,  1.,  2.], dtype=float32)
```
- - -

<a name="td.Metric.__init__"></a>
#### `td.Metric.__init__(metric_name)`





- - -

<a name="td.Identity"></a>
### `class td.Identity`

A block that merely returns its input.
- - -

<a name="td.Identity.__init__"></a>
#### `td.Identity.__init__(name=None)`





- - -

<a name="td.Void"></a>
### `td.Void(name=None)`

A block with void output type that accepts any input type.



## Layers

- - -

<a name="td.FC"></a>
### `class td.FC`

A fully connected network layer.

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

##### Attributes:


*  <b>`weights`</b>: The tensor for the weights of the FC layer.
*  <b>`bias`</b>: The tensor for the bias of the FC layer.
*  <b>`scales`</b>: The tensor for the scales of the FC layer if weight norm is enabled.
*  <b>`output_size`</b>: The size of the output as an integer.


- - -

<a name="td.FC.__init__"></a>
#### `td.FC.__init__(num_units_out, activation=relu, initializer=None, input_keep_prob=None, output_keep_prob=None, normalization_fn=None, weight_norm=False, name=None)`

Initializes the layer.

##### Args:


*  <b>`num_units_out`</b>: The number of output units in the layer.
*  <b>`activation`</b>: The activation function. Default is ReLU. Use `None` to get a
    linear layer.
*  <b>`initializer`</b>: The initializer for the weights. Defaults to uniform unit
    scaling with factor derived in <http://arxiv.org/pdf/1412.6558v3.pdf>
    if activation is ReLU, ReLU6, tanh, or linear. Otherwise defaults to
    truncated normal initialization with a standard deviation of 0.01.
*  <b>`input_keep_prob`</b>: Optional scalar float32 tensor for dropout on input.
    Feed 1.0 at serving to disable dropout.
*  <b>`output_keep_prob`</b>: Optional scalar float32 tensor for dropout on output.
    Feed 1.0 at serving to disable dropout.
*  <b>`normalization_fn`</b>: Optional normalization function that will be inserted
    before nonlinearity.
*  <b>`weight_norm`</b>: A bool to control whether weight normalization is used. See
*  <b>`https`</b>: //arxiv.org/abs/1602.07868 for how it works.
*  <b>`name`</b>: An optional string name. Defaults to `FC_%d % num_units_out`. Used
    to name the variable scope where the variables for the layer live.


- - -

<a name="td.FC.bias"></a>
#### `td.FC.bias`




- - -

<a name="td.FC.output_size"></a>
#### `td.FC.output_size`




- - -

<a name="td.FC.scales"></a>
#### `td.FC.scales`




- - -

<a name="td.FC.weight_norm"></a>
#### `td.FC.weight_norm`




- - -

<a name="td.FC.weights"></a>
#### `td.FC.weights`





- - -

<a name="td.Embedding"></a>
### `class td.Embedding`

An embedding for integers.

Embeddings require integer scalars as input, and build `float32` vector
outputs. Embeddings can be applied to multiple inputs. `Embedding` doesn't
do any hashing on its own, it just takes its inputs mod `num_buckets`
to determine which embedding(s) to return.

Implementation detail: `tf.gather` currently only supports `int32`
and `int64`. If the input type is smaller than 32 bits it will be
cast to `tf.int32`. Since all currently defined TF dtypes other than
`int32` and `int64` have less than 32 bits, this means that we
support all current integer dtypes.
- - -

<a name="td.Embedding.__init__"></a>
#### `td.Embedding.__init__(num_buckets, num_units_out, initializer=None, name=None, trainable=True, mod_inputs=True)`

Initializes the layer.

##### Args:


*  <b>`num_buckets`</b>: How many buckets the embedding has.
*  <b>`num_units_out`</b>: The number of output units in the layer.
*  <b>`initializer`</b>: the initializer for the weights. Defaults to uniform unit
    scaling. The initializer can also be a Tensor or numpy array, in which
    case the weights are initialized to this value and shape. Note that in
    this case the weights will still be trainable unless you also pass
    `trainable=False`.
*  <b>`name`</b>: An optional string name. Defaults to
    `Embedding_%d_%d % (num_buckets, num_units_out)`. Used to name the
    variable scope where the variables for the layer live.
*  <b>`trainable`</b>: Whether or not to make the weights trainable.
*  <b>`mod_inputs`</b>: Whether or not to mod the input by the number of buckets.

##### Raises:


*  <b>`ValueError`</b>: If the shape of `weights` is not
    `(num_buckets, num_units_out)`.


- - -

<a name="td.Embedding.num_buckets"></a>
#### `td.Embedding.num_buckets`




- - -

<a name="td.Embedding.num_units_out"></a>
#### `td.Embedding.num_units_out`




- - -

<a name="td.Embedding.weights"></a>
#### `td.Embedding.weights`





- - -

<a name="td.FractalNet"></a>
### `class td.FractalNet`

An implementation of FractalNet.

See https://arxiv.org/abs/1605.07648 for details.
- - -

<a name="td.FractalNet.__init__"></a>
#### `td.FractalNet.__init__(num_fractal_blocks, fractal_block_depth, base_layer_builder, mixer=None, drop_path=False, p_local_drop_path=0.5, p_drop_base_case=0.25, p_drop_recursive_case=0.25, name=None)`

Initializes the FractalNet.

##### Args:


*  <b>`num_fractal_blocks`</b>: The number of fractal blocks the net is made from.
    This variable is named `B` in the FractalNet paper.  This argument uses
    the word `block` in the sense that the FractalNet paper uses it.
*  <b>`fractal_block_depth`</b>: How deeply nested the blocks are.  This variable is
    `C-1` in the paper.
*  <b>`base_layer_builder`</b>: A callable that takes a name and returns a `Layer`
    object.  We would pass in a convolutional layer to reproduce the results
    in the paper.
*  <b>`mixer`</b>: The join operation in the paper.  Assumed to have two arguments.
    Defaults to element-wise averaging.  Mixing doesn't occur if either path
    gets dropped.
*  <b>`drop_path`</b>: A boolean, whether or not to do drop-path.  Defaults to False.
    If selected, we do drop path as described in the paper (unless drop-path
    choices is provided in which case how drop path is done can be further
    customized by the user.
*  <b>`p_local_drop_path`</b>: A probability between 0.0 and 1.0.  0.0 means always do
    global drop path.  1.0 means always do local drop path.  Default: 0.5,
    as in the paper.
*  <b>`p_drop_base_case`</b>: The probability, when doing local drop path, to drop the
    base case.
*  <b>`p_drop_recursive_case`</b>: The probability, when doing local drop path, to
    drop the recusrive case. (Requires: `p_drop_base_case +
    p_drop_recursive_case < 1`)
*  <b>`name`</b>: An optional string name.


- - -

<a name="td.FractalNet.drop_path"></a>
#### `td.FractalNet.drop_path`





- - -

<a name="td.ScopedLayer"></a>
### `class td.ScopedLayer`

Create a Fold Layer that wraps a TensorFlow layer or RNN cell.

The default TensorFlow mechanism for weight sharing is to use
tf.variable_scope, but this requires that a scope parameter be passed
whenever the layer is invoked.  ScopedLayer stores a TensorFlow layer,
along with its variable scope, and passes the scope appropriately.
For example:

```
gru_cell1 = td.ScopedLayer(tf.contrib.rnn.GRUCell(num_units=16), 'gru1')
... td.RNN(gru_cell1) ...
```
- - -

<a name="td.ScopedLayer.__init__"></a>
#### `td.ScopedLayer.__init__(layer_fn, name_or_scope=None)`

Wrap a TensorFlow layer.

##### Args:


*  <b>`layer_fn`</b>: A callable that accepts and returns nests of batched tensors. A
    nest of tensors is either a tensor or a sequence of nests of tensors.
    Must also accept a `scope` keyword argument. For example, may be an
    instance of `tf.contrib.rnn.RNNCell`.
*  <b>`name_or_scope`</b>: A variable scope or a string to use as the scope name.


- - -

<a name="td.ScopedLayer.output_size"></a>
#### `td.ScopedLayer.output_size`




- - -

<a name="td.ScopedLayer.state_size"></a>
#### `td.ScopedLayer.state_size`






## Types

- - -

<a name="td.TensorType"></a>
### `class td.TensorType`

Tensors (which may be numpy or tensorflow) of a particular shape.

Tensor types implement the numpy array protocol, which means that
e.g.  `np.ones_like(tensor_type)` will do what you expect it
to. Calling `np.array(tensor_type)` returns a zeroed array.
- - -

<a name="td.TensorType.__array__"></a>
#### `td.TensorType.__array__()`

Returns a zeroed numpy array of this type.


- - -

<a name="td.TensorType.__init__"></a>
#### `td.TensorType.__init__(shape, dtype='float32')`

Creates a tensor type.

##### Args:


*  <b>`shape`</b>: A tuple or list of non-negative integers.
*  <b>`dtype`</b>: A `tf.DType`, or stringified version thereof (e.g. `'int64'`).

##### Raises:


*  <b>`TypeError`</b>: If `shape` is not a tuple or list of non-negative integers.
*  <b>`TypeError`</b>: If `dtype` cannot be converted to a TF dtype.


- - -

<a name="td.TensorType.dtype"></a>
#### `td.TensorType.dtype`




- - -

<a name="td.TensorType.ndim"></a>
#### `td.TensorType.ndim`




- - -

<a name="td.TensorType.shape"></a>
#### `td.TensorType.shape`





- - -

<a name="td.VoidType"></a>
### `class td.VoidType`

A type used for blocks that don't return inputs or outputs.

- - -

<a name="td.PyObjectType"></a>
### `class td.PyObjectType`

The type of an arbitrary python object (usually used as an input type).

- - -

<a name="td.TupleType"></a>
### `class td.TupleType`

Type for fixed-length tuples of items, each of a particular type.

`TupleType` implements the sequence protocol, so e.g. `foo[0]` is
the type of the first item, `foo[2:4]` is a `TupleType` with the
expected item types, and `len(foo)` is the number of item types in
the tuple.
- - -

<a name="td.TupleType.__init__"></a>
#### `td.TupleType.__init__(*item_types)`

Creates a tuple type.

##### Args:


*  <b>`*item_types`</b>: A tuple of types or a single iterable of types.

##### Raises:


*  <b>`TypeError`</b>: If the items of `item_types` are not all types.



- - -

<a name="td.SequenceType"></a>
### `class td.SequenceType`

Type for variable-length sequences of elements all having the same type.
- - -

<a name="td.SequenceType.__init__"></a>
#### `td.SequenceType.__init__(elem_type)`

Creates a sequence type.

##### Args:


*  <b>`elem_type`</b>: A type.

##### Raises:


*  <b>`TypeError`</b>: If `elem_type` is not a type.


- - -

<a name="td.SequenceType.element_type"></a>
#### `td.SequenceType.element_type`





- - -

<a name="td.BroadcastSequenceType"></a>
### `class td.BroadcastSequenceType`

Type for infinite sequences of same element repeated.
- - -

<a name="td.BroadcastSequenceType.__init__"></a>
#### `td.BroadcastSequenceType.__init__(elem_type)`

Creates a sequence type.

##### Args:


*  <b>`elem_type`</b>: A type.

##### Raises:


*  <b>`TypeError`</b>: If `elem_type` is not a type.




## Plans

- - -

<a name="td.Plan"></a>
### `class td.Plan`

Base class for training, evaluation, and inference plans.

##### Attributes:


*  <b>`mode`</b>: One of 'train', 'eval', or 'infer'.
*  <b>`compiler`</b>: A `td.Compiler`, or None.
*  <b>`examples`</b>: An iterable of examples, or None.
*  <b>`metrics`</b>: An ordered dict from strings to real numeric tensors. These are
    used to make scalar summaries if they are scalars and histogram summaries
    otherwise.
*  <b>`losses`</b>: An ordered dict from strings to tensors.
*  <b>`num_multiprocess_processes`</b>: Number of worker processes to use for
    multiprocessing loom inputs. Default (None) is the CPU count.
    Set to zero to disable multiprocessing.
*  <b>`is_chief_trainer`</b>: A boolean indicating whether this is the chief training
        worker.
*  <b>`name`</b>: A string; defaults to 'plan'.
*  <b>`logdir`</b>: A string; used for saving/restoring checkpoints and summaries.
*  <b>`rundir`</b>: A string; the parent directory of logdir, shared between training
    and eval jobs for the same model.
*  <b>`plandir`</b>: A string; the parent directory of rundir, shared between many runs
    of different models on the same task.
*  <b>`master`</b>: A string; Tensorflow master to use.
*  <b>`save_summaries_secs`</b>: An integer; set to zero to disable summaries. In
    distributed training only the chief should set this to a non-zero value.
*  <b>`print_file`</b>: A file to print logging messages to; defaults to stdout.
*  <b>`should_stop`</b>: A callback to check for whether the training or eval jobs
    should be stopped.
*  <b>`report_loss`</b>: A callback for training and eval jobs to report losses.
*  <b>`report_done`</b>: A callback called by the eval jobs when they finish.


- - -

<a name="td.Plan.__init__"></a>
#### `td.Plan.__init__(mode)`




- - -

<a name="td.Plan.assert_runnable"></a>
#### `td.Plan.assert_runnable()`

Raises an exception if the plan cannot be run.


- - -

<a name="td.Plan.batch_size_placeholder"></a>
#### `td.Plan.batch_size_placeholder`

A placeholder for normalizing loss summaries.

##### Returns:

  A scalar placeholder if there are losses and finalize_stats() has been
  called, else None.


- - -

<a name="td.Plan.compute_summaries"></a>
#### `td.Plan.compute_summaries`

A bool; whether or not summaries are being computed.


- - -

<a name="td.Plan.create"></a>
#### `td.Plan.create(cls, mode)`

Creates a plan.

##### Args:


*  <b>`mode`</b>: A string; 'train', 'eval', or 'infer'.

##### Raises:


*  <b>`ValueError`</b>: If `mode` is invalid.

##### Returns:

  A Plan.


- - -

<a name="td.Plan.create_from_flags"></a>
#### `td.Plan.create_from_flags(cls, setup_plan_fn)`

Creates a plan from flags.

##### Args:


*  <b>`setup_plan_fn`</b>: A unary function accepting a plan as its argument. The
    function must assign the following attributes:
     * compiler
     * examples (excepting when batches are being read from the loom
       input tensor in train mode e.g. by a dequeuing worker)
     * losses (in train/eval mode)
     * outputs (in infer mode)

##### Returns:

  A runnable plan with finalized stats.

##### Raises:


*  <b>`ValueError`</b>: If flags are invalid.


- - -

<a name="td.Plan.create_from_params"></a>
#### `td.Plan.create_from_params(cls, setup_plan_fn, params)`

Creates a plan from a dictionary.

##### Args:


*  <b>`setup_plan_fn`</b>: A unary function accepting a plan as its argument. The
    function must assign the following attributes:
     * compiler
     * examples (excepting when batches are being read from the loom
       input tensor in train mode e.g. by a dequeuing worker)
     * losses (in train/eval mode)
     * outputs (in infer mode)
*  <b>`params`</b>: a dictionary to pull options from.

##### Returns:

  A runnable plan with finalized stats.

##### Raises:


*  <b>`ValueError`</b>: If params are invalid.


- - -

<a name="td.Plan.create_supervisor"></a>
#### `td.Plan.create_supervisor()`

Creates a TF supervisor for running the plan.


- - -

<a name="td.Plan.finalize_stats"></a>
#### `td.Plan.finalize_stats()`

Finalizes metrics and losses. Gets/creates global_step if unset.


- - -

<a name="td.Plan.global_step"></a>
#### `td.Plan.global_step`

The global step tensor.


- - -

<a name="td.Plan.has_finalized_stats"></a>
#### `td.Plan.has_finalized_stats`




- - -

<a name="td.Plan.init_loom"></a>
#### `td.Plan.init_loom(**loom_kwargs)`

Initializes compilers's loom.

The plan must have a compiler with a compiled root block and an
uninitialized loom.

In training mode this sets up enqueuing/dequeuing if num_dequeuers is
non-zero. When enqueuing, no actual training is performed; the
train op is to enqueue batches of loom inputs from `train_set`,
typically for some other training worker(s) to dequeue from. When
dequeuing, batches are read using a dequeue op, typically from a
queue that some other training worker(s) are enqueuing to.

##### Args:


*  <b>`**loom_kwargs`</b>: Arguments to `compiler.init_loom`. In enqueuing
    or dequeuing training `loom_input_tensor` may not be specified.

##### Returns:

  A pair of two bools `(needs_examples, needs_stats)`, indicating
  which of these requirements must be met in order for the plan to
  be runnable. In enqueuing training and in inference we need examples
  but not stats, whereas in dequeuing the obverse holds. In all
  other cases we need both examples and stats.

##### Raises:


*  <b>`ValueError`</b>: If `compiler` is missing.
*  <b>`RuntimeError`</b>: If `compile()` has not been called on the compiler.
*  <b>`RuntimeError`</b>: If the compiler's loom has already been initialized.


- - -

<a name="td.Plan.log_and_print"></a>
#### `td.Plan.log_and_print(msg)`




- - -

<a name="td.Plan.loss_total"></a>
#### `td.Plan.loss_total`

A scalar tensor, or None.

##### Returns:

  The total loss if there are losses and finalize_stats() has been called,
  else None.


- - -

<a name="td.Plan.run"></a>
#### `td.Plan.run(supervisor=None, session=None)`

Runs the plan with `supervisor` and `session`.

##### Args:


*  <b>`supervisor`</b>: A TF supervisor, or None. If None, a supervisor is created by
    calling `self.create_supervisor()`.
*  <b>`session`</b>: A TF session, or None. If None, a session is created by calling
    `session.managed_session(self.master)`. Will be installed as the default
    session while running the plan.

##### Raises:


*  <b>`ValueError`</b>: If the plan's attributes are invalid.
*  <b>`RuntimeError`</b>: If the plan has metrics or losses, and `finalize_stats()`
    has not been called.


- - -

<a name="td.Plan.summaries"></a>
#### `td.Plan.summaries`

A scalar string tensor, or None.

##### Returns:

  Merged summaries if compute_summaries is true and finalize_stats
  has been called, else None.



- - -

<a name="td.TrainPlan"></a>
### `class td.TrainPlan`

Plan class for training.

There are two primary training modes. When `examples` is present,
batches are created in Python and passed to TF as feeds. In this
case, `examples` must be non-empty. When `examples` is absent,
batches are read directly from the compiler's loom_input_tensor. In
the latter case, each batch must have exactly batch_size elements.

##### Attributes:


*  <b>`batches_per_epoch`</b>: An integer, or None; how many batches to
    consider an epoch when `examples` is absent. Has no effect on
    training when `examples` is present (because an epoch is defined
    as a full pass through the training set).
*  <b>`dev_examples`</b>: An iterable of development (i.e. validation) examples,
    or None.
*  <b>`train_op`</b>: An TF op, e.g. `Optimizer.minimize(loss)`, or None.
*  <b>`train_feeds`</b>: A dict of training feeds, e.g. keep probability for dropout.
*  <b>`epochs`</b>: An integer, or None.
*  <b>`batch_size`</b>: An integer, or None.
*  <b>`save_model_secs`</b>: An integer. Note that if a dev_examples is provided then we
    save the best performing models, and this is ignored.
*  <b>`task`</b>: An integer. This is a different integer from the rest;
    ps_tasks, num_dequeuers, queue_capacity all indicate some form of
    capacity, whereas this guy is task ID.
*  <b>`worker_replicas`</b>: An integer.
*  <b>`ps_tasks`</b>: An integer.
*  <b>`num_dequeuers`</b>: An integer.
*  <b>`queue_capacity`</b>: An integer.
*  <b>`optimizer_params`</b>: a dictionary mapping strings to optimizer arguments.
    Used only if train_op is not provided.
*  <b>`exact_batch_sizes`</b>: A bool; if true, `len(examples) % batch_size`
    items from the training set will be dropped each epoch to ensure
    that all batches have exactly `batch_size` items. Default is false.
    Has no effect if batches are being read from the compiler's loom
    input tensor. Otherwise, if true, `examples` must have at least
    `batch_size` items (to ensure that the training set is non-empty).


- - -

<a name="td.TrainPlan.__init__"></a>
#### `td.TrainPlan.__init__()`




- - -

<a name="td.TrainPlan.build_optimizer"></a>
#### `td.TrainPlan.build_optimizer()`





- - -

<a name="td.EvalPlan"></a>
### `class td.EvalPlan`

Plan class for evaluation.

##### Attributes:


*  <b>`eval_interval_secs`</b>: Time interval between eval runs (when running in a
    loop). Set to zero or None to run a single eval and then exit; in this
    case data will be streamed. Otherwise, data must fit in memory.
*  <b>`save_best`</b>: A boolean determining whether to save a checkpoint if this model
    has the best loss so far.
*  <b>`logdir_restore`</b>: A string or None; log directory for restoring checkpoints
    from.
*  <b>`batch_size`</b>: An integer (defaults to 10,000); maximal number of
    examples to pass to a single call to `Session.run()`. When streaming,
    this is also the maximal number of examples that will be materialized
    in-memory.


- - -

<a name="td.EvalPlan.__init__"></a>
#### `td.EvalPlan.__init__()`





- - -

<a name="td.InferPlan"></a>
### `class td.InferPlan`

Plan class for inference.

##### Attributes:


*  <b>`key_fn`</b>: A function from examples to keys, or None.
*  <b>`outputs`</b>: A list or tuple of tensors to be run to produce results, or None.
*  <b>`results_fn`</b>: A function that takes an iterable of `(key, result)` pairs if
  `key_fn` is present or `result`s otherwise; by default prints to stdout.

*  <b>`context_manager`</b>: A context manager for wrapping calls to `result_
*  <b>`batch_size`</b>: An integer (defaults to 10,000); maximal number of
    examples to materialize in-memory.
*  <b>`chunk_size`</b>: An integer (defaults to 100); chunk size for each unit
    of work, if multiprocessing.


- - -

<a name="td.InferPlan.__init__"></a>
#### `td.InferPlan.__init__()`





- - -

<a name="td.define_plan_flags"></a>
### `td.define_plan_flags(default_plan_name='plan', blacklist=None)`

Defines all of the flags used by `td.Plan.create_from_flags()`.

##### Args:


*  <b>`default_plan_name`</b>: A default value for the `--plan_name` flag.
*  <b>`blacklist`</b>: A set of string flag names to not define.


- - -

<a name="td.plan_default_params"></a>
### `td.plan_default_params()`

Returns a dict from plan option parameter names to their defaults.



## Conversion functions

- - -

<a name="td.convert_to_block"></a>
### `td.convert_to_block(block_like)`

Converts `block_like` to a block.

The conversion rules are as follows:

|type of `block_like`                   | result                   |
|-------------------------------------- | -------------------------|
|`Block`                                | `block_like`             |
|`Layer`                                | `Function(block_like)`   |
|`(tf.Tensor, tf.Variable, np.ndarray)` | `FromTensor(block_like)` |
|`(dict, list, tuple)`                  | `Record(block_like)`     |

##### Args:


*  <b>`block_like`</b>: Described above.

##### Returns:

  A block.

##### Raises:


*  <b>`TypeError`</b>: If `block_like` cannot be converted to a block.


- - -

<a name="td.convert_to_type"></a>
### `td.convert_to_type(type_like)`

Converts `type_like` to a `Type`.

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

##### Args:


*  <b>`type_like`</b>: Described above.

##### Returns:

  A `Type`.

##### Raises:


*  <b>`TypeError`</b>: If `type_like` cannot be converted to a `Type`.


- - -

<a name="td.canonicalize_type"></a>
### `td.canonicalize_type(type_like)`

Returns a canonical representation of a type.

Recursively applies a reduction rule that converts tuples/sequences
of `PyObjectType` to a single terminal `PyObjectType`.

```python
canonicalize_type((PyObjectType(), PyObjectType())) => PyObjectType()
canonicalize_type(SequenceType(PyObjectType())) => PyObjectType()
```

##### Args:


*  <b>`type_like`</b>: A type or an object convertible to one by `convert_to_type`.

##### Returns:

  A canonical representation of `type_like`.



## Utilities

- - -

<a name="td.EdibleIterator"></a>
### `class td.EdibleIterator`

A wrapper around an iterator that lets it be used as a TF feed value.

Edible iterators are useful when you have an expensive computation
running asynchronously that you want to feed repeatedly. For
example:

```python
items = my_expensive_function()  # returns an iterator, doesn't block
fd = {x: list(items)} # blocks here
while training:
  do_stuff()  # doesn't run until my_expensive_function() completes
  sess.run(fetches, fd)
```

With an edible iterator you can instead say:

```python
items = my_expensive_function()  # returns an iterator, doesn't block
fd = {x: EdibleIterator(items)} # doesn't block
while training:
  do_stuff()  # runs right away
  sess.run(fetches, fd)  # blocks here
```

Python iterators are only good for a single traversal. This means
that if you call `next()` before a feed dict is created, then that
value will be lost. When an edible iterator gets fed to TF the base
iterator is exhausted, but the results are cached, so the same
edible may be consumed repeatedly.

Implementation details: TF consumes feed values by converting them
to NumPy arrays. NumPy doesn't like iterators (otherwise we could
e.g use tee(), which would be simpler), so we use `__array__` to
tell NumPy how to do the conversion.
- - -

<a name="td.EdibleIterator.__array__"></a>
#### `td.EdibleIterator.__array__(dtype=None)`

NumPy array protocol; returns iterator values as an ndarray.


- - -

<a name="td.EdibleIterator.__init__"></a>
#### `td.EdibleIterator.__init__(iterable)`




- - -

<a name="td.EdibleIterator.__next__"></a>
#### `td.EdibleIterator.__next__()`




- - -

<a name="td.EdibleIterator.value"></a>
#### `td.EdibleIterator.value`

Returns iterator values as an ndarray if it exists, else None.



- - -

<a name="td.group_by_batches"></a>
### `td.group_by_batches(iterable, batch_size, truncate=False)`

Yields successive batches from an iterable, as lists.

##### Args:


*  <b>`iterable`</b>: An iterable.
*  <b>`batch_size`</b>: A positive integer.
*  <b>`truncate`</b>: A bool (default false). If true, then the last
    `len_iterable % batch_size` items are not yielded, ensuring that
    all batches have exactly `batch_size` items.

##### Yields:

  Successive batches from `iterable`, as lists of at most
  `batch_size` items.

##### Raises:


*  <b>`ValueError`</b>: If `batch_size` is non-positive.


- - -

<a name="td.epochs"></a>
### `td.epochs(items, n=None, shuffle=True, prng=None)`

Yields the items of an iterable repeatedly.

This function is particularly useful when `items` is expensive to compute
and you want to memoize it without blocking. For example:

```python
for items in epochs((my_expensive_function(x) for x in inputs), n):
  for item in items:
    f(item)
```

This lets `f(item)` run as soon as the first item is ready.

As an optimization, when n == 1 items itself is yielded without
memoization.

##### Args:


*  <b>`items`</b>: An iterable.
*  <b>`n`</b>: How many times to yield; zero or None (the default) means loop forever.
*  <b>`shuffle`</b>: Whether or not to shuffle the items after each yield. Shuffling is
    performed in-place. We don't shuffle before the first yield because this
    would require us to block until all of the items were ready.
*  <b>`prng`</b>: Nullary function returning a random float in [0.0, 1.0); defaults
    to `random.random`.

##### Yields:

  An iterable of `items`, `n` times.

##### Raises:


*  <b>`TypeError`</b>: If `items` is not an iterable.
*  <b>`ValueError`</b>: If `n` is negative.


- - -

<a name="td.parse_spec"></a>
### `td.parse_spec(spec)`

Parses a list of key values pairs.

##### Args:


*  <b>`spec`</b>: A comma separated list of strings of the form `<key>=<value>`.

##### Raises:


*  <b>`ValueError`</b>: If `spec` is malformed or contains duplicate keys.

##### Returns:

  A dict.


- - -

<a name="td.build_optimizer_from_params"></a>
### `td.build_optimizer_from_params(optimizer='adam', global_step=None, learning_rate_decay_params=None, **kwargs)`

Constructs an optimizer from key-value pairs.

For example

```python
build_optimizer_from_params('momentum', momentum=0.9, learning_rate=1e-3)
```
creates a MomentumOptimizer with momentum 0.9 and learning rate 1e-3.

##### Args:


*  <b>`optimizer`</b>: The name of the optimizer to construct.
*  <b>`global_step`</b>: The tensor of the global training step.
*  <b>`learning_rate_decay_params`</b>: The params to construct the learning rate decay
    algorithm. A dictionary.
*  <b>`**kwargs`</b>: Arguments for the optimizer's constructor.

##### Raises:


*  <b>`ValueError`</b>: If `optimizer` is unrecognized.
*  <b>`ValueError`</b>: If `kwargs` sets arguments that optimizer doesn't have, or
    fails to set arguments the optimizer requires.

##### Returns:

  A tf.train.Optimizer of the appropriate type.


- - -

<a name="td.create_variable_scope"></a>
### `td.create_variable_scope(name)`

Creates a new variable scope based on `name`, nested in the current scope.

If `name` ends with a `/` then the new scope will be created exactly as if
you called `tf.variable_scope(name)`.  Otherwise, `name` will be
made globally unique, in the context of the current graph (e.g.
`foo` will become `foo_1` if a `foo` variable scope already exists).

##### Args:


*  <b>`name`</b>: A non-empty string.

##### Returns:

  A variable scope.

##### Raises:


*  <b>`TypeError`</b>: if `name` is not a string.
*  <b>`ValueError`</b>: if `name` is empty.



## Abstract classes

- - -

<a name="td.IOBase"></a>
### `class td.IOBase`

Base class for objects with associated input/output types and names.
- - -

<a name="td.IOBase.__init__"></a>
#### `td.IOBase.__init__(input_type=None, output_type=None, name=None)`




- - -

<a name="td.IOBase.input_type"></a>
#### `td.IOBase.input_type`

Returns the input type if known, else None.


- - -

<a name="td.IOBase.name"></a>
#### `td.IOBase.name`




- - -

<a name="td.IOBase.output_type"></a>
#### `td.IOBase.output_type`

Returns the output type if known, else None.


- - -

<a name="td.IOBase.set_input_type"></a>
#### `td.IOBase.set_input_type(input_type)`

Updates the input type.

##### Args:


*  <b>`input_type`</b>: A type, or None.

##### Returns:

  `self`

##### Raises:


*  <b>`TypeError`</b>: If `input_type` is not compatible with the current input type
    or its expected type classes.


- - -

<a name="td.IOBase.set_input_type_classes"></a>
#### `td.IOBase.set_input_type_classes(*input_type_classes)`

Updates the type classes of the input type.

##### Args:


*  <b>`*input_type_classes`</b>: A tuple of type classes.

##### Returns:

  `self`

##### Raises:


*  <b>`TypeError`</b>: If `input_type_classes` are not compatible with the current
    input type or its expected type classes.


- - -

<a name="td.IOBase.set_io_types"></a>
#### `td.IOBase.set_io_types(other)`

Updates input and output types of two `IOBase` objects to match.

##### Args:


*  <b>`other`</b>: An instance of IOBase.

##### Returns:

  `self`

##### Raises:


*  <b>`TypeError`</b>: If the input/output types of self and other are incompatible.


- - -

<a name="td.IOBase.set_output_type"></a>
#### `td.IOBase.set_output_type(output_type)`

Updates the output type.

##### Args:


*  <b>`output_type`</b>: A type, or None.

##### Returns:

  `self`

##### Raises:


*  <b>`TypeError`</b>: If `output_type` is not compatible with the current output
    type.


- - -

<a name="td.IOBase.set_output_type_classes"></a>
#### `td.IOBase.set_output_type_classes(*output_type_classes)`

Updates the type class of the output type.

##### Args:


*  <b>`*output_type_classes`</b>: A tuple of type classes.

##### Returns:

  `self`

##### Raises:


*  <b>`TypeError`</b>: If `output_type_classes` are not compatible with the current
    output type or its expected type classes.



- - -

<a name="td.Block"></a>
### `class td.Block`

Base class for all blocks.

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
- - -

<a name="td.Block.__getitem__"></a>
#### `td.Block.__getitem__(i)`

Return a reference to the i^th output from this block.


- - -

<a name="td.Block.__init__"></a>
#### `td.Block.__init__(children=None, input_type=None, output_type=None, name=None)`




- - -

<a name="td.Block.__rrshift__"></a>
#### `td.Block.__rrshift__(lhs)`

Function composition; `(a >> b).eval(x) => b(a(x))`.


- - -

<a name="td.Block.__rshift__"></a>
#### `td.Block.__rshift__(rhs)`

Function composition; `(a >> b).eval(x) => b(a(x))`.


- - -

<a name="td.Block.eval"></a>
#### `td.Block.eval(inp, feed_dict=None, session=None, tolist=False, use_while_loop=True)`

Evaluates this block on `inp` in a TF session.

Intended for testing and interactive development. If there are any
uninitialized variables, they will be initialized prior to evaluation.

##### Args:


*  <b>`inp`</b>: An input to the block.
*  <b>`feed_dict`</b>: A dictionary that maps `Tensor` objects to feed values.
*  <b>`session`</b>: The TF session to be used. Defaults to the default session.
*  <b>`tolist`</b>: A bool; whether to return (possibly nested) Python lists
    in place of NumPy arrays.
*  <b>`use_while_loop`</b>: A bool; whether to use a `tf.while_loop` in evaluation
    (default) or to unroll the loop. Provided for testing and debugging,
    should not affect the result.

##### Returns:

  The result of running the block. If `output_type` is tensor, then a
  NumPy array (or Python list, if `tolist` is true). If a tuple, then a
  tuple. If a sequence, then a list, or an instance of itertools.repeat
  in the case of an infinite sequence. If metrics are defined then `eval`
  returns a `(result, metrics)` tuple, where `metrics` is a dict mapping
  metric names to NumPy arrays.

##### Raises:


*  <b>`ValueError`</b>: If `session` is none and no default session is registered.
    If the block contains no TF tensors or ops then a session is not
    required.


- - -

<a name="td.Block.is_forward_declaration_ref"></a>
#### `td.Block.is_forward_declaration_ref`




- - -

<a name="td.Block.max_depth"></a>
#### `td.Block.max_depth(inp)`

Returns the loom `max_depth` needed to evaluate `inp`.

Like `eval`, this is a convenience method for testing and
interactive development. It cannot be called after the TF graph
has been finalized (nb. [`Compiler.max_depth`](#td.Compiler.max_depth)
does not have this limitation).

##### Args:


*  <b>`inp`</b>: A well-formed input to this block.

##### Returns:

  An int (see above).


- - -

<a name="td.Block.reads"></a>
#### `td.Block.reads(*other)`

Sets `self` to read its inputs from `other`.

##### Args:


*  <b>`*other`</b>: which blocks to make the current block read from.

##### Returns:

  `self`

##### Raises:


*  <b>`AssertionError`</b>: if no composition scope has been entered.


- - -

<a name="td.Block.set_constructor_name_args"></a>
#### `td.Block.set_constructor_name_args(name, args, kwargs)`

Sets the constructor args used to pretty-print this layer.

Should be called by derived classes in __init__.

##### Args:


*  <b>`name`</b>: the fully qualified name of the constructor
*  <b>`args`</b>: a list of constructor arguments
*  <b>`kwargs`</b>: a list of (key,value,default) triples for keyword arguments

##### Returns:

  self



- - -

<a name="td.Layer"></a>
### `class td.Layer`

A callable that accepts and returns nests of batched of tensors.
- - -

<a name="td.Layer.__init__"></a>
#### `td.Layer.__init__(input_type=None, output_type=None, name_or_scope=None)`

Creates the layer.

##### Args:


*  <b>`input_type`</b>: A type.
*  <b>`output_type`</b>: A type.
*  <b>`name_or_scope`</b>: A string or variable scope. If a string, a new variable
    scope will be created by calling
    [`create_variable_scope`](#create_variable_scope), with defaults
    inherited from the current variable scope. If no caching device is set,
    it will be set to `lambda op: op.device`. This is because `tf.while` can
    be very inefficient if the variables it uses are not cached locally.


- - -

<a name="td.Layer.__rrshift__"></a>
#### `td.Layer.__rrshift__(lhs)`




- - -

<a name="td.Layer.__rshift__"></a>
#### `td.Layer.__rshift__(rhs)`




- - -

<a name="td.Layer.constructor_args"></a>
#### `td.Layer.constructor_args`




- - -

<a name="td.Layer.constructor_kwargs"></a>
#### `td.Layer.constructor_kwargs`




- - -

<a name="td.Layer.constructor_name"></a>
#### `td.Layer.constructor_name`





- - -

<a name="td.ResultType"></a>
### `class td.ResultType`

Base class for types that can be used as inputs/outputs to blocks.
- - -

<a name="td.ResultType.flatten"></a>
#### `td.ResultType.flatten(instance)`

Converts an instance of this type to a flat list of terminal values.


- - -

<a name="td.ResultType.for_each_terminal"></a>
#### `td.ResultType.for_each_terminal(fn, instance)`

Calls fn(terminal_type, value) for all terminal values in instance.


- - -

<a name="td.ResultType.size"></a>
#### `td.ResultType.size`

Returns the total number of scalar elements in the type.

Returns None if the size is not fixed -- e.g. for a variable-length
sequence.


- - -

<a name="td.ResultType.terminal_types"></a>
#### `td.ResultType.terminal_types()`

Returns an iterable of all terminal types in this type, in pre-order.

Void is not considered to be a terminal type since no terminal values are
needed to construct it. Instead, it has no terminal types.

##### Returns:

  An iterable with the terminal types.


- - -

<a name="td.ResultType.unflatten"></a>
#### `td.ResultType.unflatten(flat, unused_shaped_like)`

Converts a iterator over terminal values to an instance of this type.



