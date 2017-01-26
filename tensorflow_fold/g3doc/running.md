# Running Blocks in TensorFlow

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Evaluating individual inputs](#evaluating-individual-inputs)
- [Batching inputs](#batching-inputs)
  - [Scenario 1: Feeding with in-memory data](#scenario-1-feeding-with-in-memory-data)
  - [Scenario 2: Feeding with streamed data](#scenario-2-feeding-with-streamed-data)
  - [Scenario 3: Reading from a tensor](#scenario-3-reading-from-a-tensor)
  - [Accelerating evaluation with multiprocessing](#accelerating-evaluation-with-multiprocessing)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

A model defined with the high-level TensorFlow Fold API is a
single [Block](py/td.md/#td.Block) object that converts an input data structure
of some kind to a tensor or tuple of tensors. There are a number of ways to
accomplish this, depending on the use-case.

## Evaluating individual inputs

The simplest way to run a block is to [`eval`](py/td.md#td.Block.eval)
individual inputs interactively from a REPL.

For example:

```python
td.OneHot(5).eval(3) => array([ 0.,  0.,  0.,  1.,  0.], dtype=float32)
```

This works for composite blocks also:

```python
(td.Scalar() >>
 td.AllOf(td.Function(tf.negative), td.Function(tf.square))).eval(2)
 => (array(-2.0, dtype=float32), array(4.0, dtype=float32))
```

Eval also works on blocks that produce sequences of indeterminate
length (i.e. varying on a per-example basis) as outputs:

```python
td.Map(td.Scalar() >> td.Function(tf.square)).eval(xrange(5))
 => [array(0.0, dtype=float32),
     array(1.0, dtype=float32),
     array(4.0, dtype=float32),
     array(9.0, dtype=float32),
     array(16.0, dtype=float32)]
```

Such blocks cannot be compiled and run on batches of inputs, because TensorFlow
does not support ragged-edged tensors. Fold does however provide
the [Metric](py/td.md#td.Metric) block, which supports batched
outputs that vary in size on a per-example basis.

## Batching inputs

Typically we want to run Fold on batches of inputs, to exploit TensorFlow
parallelism. In this case, we first create an explicit
[`Compiler`](#td.Compiler), which compiles our model down to a
TensorFlow graph. The compiler will also do type inference and
validation on the model. The outputs of the model are exposed as
ordinary TensorFlow tensors, which can be connected to TensorFlow loss
functions and optimizers in the usual way.

By default, the graph produced by the compiler uses a place-holder,
[`Compiler.loom_input_tensor`](#td.Compiler.loom_input_tensor), for
input data. This gets filled in with loom inputs (serialized
protobufs) using the TensorFlow `feed_dict` mechanism.  During
training, the compiler is also used to convert batches of input data
into `feed_dict`s, by building them into loom inputs.

The overall structure of a running model uses the following outline:

```python
  # Define the Fold model.
  root_block = insert_some_block_definition_here()

  # Compile root_block to get a TensorFlow model that we can run.
  compiler = td.Compiler(root_block)

  # Get the TensorFlow tensor(s) that correspond to the outputs of root_block.
  (model_output,) = compiler.output_tensors

  # Compute a loss of some kind using TensorFlow.
  loss = tf.l2loss(model_output)

  # Hook up a TensorFlow optimizer.
  train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

  # Convert a batch of examples into a TensorFlow feed_dict.
  fd = compiler.build_feed_dict(batch_of_examples)

  # Now train the model on the batch of examples.
  sess.run(train_op, feed_dict=fd)
```

### Scenario 1: Feeding with in-memory data

If your dataset is small enough to fit in memory, we recommend building
loom inputs once and keeping them around. A simple Fold
training loop for small datasets is as follows:

```python
  train_set = compiler.build_loom_inputs(load_examples(train_file))
  train_set_dict = {}  # add feeds for e.g. dropout here
  dev_feed_dict = compiler.build_feed_dict(load_examples(dev_file))
  for epoch, shuffled in enumerate(td.epochs(train_set, num_epochs), 1):
    train_loss = 0.0
    for batch in td.group_by_batches(shuffled, batch_size):
      train_feed_dict[compiler.loom_input_tensor] = batch
      _, batch_loss = sess.run([train, loss], train_feed_dict)
      train_loss += batch_loss
    dev_loss = sess.run(loss, dev_feed_dict)
    print 'epoch: %s train: %s dev: %s' % (epoch, train_loss, dev_loss)
```

The code assumes that you have already created a compiler for your
model, have `train` and `loss` tensors, and have a `load_examples` function.

This idiom exploits Python generators to do work as needed and avoid blocking.

* [`Compiler.build_loom_inputs`](#td.Compiler.build_loom_inputs) takes
  an iterable of examples and returns an iterable of loom inputs in
  one-to-one correspondence.

* [`Compiler.build_feed_dict`](#td.Compiler.build_feed_dict) takes an
  iterable of examples and returns a lazily computed feed
  dictionary. The feed values are cached internally, so they only get
  computed once.

* [`td.epochs`](#td.epochs) is a simple utility function that
  repeatedly yields an iterable, caching and shuffling it after the
  first yield. The reason for using `build_loom_inputs` for our
  training set is so that we can shuffle after each epoch and get
  different batches on each pass.

* [`td.group_by_batches`](#td.group_by_batches) is an even simpler
  utility that lazily batches an iterable into lists.

What all this means is that TensorFlow training begins as soon as the
first batch of examples has been built into loom inputs. To take full
advantage of this, `load_examples` should ideally return a generator.

### Scenario 2: Feeding with streamed data

If your dataset is too large to fit in memory, or if you only want to
process each example once (e.g. for inference), it is cleaner and more
efficient to use
[`Compiler.build_loom_input_batched`](#td.Compiler.build_loom_input_batched),
like so:

```python
for batch_feed in compiler.build_loom_input_batched(examples, batch_size):
  sess.run(fetches, {compiler.loom_input_tensor: batch_feed, ...}
  ...
```

As you'd expect, `build_loom_input_batched` is a generator, yielding
batch feed values one at a time. So if `examples` is also a generator,
only `batch_size` examples will get materialized in-memory at a time.

### Scenario 3: Reading from a tensor

Fold also supports reading from a tensor following [the standard TF
idiom](https://www.TensorFlow.org/versions/r0.10/how_tos/reading_data/index.html#reading-from-files).
For example, let's say we have a tensor `loom_input_batch` containing batches of
precomputed loom inputs (e.g. queued up from files):

```python
compiler = td.Compiler(root_block, init_loom=False)
compiler.init_loom(loom_input_tensor=loom_input_batch)
...
```

We can now use `compiler.output_tensors` without needing to feed any values.

If the input to your model is a string, then it can be built directly
(without separately computing loom inputs) like so:

```python
compiler = td.Compiler(root_block, init_loom=False)
compiler.init_loom(input_tensor=model_input_batch)
```

For example:

```python
x = tf.constant(['foo', 'bar', 'foobar'])
compiler = td.Compiler(td.Length(), init_loom=False)
compiler.init_loom(input_tensor=x)
sess.run(compiler.output_tensors) => [array([ 3.,  3.,  6.], dtype=float32)]
```

### Accelerating evaluation with multiprocessing

TensorFlow exploits multicore to speed up operations like matrix
multiplication which are implemented in C++. Fold evaluation is
currently implemented in Python, so we run into the notorious [global
interpreter
lock](https://wiki.python.org/moin/GlobalInterpreterLock). We can
overcome this using a pool of subprocesses, which can be easily
created using the
[`Compiler.multiprocessing_pool`](py/td.md#td.Compiler.multiprocessing_pool)
context manager. Inside of a multiprocessing pool context, all calls
to the compiler that build loom inputs will seamlessly use the pool to
evaluate examples in parallel. By default results will be unordered,
which is faster. For example:

```python
with compiler.multiprocessing_pool():
  batch_feeds = compiler.build_loom_input_batched(examples, batch_size):
  sess.run(fetches, {compiler.loom_input_tensor: next(batch_feeds)})
```

Here `Session.run` will be called on the first batch of examples to
get computed, which will not necessarily be the first `batch_size`
elements of the `examples` iterable. To preserve order, simply pass
`ordered=True` to `build_loom_input_batched` (or `build_feed_dict`, or
`build_loom_inputs`).

The implementation uses the standard [Python
mulitprocessing](https://docs.python.org/2/library/multiprocessing.html)
library. When the context manager exits the pool is closed and we
block until all ongoing work is completed.
