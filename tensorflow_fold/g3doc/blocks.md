# Blocks Tutorial

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Introduction](#introduction)
  - [Motivating example](#motivating-example)
- [Basic concepts](#basic-concepts)
- [Primitive blocks](#primitive-blocks)
  - [Converting Python objects into tensors.](#converting-python-objects-into-tensors)
  - [Using TensorFlow tensors.](#using-tensorflow-tensors)
  - [Functions and Layers](#functions-and-layers)
  - [Python operations](#python-operations)
- [Block composition](#block-composition)
  - [Wiring blocks together](#wiring-blocks-together)
  - [Dealing with sequences](#dealing-with-sequences)
  - [Dealing with records](#dealing-with-records)
  - [Wiring things together, in more complicated ways.](#wiring-things-together-in-more-complicated-ways)
  - [Recursion and forward declarations](#recursion-and-forward-declarations)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Introduction

We assume the reader understands the basic concepts of TensorFlow and deep
learning.  If not, the [TensorFlow
tutorials](https://www.tensorflow.org/tutorials/)
are a good place to start.

The input to a Fold model is a mini-batch of Python objects. These
objects may be produced by deserializing a protocol buffer, JSON, XML, or a
custom parser of some kind.  The input objects are assumed to be
tree-structured.  The output of a Fold model is a set of TensorFlow tensors,
which can be hooked up to a loss function and optimizers in the usual way.

Given a mini-batch of data structures as input, Fold will take care of
traversing the input data, and combining and scheduling operations in a way
that can be executed efficiently by TensorFlow.  For example, if each node
in a tree outputs a vector using a fully-connected layer with shared
weights, then Fold will not simply traverse the tree and do a bunch of
vector-matrix multiply operations.  Instead, it will merge nodes at the same
depth in the tree that can be executed in parallel into larger and more
efficient matrix-matrix multiply operations, and then split up the output
matrix into vectors again.

### Motivating example

The following example code implements a hierarchical LSTM, something that is
easy to do in Fold, but hard to do in TensorFlow:

```python
# Create RNN cells using the TensorFlow RNN library
char_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=16), 'char_cell')
word_cell = td.ScopedLayer(tf.contrib.rnn.BasicLSTMCell(num_units=32), 'word_cell')

# character LSTM converts a string to a word vector
char_lstm = (td.InputTransform(lambda s: [ord(c) for c in s]) >>
             td.Map(td.Scalar('int32') >>
                    td.Function(td.Embedding(128, 8))) >>
             td.RNN(char_cell))
# word LSTM converts a sequence of word vectors to a sentence vector.
word_lstm = td.Map(char_lstm >> td.GetItem(1) >> td.GetItem(1)) >> td.RNN(word_cell)
```

A hierarchical LSTM takes a list of strings as input, where each string is a
word, and produces a sentence vector as output.  It does this using two nested
LSTMs.  A character LSTM takes a string as input, and produces a word vector as
output.  It converts the string to a list of integers,
([`td.InputTransform`](py/td.md#td.InputTransform)), looks up each integer in an
embedding table ([`td.Embedding`](py/td.md#td.Embedding)), and processes the
sequence of embeddings with an LSTM to yield a word vector.  The word LSTM maps
([`td.Map`](py/td.md#td.Map)) the character LSTM over a sequence of words to get
a sequence of word vectors, and processes the word vectors with a second, larger
LSTM to yield a sentence vector.

The following sections explain these operations in more detail.

## Basic concepts

The basic component of a Fold model is the [`td.Block`](py/td.md#td.Block).  A
block is essentially a function -- it takes an object as input, and produces
another object as output.  The objects in question may be tensors, but they may
also be tuples, lists, Python dictionaries, or combinations thereof.
The [types](types.md) page describes the Fold type system in more detail.

Blocks are organized hierarchically into a tree, much like expressions in a
programming language, where larger and more complex blocks are composed from
smaller, simpler blocks.  Note that the block structure must be a tree, not
a DAG.  In other words, each block (i.e. each instance of one of the block
classes below) must have a unique position in the tree.  The type-checking
and [compilation steps](running.md) depend on tree property.

## Primitive blocks

Primitive blocks form the leaves of the block hierarchy, and are responsible for
basic computations on tensors.

### Converting Python objects into tensors.

A [`td.Scalar()`](py/td.md#td.Scalar) block converts a Python scalar to a
0-rank tensor.

A [`td.Vector(shape)`](py/td.md#td.Vector) block converts a Python list
into a tensor of the given shape.

### Using TensorFlow tensors.

A [`td.FromTensor`](py/td.md#td.FromTensor) block will wrap a TensorFlow tensor
into a block.  Like a function with no arguments, a `FromTensor` block does not
accept any input; it just produces the corresponding tensor as output.
`FromTensor` can also be used with numpy tensors, e.g.

```python
  td.FromTensor(np.zeros([]))  # a constant with dtype=float64
  td.FromTensor(tf.zeros([]))  # a constant with dtype=float32
  td.FromTensor(tf.Variable(tf.zeros([])))  # a trainable variable
```

<a name="functions"></a>
### Functions and Layers

A [`td.Function`](py/td.md#td.Function) block wraps a TensorFlow operation into
a block.  It accepts tensor(s) as input, and produces tensor(s) as output.
Functions which take multiple arguments, or produce multiple results, pass
tuples of tensors as input/output.  For example, `td.Function(tf.add)` is a
block that takes a tuple of two tensors as input, and produces a single tensor
(the sum) as output.

Function blocks can be used in conjuction with *Layers* to perform typical
neural network computations, such as fully-connected layers and embeddings.
A [`td.Layer`](py/td.md#td.Layer) is a callable Python object that implements
weight sharing between different instances of the layer.

In the example below, `ffnet` is a three-layer feed forward-network, where
each of the three layers shares weights.

```python
# fclayer defines the weights for a fully-connected layer with 1024 hidden units.
fclayer = td.FC(1024)
# Each call to Function(fclayer) creates a fully-connected layer,
# all of which share the weights provided by the fclayer object.
ffnet = td.Function(fclayer) >>  td.Function(fclayer) >> td.Function(fclayer)
```

The [`>>`](py/td.md#td.Block.__rshift__) operator
denotes [function composition](#composition).

### Python operations

The [`td.InputTransform`](py/td.md#td.InputTransform) block wraps an arbitrary
Python function into a block.  It takes a Python object as input, and returns a
Python object as output.  For example, the following block converts a Python
string to a list of floats.

```python
td.InputTransform(lambda s: [ord(c)/255.0 for c in s])
```

As its name suggests, InputTransform is primarily used to preprocess the
input data in Python before passing it to TensorFlow.  Once data has reached
the TensorFlow portion of the pipeline, it is no longer possible to run
arbitrary Python code on the data, and Fold will produce a type error
on any such attempt.

## Block composition

Blocks can be composed with other blocks in various ways to create blocks
with more complex behavior.

<a name="composition"></a>
### Wiring blocks together

The simplest form of composition is to wire the output of one block to the
input of another, using the `>>` operator.  The syntax `f >> g` denotes
function composition.  It creates a new block that feeds its input to `f`,
passes the output of `f` into `g`, and returns the output of `g`.

For example, the following block is part of an MNIST model, where the MNIST
images are stored serialized as strings of length 784.  It converts each
string into a list of floats, converts the list into a tensor, and runs the
tensor through two fully connected layers.

```python
mnist_model = (td.InputTransform(lambda s: [ord(c) / 255.0 for c in s]) >>
               td.Vector(784) >>             # convert python list to tensor
               td.Function(td.FC(100)) >>    # layer 1, 100 hidden units
               td.Function(td.FC(100)))      # layer 2, 100 hidden units
```

<a name="sequences"></a>
### Dealing with sequences

Fold processes sequences of data using blocks which are analagous to
higher-order functions like
[*map*](https://en.wikipedia.org/wiki/Map_(higher-order_function)) and
[*fold*](https://en.wikipedia.org/wiki/Fold_(higher-order_function)).
Sequences may be of arbitrary length, and may vary in length from example to
example; there is no need to truncate or pad the sequence out to a
pre-defined length.

* [`td.Map(f)`](py/td.md#td.Map): Takes a sequence as input, applies block
  `f` to every element in the sequence, and produces a sequence as output.

* [`td.Fold(f, z)`](py/td.md#td.Fold): Takes a sequence as input, and
  performs a
  left-[fold](https://en.wikipedia.org/wiki/Fold_(higher-order_function)), using
  the output of block `z` as the initial element.

* [`td.RNN(c)`](py/td.md#td.RNN): A recurrent neural network, which is a
  combination of Map and Fold.  Takes an initial state and input sequence, uses
  the rnn-cell `c` to produce new states and outputs from previous states and
  inputs, and returns a final state and output sequence.

* [`td.Reduce(f)`](py/td.md#td.Reduce): Takes a sequence as input, and
  reduces it to a single value by applying `f` to elements pair-wise,
  essentially executing a binary expression tree with `f`.

* [`td.Zip()`](py/td.md#td.Zip): Takes a tuple of sequences as inputs, and
  produces a sequence of tuples as output.

* [`td.Broadcast(a)`](py/td.md#td.Broadcast): Takes the output of block `a`,
  and turns it into an infinite repeating sequence.  Typically used in
  conjunction with Zip and Map, to process each element of a sequence with a
  function that uses `a`.

Examples:

```python
# Convert a python list of scalars to a sequence of tensors, and take the
# absolute value of each one.
abs = td.Map(td.Scalar() >> td.Function(tf.abs))

# Compute the sum of a sequence, processing elements in order.
sum = td.Fold(td.Function(tf.add), td.FromTensor(tf.zeros(shape)))

# Compute the sum of a sequence, in parallel.
sum = td.Reduce(td.Function(tf.add))

# Convert a string to a vector with a character RNN, using Map and Fold
char_rnn = (td.InputTransform(lambda s: [ord(c) for c in s]) >>
            # Embed each character using an embedding table of size 128 x 16
            td.Map(td.Scalar('int32') >>
                   td.Function(td.Embedding(128, 16))) >>
            # Fold over the sequence of embedded characters,
            # producing an output vector of length 64.
            td.Fold(td.Concat() >> td.Function(td.FC(64)),
                    td.FromTensor(tf.zeros(64))))
```

The Fold and RNN blocks can be used to apply sequence models like
[LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)s to lists of
data.  One caveat is that backpropogation over very long sequences still
suffers from the vanishing gradient problem, so for long sequences (more
than 100 elements), using `Reduce` may be preferable to `Fold`, since the
depth of `Reduce` is the log of the sequence length.

Moreover, TensorFlow itself may run out of memory while processing long
sequences, because all intermediate results must be kept in memory to
compute the gradients.  Thus, even though Fold does not impose any limit
on sequence length, it may still be desirable to truncate long sequences
to a manageable length using `td.InputTransform`.

### Dealing with records

A record is a set of named fields, each of which may have a different type, such
as a Python dictionary, or protobuf message.
A [`td.Record`](py/td.md#td.Record) block takes a record as input, applies a
child block to each field, and combines the results into a tuple, which it
produces as output.  The output tuple can be passed
to [`td.Concat()`](py/td.md#td.Concat) to get an output vector.

For example, the following block converts a record of three fields into a
128-dimensional vector.  It converts the `id` field to a vector using an
embedding table, runs the `name` field through a character RNN [as defined
above](#sequences), treats the `location` field as-is, then concatenates all
three results together and passes them through a fully-connected layer.

```python
# Takes as input records of the form:
# {'id':       some_id_number,
#  'name':     some_string,
#  'location': (x,y)
# }
rec = (td.Record([('id', td.Scalar('int32') >>
                         td.Function(td.Embedding(num_ids, embed_len))),
                  ('name', char_rnn),
                  ('location', td.Vector(2))]) >>
       td.Concat() >> td.Function(td.FC(128)))
```

The fully-connected layer has 128 hidden units, and thus outputs a vector of
length 128.  It will infer the size of its input (embed_len + 64 + 2) when
`rec` is compiled.

### Wiring things together, in more complicated ways.

Simple function composition using the `>>` operator resembles standard Unix
[pipes](https://en.wikipedia.org/wiki/Pipeline_(Unix)), and is usually
sufficient in most cases, especially when combined with blocks like `Record`
and `Fold` that traverse the input data structure.

However, some models may need to wire things together in more complicated ways.
A [`td.Composition`](py/td.md#td.Composition) block allows the inputs and
outputs of its children to be wired together in an arbitrary DAG.  For example,
the following code defines an LSTM cell as a block, which is suitable for use
with the RNN block [mentioned above](#sequences). Every block defined within the
`lstm_cell.scope()` becomes a child of `lstm_cell`. The `b.reads(...)` method
wires the output of another block or tuple of blocks to the input of `b`, and
roughly corresponds to function application.  If the output of `b` is a tuple,
then the `b[i]` syntax can be used to select individual elements of the tuple.

```python
# The input to lstm_cell is (input_vec, (previous_cell_state, previous_output_vec))
# The output of lstm_cell is (output_vec, (next_cell_state, output_vec))
lstm_cell = td.Composition()
with lstm_cell.scope():
  in_state = td.Identity().reads(lstm_cell.input[1])
  bx = td.Concat().reads(lstm_cell.input[0], in_state[1])     # inputs to gates
  bi = td.Function(td.FC(num_hidden, tf.nn.sigmoid)).reads(bx)  # input gate
  bf = td.Function(td.FC(num_hidden, tf.nn.sigmoid)).reads(bx)  # forget gate
  bo = td.Function(td.FC(num_hidden, tf.nn.sigmoid)).reads(bx)  # output gate
  bg = td.Function(td.FC(num_hidden, tf.nn.tanh)).reads(bx)     # modulation
  bc = td.Function(lambda c,i,f,g: c*f + i*g).reads(in_state[0], bi, bf, bg)
  by = td.Function(lambda c,o: tf.tanh(c) * o).reads(bc, bo)    # final output
  out_state = td.Identity().reads(bc, by)   # make a tuple of (bc, by)
  lstm_cell.output.reads(by, out_state)
```

Note that we provide this definition for the illustrative purposes only.
Since the body of the cell consists only of `Function` blocks that wrap the
corresponding TensorFlow operations, it would be both cleaner and more
efficient to implement LSTM cells directly as a simple `Layer`, rather than a
`Composition`.


### Recursion and forward declarations

Implementing a tree-recursive neural network requires a recursive block
definition.  The type of the block is first declared with
a [`td.ForwardDeclaration`](py/td.md#td.ForwardDeclaration).  Then the block
itself is defined as usual using the forward declaration to create recursive
references.  Finally, a call
to
[`td.ForwardDeclaration.resolve_to`](py/td.md#td.ForwardDeclaration.resolve_to)
will tie the recursive definition to its forward declaration.

For example, here is a block that uses TensorFlow to evaluate arithmetic
expressions:

```python
# the expr block processes objects of the form:
# expr_type ::=  {'op': 'lit', 'val': <float>}
#             |  {'op': 'add', 'left': <expr_type>, 'right': <expr_type>}
expr_fwd = td.ForwardDeclaration(pvt.PyObjectType(), pvt.Scalar())
lit_case = td.GetItem('val') >> td.Scalar()
add_case = (td.Record({'left': expr_fwd(), 'right': expr_fwd()}) >>
            td.Function(tf.add))
expr = td.OneOf(lambda x: x['op'], {'lit': lit_case, 'add': add_case})
expr_fwd.resolve_to(expr)
```

Notice that `expr_fwd` is a declaration, not a block.  Each call to
`expr_fwd()` creates a block that references the declaration.
