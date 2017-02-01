# Blocks Type System

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->


- [Overview](#overview)
- [Type inference](#type-inference)
- [Type canoncialization](#type-canoncialization)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Overview

Blocks have associated input and output types. For example:

```python
td.Scalar().input_type => PyObjectType()
td.Scalar().output_type => TensorType((), 'float32')
```

Types are instances of the abstract [`ResultType`](py/td.md#td.ResultType)
class, which has the following concrete subclasses:

1. [`VoidType()`](py/td.md#td.VoidType): No input (or no output).

2. [`PyObjectType()`](py/td.md#td.PyObjectType): an arbitrary Python object.

3. [`TensorType(shape, dtype='float32')`](py/td.md#td.TensorType): tensors of a
particular shape and dtype. For scalars, the shape is the empty tuple.

4. [`TupleType(T`<sub>`1`</sub>`, ..., T`<sub>`N`</sub>`)`
](py/td.md#td.TupleType):
tuples of `N` items.  For `1 <= i <= N`, item `i` has type `T`<sub>`i`</sub>.

5. [`SequenceType(T)`](py/td.md#td.SequenceType): variable-length sequences all
of whose elements are the same type (`T`).

6. [`BroadcastSequenceType(T)`](py/td.md#td.BroadcastSequenceType): an infinite
sequence (i.e. generator) of type `T`.

Types are immutable and well-ordered.

For tensor and tuple types, there is also a short-hand representation. Lists
and integers denote tensor types, and tuples denote tuple types.  The
`dtype` of a tensor can be specified as the last element of a list, and
defaults to `'float32'`.

1. `[16, 16]` = `[16, 16, 'float32']` = `TensorType((16, 16), 'float32')`

2. `42` = `[42, 'float32']` = `TensorType((42,), 'float32')`

3. `['bool']` = `TensorType((), 'bool')`

4. `([], [3, 3, 'int64'])` = `(['float32'], [3, 3, 'int64'])`

The [`convert_to_type`](py/td.md#td.convert_to_type) function converts a
short-hand to a type; API functions that take types as their arguments also
accept short-hands.

## Type inference

Some blocks, like [`td.Scalar()`](py/td.md#td.Scalar), have predetermined
input and output types. This is not always the case. For example:

``` python
td.Length().input_type => None
td.Length().output_type => TensorType((), 'float32')
```

The length block always returns a scalar, but its input type can be a
`PyObjectType`, a `TupleType`, or a `SequenceType`. In most cases, we can infer
the output type of a block from its input type. For example:

``` python
length_block = td.Length()
length_block.input_type => None
seq_len = td.Map(td.Scalar()) >> length_block
length_block.input_type => SequenceType(TensorType((), 'float32'))
```

It is also possible to explicitly set the input / output type of a block:

```python
length_block = td.Length()
length_block.set_input_type(SequenceType(TensorType((), 'float32')))
length_block.input_type => SequenceType(TensorType((), 'float32'))
```

If you attempt to set the input / output type of a block to a type that could
never be valid, you will get an error right away:

``` python
td.Length().set_input_type(td.VoidType())  =>
TypeError: bad input type VoidType for <td.Length dtype='float32'>, expected PyObjectType or SequenceType or TupleType
```

It is an error explicitly set the input / output type of a block to a type that
is valid and then later compose it with a block that implies some other input /
output type:

``` python
length_block = td.Length()
length_block.set_input_type(td.PyObjectType())
td.Map(td.Scalar()) >> length_block  =>
TypeError: Type mismatch between input type SequenceType(TensorType((), 'float32')) and expected input type PyObjectType() in <td.Length dtype='float32'>.
```

The [`Function`](py/td.md#Function) block is a special case. Its input and
output types must be nests of tensors, where a nest of `T`s is defined
recursively as either a `T` or a tuple of nests of `T`s. When a function block
is used in a composition, we can infer its input type as usual:

``` python
f = td.Function(tf.add)
g = (td.Scalar(), td.Scalar()) >> f
f.input_type => TupleType(TensorType((), 'float32'), TensorType((), 'float32'))
```

But in order to obtain the output type, we need to actually call the function
that we are wrapping (`tf.add`, in this case). These calls are made with
placeholders in a special `tensorflow_fold_output_type_inference` name
scope. Because of TensorFlow's execution semantics, the subgraphs thus created
are never run, but they are still there. If you don't want them, simply pass
`infer_output_type=False` when you construct the block, and call
`set_output_type` explicitly on your Function block immediately after it is
constructed.

## Type canoncialization

The `PyObjectType` complicates things, slightly, since a sequence or tuple of
`PyObjectType`s can also be a `PyObjectType`. We handle this by applying a
simple reduction rule; tuples and sequences of `PyObjectType`s are recursively
converted to terminal `PyObjectType`s. For example:

``` python
td.Scalar().output_type => TensorType((), 'float32')
td.Map(td.Scalar()).output_type => SequenceType(TensorType((), 'float32'))
# vs.
td.InputTransform(str.lower).output_type => PyObjectType()
td.Map(td.InputTransform(str.lower)).output_type => PyObjectType()
```

Type canonicalization is performed by
the [`td.canonicalize_type`](py/td.md#td.canonicalize_type) function. Note that
it should not be necessary to ever call this function explicitly unless you are
writing your own type manipulation routines, since all types accepted and
returned by Fold functions get canonicalized internally.
