<!-- This file is machine generated: DO NOT EDIT! -->

# TensorFlow Fold C++ Weaver API

## Weaver

The Weaver API allows the user to build a schedule to be run by a Loom,
producing vectors of integers (wiring diagrams) to be fed to Loom's `tf.gather`
ops in order to drive the loom according to the schedule the user specified.

* [tensorflow::fold::Weaver](ClassWeaver.md)

## WeaverOpBase

WeaverOpBase contains the common code required to encapsulate a function
that uses a weaver to build a graph for a Loom to run inside of a TensorFlow
operation capable of driving the aforementioned Loom.

* [tensorflow::fold::WeaverOpBase](ClassWeaverOpBase.md)


