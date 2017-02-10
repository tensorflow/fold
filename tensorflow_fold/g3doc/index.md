# TensorFlow Fold: Deep Learning with Dynamic Computation Graphs

TensorFlow Fold is a library for creating TensorFlow models that consume
structured data, such as nested lists, dictionaries,
and
[protocol buffers](https://developers.google.com/protocol-buffers/). Examples of
such models
are
[tree-recursive neural networks](https://en.wikipedia.org/wiki/Recursive_neural_network)
such as models of the
[Stanford sentiment treebank](http://nlp.stanford.edu/sentiment/index.html),
[tree LSTMs](https://arxiv.org/pdf/1503.00075.pdf),
[hierarchical LSTMs](https://arxiv.org/pdf/1506.01057v2.pdf), and
[graph-convolutional neural networks](https://arxiv.org/pdf/1603.00856v3.pdf).

TensorFlow by itself was not designed to work with tree or graph structured
data.  It does not natively support any data types other than tensors, nor does
it support the complex control flow, such as recursive functions, that are
typically used to run models like tree-RNNs.  When the input consists of trees
(e.g. parse trees from a natural language model), each tree may have a different
size and shape.  A standard TensorFlow model consists of a fixed graph of
operations, which cannot accommodate variable-shaped data.  Fold overcomes this
limitation by using
the [dynamic batching algorithm](https://arxiv.org/abs/1702.02181).

Fold consists of a high-level API called Blocks, and a low-level API called
Loom. Blocks are pure Python, whereas Loom is a mixture of Python and
C++. Internally, Blocks uses Loom as its execution engine. Loom is an
abstraction layer on top of TensorFlow that makes it possible to easily express
computations over structures of varying sizes and shapes without the need to
modify the underlying computation graph at run-time.

## Quick Links

* [Blocks Tutorial](blocks.md)
* [Running Blocks in TensorFlow](running.md)
* [Blocks Type System](types.md)
* [Python Blocks API](py/td.md)
* [Python Loom API](py/loom.md)
* [C++ Weaver API](cc/index.md)
* [Protocol Buffer Decoding](proto.md)
