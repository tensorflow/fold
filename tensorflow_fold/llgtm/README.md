### Introduction

LLGTM *(The Low-level Library for Gradients, Tensors, and Matrices)* is a C++
library for deep learning models that use dynamic computation graphs.

LLGTM is intended to be an alternative to the Loom library.  Loom is written in
python, and implements dynamic computation graphs by emulating such graphs on
top of TensorFlow.  The advantage of Loom is that it integrates cleanly with the
rest of tensorflow, but the disadvantage is that the python interpreter becomes
part of the main evaluation loop. Since python is a single-threaded, interpreted
language, it can become a significant bottleneck.

LLGTM has two major design goals. First, it makes graph construction and
differentiation very fast. Graphs are light-weight, and allocated in an arena,
so they can be created and destroyed quickly.

Second, LLGTM separates graph construction and differentiation from graph
evaluation. In other words, it supports multiple evaluation backends.  The
initial release supports two backends: a reference implementation that uses
[Eigen](http://eigen.tuxfamily.org), and a TensorFlow backend that invokes
TensorFlow kernels.  Additional backends may be provided in the future, with no
change to the user-facing API.

### Known Issues

LLGTM is currently in *pre-alpha*. Simple examples compile and run, but but many
features are still missing and/or incomplete. LLGTM is being released as open
source in order to solicit feedback and contributions from users of TensorFlow
Fold.  However, we ***strongly*** suggest that users continue to use the Loom
library for the time being.

Known issues include the following:

* Very few operations are supported.
* No way to save or restore models.
* Lots of other missing features.
* The TensorFlow backend relies on non-public APIs, and thus does not compile without visibility chainges to the TensorFlow BUILD file.
* The Eigen backend includes several kernels that have not been optimized. Do not expect performance to be competitive with state-of-the art models.

### Usage

To build and test LLGTM, run the following from the *llgtm* directory:

```
bazel test :cpu_tests
bazel test --config=cuda :gpu_tests
source examples/run_all_examples.sh
```
