# TensorFlow Fold

TensorFlow Fold is a library for
creating [TensorFlow](https://www.tensorflow.org) models that consume structured
data, where the structure of the computation graph depends on the structure of
the input data. For example, [this model](tensorflow_fold/g3doc/sentiment.ipynb)
implements [TreeLSTMs](https://arxiv.org/abs/1503.00075) for sentiment analysis
on parse trees of arbitrary shape/size/depth.

Fold implements [*dynamic batching*](https://openreview.net/pdf?id=ryrGawqex).
Batches of arbitrarily shaped computation graphs are transformed to produce a
static computation graph. This graph has the same structure regardless of what
input it receives, and can be executed efficiently by TensorFlow.

* [Download and Setup](tensorflow_fold/g3doc/setup.md)
* [Documentation](tensorflow_fold/g3doc/index.md)

If you'd like to contribute to TensorFlow Fold, please review the
[contribution guidelines](CONTRIBUTING.md).
  
TensorFlow Fold is not an official Google product.
