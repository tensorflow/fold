# Download and Setup

Fold must be installed from the github source; we do not currently provide
pre-built binaries.

## Requirements

Fold runs under Linux; we have not tested it on other platforms. Python 2.7 and
3.3+ are both supported.

TensorFlow sources and binaries are required; we strongly
recommend
[installing from sources](https://www.tensorflow.org/get_started/os_setup#installing_from_sources) to ensure consistency.

The [sentiment model](tensorflow_fold/g3doc/sentiment.ipynb) example
requires [NLTK](http://www.nltk.org/); `sudo pip install nltk` or
`sudo pip3 install nltk`.

Make sure that TensorFlow and all of its prerequisites are installed and that
you have run `./configure` from the root of the TensorFlow source tree. Fold
inherits its configuration options, such as the location of Python and which
optimization flags to use during compilation, from TensorFlow.

## Download

```
git clone https://github.com/tensorflow/fold
cd fold
```

Fold requires a symlink from the TensorFlow source tree. Assuming both source
trees are in the same parent directory:

```
ln -s ../tensorflow/ .
```

## Test (optional)

To run the unit tests, do:

```
bazel test --config=opt tensorflow_fold/...
```

There is also a smoke test that runs all of the included examples:

```
./tensorflow_fold/run_all_examples.sh --config=opt
```

## Build a pip wheel

```
bazel build --config=opt //tensorflow_fold/util:build_pip_package
./bazel-bin/tensorflow_fold/util/build_pip_package /tmp/tensorflow_fold_pkg
```

## Install

The precise name of the .whl file will depend on your platform.

For Python 2.7:

```
sudo pip install /tmp/tensorflow_fold_pkg/tensorflow_fold-0.0.1-PLATFORM.whl
```

For Python 3.3+:

```
sudo pip3 install /tmp/tensorflow_fold_pkg/tensorflow_fold-0.0.1-PLATFORM.whl
```

## Next steps

* browse the [documentation](index.md)
* hacks and glory
