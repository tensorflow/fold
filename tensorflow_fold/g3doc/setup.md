# Download and Setup

Fold runs under Linux; we have not tested it on other platforms. Python 2.7 and
3.3+ are both supported. We recommend installing
using [Virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/)
and [pip](https://pip.pypa.io/en/stable/). For Python 3.3+, say `pip3` instead
of `pip` below. See [here](sources.md) for instructions on installing from
sources, if that's how you roll. If you run into trouble, the TensorFlow main
site has a list
of
[common problems](https://www.tensorflow.org/versions/r1.0/get_started/os_setup#common_problems) with
some solutions that might be helpful.

Please note that Fold requires TensorFlow 1.0; it is not compatible with earlier
versions due to breaking API changes.

First install Python, pip, and Virtualenv:

```
sudo apt-get install python-pip python-dev python-virtualenv
```

Create a Virtualenv environment in the directory `foo`:

```
virtualenv foo             # for Python 2.7
virtualenv -p python3 foo  # for Python 3.3+
```

Activate the environment:

```
source ~/tensorflow/bin/activate      # if using bash
source ~/tensorflow/bin/activate.csh  # if using csh
```

Install the pip for TensorFlow. For Python 2.7 CPU-only, this will be:

```
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.0.0rc0-cp27-none-linux_x86_64.whl
```

For Python 3.3+ and/or GPU,
see
[here](https://www.tensorflow.org/versions/r1.0/get_started/os_setup#using_pip)
for the full list of available TF binaries. 

Install the pip for Fold.  For Python 2.7, this will be:

```
pip install https://storage.googleapis.com/tensorflow_fold//tensorflow_fold-0.0.1-cp27-none-linux_x86_64.whl
```

For Python 3.3:

```
pip3 install https://storage.googleapis.com/tensorflow_fold//tensorflow_fold-0.0.1-py3-none-linux_x86_64.whl
```

## Next steps

* try out the [quick start notebook](quick.ipynb)
* browse the [documentation](index.md)
* hacks and glory
