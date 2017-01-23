# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""TensorFlow Fold for MNIST with fully connected layers and dropout.

With default settings the test accuracy after 20 epochs is ~ 98.4%.

Build:
  bazel build --config=opt \
    //tensorflow_fold/blocks/examples/mnist

Train:
  ./bazel-bin/tensorflow_fold/blocks/examples/mnist/mnist

Eval:
  ./bazel-bin/tensorflow_fold/blocks/examples/mnist/mnist \
    --mode=eval --eval_interval_secs=10  # set to 0 to evaluate once and exit

Inference:
  ./bazel-bin/tensorflow_fold/blocks/examples/mnist/mnist \
    --mode=infer

See below and <tensorflow_fold/blocks/plan.py>
for additional flag options.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_fold.public.blocks as td


NUM_LABELS = 10
INPUT_LENGTH = 784  # 28 x 28

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('num_layers', 2, 'Number of hidden layers.')
flags.DEFINE_integer('num_units', 500, 'Number of units per hidden layer.')
flags.DEFINE_float('keep_prob', 0.75, 'Keep probability for dropout.')
td.define_plan_flags(default_plan_name='mnist')


def setup_plan(plan):
  """Sets up a TensorFlow Fold plan for MNIST.

  The inputs are 28 x 28 images represented as 784-dimensional float32
  vectors (scaled to [0, 1] and categorical digit labels in [0, 9].

  The training loss is softmax cross-entropy. There is only one
  metric, accuracy. In inference mode, the output is a class label.

  Dropout is applied before every layer (including on the inputs).

  Args:
    plan: A TensorFlow Fold plan to set up.
  """
  # Convert the input NumPy array into a tensor.
  model_block = td.Vector(INPUT_LENGTH)

  # Create a placeholder for dropout, if we are in train mode.
  keep_prob = (tf.placeholder_with_default(1.0, [], name='keep_prob')
               if plan.mode == plan.mode_keys.TRAIN else None)

  # Add the fully connected hidden layers.
  for _ in xrange(FLAGS.num_layers):
    model_block >>= td.FC(FLAGS.num_units, input_keep_prob=keep_prob)

  # Add the linear output layer.
  model_block >>= td.FC(NUM_LABELS, activation=None, input_keep_prob=keep_prob)

  if plan.mode == plan.mode_keys.INFER:
    # In inference mode, we run the model directly on images.
    plan.compiler = td.Compiler.create(model_block)
    logits, = plan.compiler.output_tensors
  else:
    # In training/eval mode, we run the model on (image, label) pairs.
    plan.compiler = td.Compiler.create(
        td.Record((model_block, td.Scalar(tf.int64))))
    logits, y_ = plan.compiler.output_tensors

  y = tf.argmax(logits, 1)  # create the predicted output tensor

  datasets = tf.contrib.learn.datasets.mnist.load_mnist(FLAGS.logdir_base)
  if plan.mode == plan.mode_keys.INFER:
    plan.examples = datasets.test.images
    plan.outputs = [y]
  else:
    # Create loss and accuracy tensors, and add them to the plan.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=y_)
    plan.losses['cross_entropy'] = loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_), tf.float32))
    plan.metrics['accuracy'] = accuracy
    if plan.mode == plan.mode_keys.TRAIN:
      plan.examples = zip(datasets.train.images, datasets.train.labels)
      plan.dev_examples = zip(datasets.validation.images,
                              datasets.validation.labels)
      # Turn dropout on for training, off for validation.
      plan.train_feeds[keep_prob] = FLAGS.keep_prob
    else:
      assert plan.mode == plan.mode_keys.EVAL
      plan.examples = zip(datasets.test.images, datasets.test.labels)


def main(_):
  assert 0 < FLAGS.keep_prob <= 1, '--keep_prob must be in (0, 1]'
  td.Plan.create_from_flags(setup_plan).run()


if __name__ == '__main__':
  tf.app.run()
