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
"""Fizzbuzz machine learning example for TensorFlow Fold.

Inspiration from http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random
# import google3
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import tensorflow_fold.public.blocks as td


# Data generation flags:
tf.app.flags.DEFINE_integer(
    'seed', 42, 'Random seed.')
tf.app.flags.DEFINE_integer(
    'max_input', 9999,
    'Upper end of the range to pick numbers from.')
tf.app.flags.DEFINE_integer(
    'base', 10,
    'What base to interpret the numbers in.')

# Model flags:
tf.app.flags.DEFINE_integer(
    'state_vector_len', 100,
    'The length of the state vector for the RNN to maintain.')

# Training flags:
tf.app.flags.DEFINE_integer(
    'validation_size', 1000,
    'How many samples to generate for the validation set.')
tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'How many samples to generate per batch.')
tf.app.flags.DEFINE_integer(
    'steps', 1000000, 'How many training steps to perform.')
tf.app.flags.DEFINE_integer(
    'batches_per_step', 100,
    'How many batches to perform per step (i.e. between validation evals).')
tf.app.flags.DEFINE_float(
    'learning_rate', 1e-4, 'Learning rate (alpha) for ADAM optimizer.')

FLAGS = tf.app.flags.FLAGS

NUM_LABELS = 4


def digits(n, base):
  result = []
  while n > 0:
    result.append(n % base)
    n //= base

  return list(reversed(result))


def make_example(x):
  return {
      'digits': digits(x, FLAGS.base),
      'label': int(x % 3 == 0) + 2*int(x % 5 == 0)
  }


def random_batch(size):
  for _ in xrange(size):
    yield make_example(random.randint(0, FLAGS.max_input))


def main(unused_argv):
  with tf.Session() as sess:
    random.seed(FLAGS.seed)
    tf.set_random_seed(random.randint(0, 2**32))

    # First we set up TensorFlow Fold, building an RNN and one-hot labels:

    # 'process_next_digit' maps a state vector and a digit to a state vector.
    process_next_digit = (
        # leave the state vector alone and one-hot encode the digit
        (td.Identity(), td.OneHot(FLAGS.base)) >>
        # concatenate the state vector and the encoded digit together
        td.Concat() >>
        # pass the resulting vector through a fully connected neural network
        # layer to produce a state vector as output
        td.Function(td.FC(FLAGS.state_vector_len)))

    # td.Fold unrolls the neural net defined in 'process_next_digit', and
    # applies it to every element of a sequence, using zero as the initial
    # state vector.  Thus, process_digits takes a sequence of digits as input,
    # and returns the final state vector as output.
    process_digits = td.Fold(process_next_digit,
                             tf.zeros(FLAGS.state_vector_len))

    # This is the final model. It takes a dictionary (i.e, a record) of the
    # form {'digit': digit-sequence, 'label', label} as input,
    # and produces a tuple of (output-vector, OneHot(label)) as output.
    root_block = td.Record([('digits', process_digits),
                            ('label', td.OneHot(NUM_LABELS))])

    # An alternative to using Record is to use the following definition:
    # root_block = td.AllOf(td.GetItem('digits') >> process_digits,
    #                       td.GetItem('label') >> td.OneHot(NUM_LABELS))
    # AllOf passes its input to each of its children, and GetItem extracts
    # a field.  GetItem offers additional flexibility in more complex cases.

    # Compile root_block to get a tensorflow model that we can run.
    compiler = td.Compiler.create(root_block)

    # Get the tensorflow tensors that correspond to the outputs of root_block.
    digits_vecs, labels = compiler.output_tensors

    # We can now use digits_vecs and labels to compute losses and training
    # operations with tensorflow in the usual way.
    final_layer_weights = tf.Variable(
        tf.truncated_normal([FLAGS.state_vector_len, NUM_LABELS]),
        name='final_layer_weights')
    final_layer_biases = tf.Variable(
        tf.truncated_normal([NUM_LABELS]),
        name='final_layer_biases')

    logits = tf.matmul(
        digits_vecs, final_layer_weights) + final_layer_biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(labels, 1),
                         tf.argmax(logits, 1)),
                dtype=tf.float32))

    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)

    # Create a random batch of data to use for validation.
    validation_set = random_batch(FLAGS.validation_size)

    # TensorFlow Fold passes data into the model using feed_dict.
    validation_feed_dict = compiler.build_feed_dict(validation_set)

    # Now we actually train:
    sess.run(tf.global_variables_initializer())
    for step in xrange(FLAGS.steps):
      for _ in xrange(FLAGS.batches_per_step):
        # Create more random batches of training data, and build feed_dicts
        training_fd = compiler.build_feed_dict(random_batch(FLAGS.batch_size))
        sess.run(train_op, feed_dict=training_fd)

      validation_loss, validation_accuracy = sess.run(
          [loss, accuracy], feed_dict=validation_feed_dict)
      print('step:{:3}, loss ({}), accuracy: {:.0%}'.format(
          step, validation_loss, validation_accuracy))


if __name__ == '__main__':
  tf.app.run()
