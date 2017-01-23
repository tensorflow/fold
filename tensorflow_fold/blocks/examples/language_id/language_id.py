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
r"""Example of how to use TensorFlow Fold for a language model.

We create a model that splits sentences into words. Each word is processed by a
character-level RNN. The output of these character-level RNNs are used to feed a
word-level RNN.

The data is in a csv of the format:

  ```
  label1,example sentence one with only lowercase letters
  label2,example sentence two with only lowercase letters
  ...
  ```

Sentences consist of only lower case letters and spaces. There is no
punctuation. Labels are three letter abbreviations, like eng (English) or fra
(French).

Usage:

./tensorflow_fold/blocks/examples/language_id/fetch_datasets.sh

bazel run --config=opt \
    //tensorflow_fold/blocks/examples/language_id

See below and <tensorflow_fold/blocks/plan.py>
for flag options.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import csv
import random
# import google3
import tensorflow as tf
import tensorflow_fold.public.blocks as td


# Data flags:
tf.app.flags.DEFINE_string(
    'train_file', '/tmp/roman_sentences_train.csv', 'Train set CSV file.')
tf.app.flags.DEFINE_string(
    'dev_file', '/tmp/roman_sentences_dev.csv', 'Dev set CSV file.')


# Model flags:
tf.app.flags.DEFINE_integer(
    'char_embed_vector_length', 16,
    'The length of the embedding for each character.')
tf.app.flags.DEFINE_integer(
    'char_state_vector_length', 32,
    'The state vector length of the character RNN.')
tf.app.flags.DEFINE_integer(
    'word_state_vector_length', 32,
    'The state vector length of the word RNN.')

# Training flags:
tf.app.flags.DEFINE_integer(
    'seed', 42,
    'Random seed.')

td.define_plan_flags(default_plan_name='language_id')

FLAGS = tf.app.flags.FLAGS

NUM_LETTERS = 26  # data is preprocessed to only contain lowercase roman letters
LABELS = ['deu', 'eng', 'epo', 'fra', 'ita', 'nld', 'por', 'spa']
NUM_LABELS = len(LABELS)
LABEL_STR_TO_LABEL_IDX = dict([(l, idx) for idx, l in enumerate(LABELS)])


def sentence_rows(csv_file_in_path):
  """Yields CSV rows as dicts."""
  with tf.gfile.Open(csv_file_in_path, 'r') as csv_file_in:
    for label, sentence in csv.reader(csv_file_in):
      yield {'sentence': sentence, 'label': LABEL_STR_TO_LABEL_IDX[label]}


def rnn_block(input_block, state_length):
  """Get a fully connected RNN block.

  The input is concatenated with the state vector and put through a fully
  connected layer to get the next state vector.

  Args:
    input_block: Put each input through this before concatenating it with the
      current state vector.
    state_length: Length of the RNN state vector.

  Returns:
    RNN Block (seq of input_block inputs -> output state)
  """
  combine_block = ((td.Identity(), input_block) >> td.Concat()
                   >> td.Function(td.FC(state_length)))
  return td.Fold(combine_block, tf.zeros(state_length))


# All characters are lowercase, so subtract 'a' to make them 0-indexed.
def word_to_char_index(word):
  return [ord(char) - ord('a') for char in word]


def sentence_to_words(sentence):
  return sentence.split(' ')


def setup_plan(plan):
  """Sets up a TensorFlow Fold plan for language identification."""
  if plan.mode != 'train': raise ValueError('only train mode is supported')

  embed_char = (td.Scalar(tf.int32) >>
                td.Embedding(NUM_LETTERS, FLAGS.char_embed_vector_length))

  process_word = (td.InputTransform(word_to_char_index) >>
                  rnn_block(embed_char, FLAGS.char_state_vector_length))

  sentence = (td.InputTransform(sentence_to_words) >>
              rnn_block(process_word, FLAGS.word_state_vector_length) >>
              td.FC(NUM_LABELS, activation=None))

  label = td.Scalar('int64')

  # This is the final model. It takes a dictionary (i.e, a record) of the
  # form
  #
  #   {
  #     'sentence': sentence-string,
  #     'label': label-integer
  #   }
  #
  # as input and produces a tuple of (unscaled_logits, label) as output.
  root_block = td.Record([('sentence', sentence), ('label', label)])

  # Compile root_block to get a tensorflow model that we can run.
  plan.compiler = td.Compiler.create(root_block)

  # Get the tensorflow tensors that correspond to the outputs of root_block.
  # These are TF tensors, and we can use them to compute loss in the usual way.
  logits, labels = plan.compiler.output_tensors

  plan.losses['cross_entropy'] = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
  predictions = tf.argmax(logits, 1)
  plan.metrics['accuracy'] = tf.reduce_mean(
      tf.cast(tf.equal(predictions, labels), dtype=tf.float32))

  plan.examples = sentence_rows(FLAGS.train_file)
  plan.dev_examples = sentence_rows(FLAGS.dev_file)


def main(_):
  random.seed(FLAGS.seed)
  tf.set_random_seed(random.randint(0, 2**32))
  td.Plan.create_from_flags(setup_plan).run()

if __name__ == '__main__':
  tf.app.run()
