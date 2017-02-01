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
"""Recursive neural network for sentiment analysis using TensorFlow Fold.

Dataset: <nlp.stanford.edu/sentiment/treebank.html>

The input is an s-expression encoded as a string, e.g.
`(4 (2 Spiderman) (3 ROCKS))`.

For training there is a TensorFlow Fold metric which sums the loss
over all of all of the nodes ('all_loss'). For calculating accuracy,
there are metric tensors for root vs. all hits as well as fine-grained
vs. binary.

The word model is based on GloVe vectors <https://github.com/stanfordnlp/GloVe>.
Subtree embeddings and predictions are made using a tree LSTM. Dropout is used
exclively for regularization.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
# import google3
from nltk.tokenize import sexpr
import numpy as np
import tensorflow as tf
import tensorflow_fold.public.blocks as td

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'tree_dir', None, 'Directory for trees (train.txt, dev.txt, test.txt).')
flags.DEFINE_string(
    'embedding_file', None, 'File name for word embeddings.')
flags.DEFINE_integer(
    'lstm_num_units', 300, 'Number of units for tree LSTM.')
NUM_CLASSES = 5  # number of classes for fine-grained sentiment analysis


def load_trees(filename):
  with codecs.open(filename, encoding='utf-8') as f:
    # Drop the trailing newline and strip \s.
    return [line.strip().replace('\\', '') for line in f]


def load_embeddings(filename):
  """Loads embedings, returns weight matrix and dict from words to indices."""
  weight_vectors = []
  word_idx = {}
  with codecs.open(filename, encoding='utf-8') as f:
    for line in f:
      word, vec = line.split(u' ', 1)
      word_idx[word] = len(weight_vectors)
      weight_vectors.append(np.array(vec.split(), dtype=np.float32))
  # Annoying implementation detail; '(' and ')' are replaced by '-LRB-' and
  # '-RRB-' respectively in the parse-trees.
  word_idx[u'-LRB-'] = word_idx.pop(u'(')
  word_idx[u'-RRB-'] = word_idx.pop(u')')
  # Random embedding vector for unknown words.
  weight_vectors.append(np.random.uniform(
      -0.05, 0.05, weight_vectors[0].shape).astype(np.float32))
  return np.stack(weight_vectors), word_idx


def create_embedding(weight_matrix):
  return td.Embedding(*weight_matrix.shape, initializer=weight_matrix,
                      name='word_embedding')


def create_model(word_embedding, word_idx, lstm_num_units, keep_prob=1):
  """Creates a sentiment model. Returns (compiler, mean metrics)."""
  tree_lstm = td.ScopedLayer(
      tf.contrib.rnn.DropoutWrapper(
          BinaryTreeLSTMCell(lstm_num_units, keep_prob=keep_prob),
          input_keep_prob=keep_prob, output_keep_prob=keep_prob),
      name_or_scope='tree_lstm')

  output_layer = td.FC(NUM_CLASSES, activation=None, name='output_layer')

  embed_subtree = td.ForwardDeclaration(output_type=tree_lstm.state_size)

  unknown_idx = len(word_idx)
  def lookup_word(word):
    return word_idx.get(word, unknown_idx)

  def logits_and_state():
    """Creates a block that goes from tokens to (logits, state) tuples."""
    word2vec = (td.GetItem(0) >> td.InputTransform(lookup_word) >>
                td.Scalar('int32') >> word_embedding)

    pair2vec = (embed_subtree(), embed_subtree())

    # Trees are binary, so the tree layer takes two states as its input_state.
    zero_state = td.Zeros((tree_lstm.state_size,) * 2)
    # Input is a word vector.
    zero_inp = td.Zeros(word_embedding.output_type.shape[0])

    word_case = td.AllOf(word2vec, zero_state)
    pair_case = td.AllOf(zero_inp, pair2vec)

    tree2vec = td.OneOf(len, [(1, word_case), (2, pair_case)])

    return tree2vec >> tree_lstm >> (output_layer, td.Identity())

  model = embed_tree(logits_and_state(), is_root=True)

  embed_subtree.resolve_to(embed_tree(logits_and_state(), is_root=False))

  compiler = td.Compiler.create(model)
  metrics = {k: tf.reduce_mean(v) for k, v in compiler.metric_tensors.items()}
  return compiler, metrics


def embed_tree(logits_and_state, is_root):
  """Creates a block that embeds trees; output is tree LSTM state."""
  return td.InputTransform(tokenize) >> td.OneOf(
      key_fn=lambda pair: pair[0] == '2',  # label 2 means neutral
      case_blocks=(add_metrics(is_root, is_neutral=False),
                   add_metrics(is_root, is_neutral=True)),
      pre_block=(td.Scalar('int32'), logits_and_state))


def tokenize(s):
  # sexpr_tokenize can't parse 'foo bar', only '(foo) (bar)', so we
  # use split to handle the case of a leaf (e.g. 'label word').
  label, phrase = s[1:-1].split(None, 1)
  return label, sexpr.sexpr_tokenize(phrase)


def add_metrics(is_root, is_neutral):
  """A block that adds metrics for loss and hits; output is the LSTM state."""
  c = td.Composition(
      name='predict(is_root=%s, is_neutral=%s)' % (is_root, is_neutral))
  with c.scope():
    # destructure the input; (label, (logits, state))
    y_ = c.input[0]
    logits = td.GetItem(0).reads(c.input[1])
    state = td.GetItem(1).reads(c.input[1])

    # predict the label from the logits
    y = td.Function(lambda x: tf.cast(tf.argmax(x, 1), tf.int32)).reads(logits)

    # calculate loss
    loss = td.Function(_loss)
    td.Metric('all_loss').reads(loss.reads(logits, y_))
    if is_root: td.Metric('root_loss').reads(loss)

    # calculate hits
    hits = td.Function(lambda y, y_: tf.cast(tf.equal(y, y_), tf.float64))
    td.Metric('all_hits').reads(hits.reads(y, y_))
    if is_root: td.Metric('root_hits').reads(hits)

    # calculate binary hits, if the label is not neutral
    if not is_neutral:
      binary_hits = td.Function(tf_binary_hits).reads(logits, y_)
      td.Metric('all_binary_hits').reads(binary_hits)
      if is_root: td.Metric('root_binary_hits').reads(binary_hits)

    # output the state, which will be read by our by parent's LSTM cell
    c.output.reads(state)
  return c


def _loss(logits, labels):
  return tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)


def tf_binary_hits(logits, y_):
  softmax = tf.nn.softmax(logits)
  binary_y = (softmax[:, 3] + softmax[:, 4]) > (softmax[:, 0] + softmax[:, 1])
  binary_y_ = y_ > 2
  return tf.cast(tf.equal(binary_y, binary_y_), tf.float64)


class BinaryTreeLSTMCell(tf.contrib.rnn.BasicLSTMCell):
  """LSTM with two state inputs.

  This is the model described in section 3.2 of 'Improved Semantic
  Representations From Tree-Structured Long Short-Term Memory
  Networks' <http://arxiv.org/pdf/1503.00075.pdf>, with recurrent
  dropout as described in 'Recurrent Dropout without Memory Loss'
  <http://arxiv.org/pdf/1603.05118.pdf>.
  """

  def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh,
               keep_prob=1.0, seed=None):
    """Initialize the cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      activation: Activation function of the inner states.
      keep_prob: Keep probability for recurrent dropout.
      seed: Random seed for dropout.
    """
    super(BinaryTreeLSTMCell, self).__init__(
        num_units, forget_bias=forget_bias, activation=activation)
    self._keep_prob = keep_prob
    self._seed = seed

  def __call__(self, inputs, state, scope=None):
    with tf.variable_scope(scope or type(self).__name__):
      lhs, rhs = state
      c0, h0 = lhs
      c1, h1 = rhs
      concat = tf.contrib.layers.linear(
          tf.concat([inputs, h0, h1], 1), 5 * self._num_units)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f0, f1, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

      j = self._activation(j)
      if not isinstance(self._keep_prob, float) or self._keep_prob < 1:
        j = tf.nn.dropout(j, self._keep_prob, seed=self._seed)

      new_c = (c0 * tf.sigmoid(f0 + self._forget_bias) +
               c1 * tf.sigmoid(f1 + self._forget_bias) +
               tf.sigmoid(i) * j)
      new_h = self._activation(new_c) * tf.sigmoid(o)

      new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

      return new_h, new_state
