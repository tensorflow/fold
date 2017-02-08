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
"""This is the benchmark code used in the ICLR 2017 paper.

The paper is entitled Deep Learning with Dynamic Computation graphs."
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import random
import time

# import google3

import numpy as np
import six

import tensorflow as tf

from tensorflow_fold.public import loom

tf.flags.DEFINE_integer("vector_size", 1024, "Size of tree RNN output vector.")
tf.flags.DEFINE_integer("tree_size", 128, "Size of trees to test.")
tf.flags.DEFINE_integer("num_repeats", 2, "Numer of times to repeat test.")
tf.flags.DEFINE_integer("num_epochs", 1, "Number of epochs.")
tf.flags.DEFINE_boolean("tree_lstm", True, "Use a tree lstm.")
tf.flags.DEFINE_string("tree_type", "random", "Type of tree to construct."
                       "Valid values are: random, sequence, or balanced")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log device placement.")
tf.flags.DEFINE_boolean("direct_feed_dict", True, "Use direct feed_dict.")
tf.flags.DEFINE_boolean("serialize_and_merge", False,
                        "Serialize each tree separately and merge them.")
tf.flags.DEFINE_boolean("train_with_loss", True, "Run SGD on a loss.")
tf.flags.DEFINE_boolean("quick_run", False, "Use a limited set of batch sizes.")

FLAGS = tf.flags.FLAGS


logging.basicConfig(format="%(asctime)s %(message)s")
_logger = logging.getLogger("benchmark")
_logger.setLevel(logging.INFO)


def make_random_tree(size):
  """Make a random binary tree with size nodes."""
  if size <= 1:
    return 0
  r = random.randint(1, size-1)
  return (make_random_tree(r), make_random_tree(size-r))


def make_sequence_tree(size):
  """Make a maximally unbalanced tree (a sequence) with size nodes."""
  if size <= 1:
    return 0
  return (make_sequence_tree(size-1), 0)


def make_balanced_tree(size):
  """Make a balanced binary tree with size nodes, where size is a power of 2."""
  if size <= 1:
    return 0
  return (make_balanced_tree(size/2), make_balanced_tree(size/2))


def make_input_tree(size):
  """Make a tree based on the value of the tree_type flag."""
  if FLAGS.tree_type == "sequence":
    return make_sequence_tree(size)
  elif FLAGS.tree_type == "balanced":
    return make_balanced_tree(size)
  elif FLAGS.tree_type == "random":
    return make_random_tree(size)
  raise ValueError("Invalid tree type: %s." % FLAGS.tree_type)


def index_type():
  return loom.TypeShape("int32", ())


def vector_type():
  return loom.TypeShape("float32", (FLAGS.vector_size,))


class LeafOp(loom.LoomOp):
  """Create a LoomOp for the leaf nodes.  We use a simple embedding table."""

  def __init__(self, embedding_size):
    super(LeafOp, self).__init__([index_type()],
                                 [vector_type()])
    self._embedding_size = embedding_size
    self._embedding = None
    self._vscope = "Leaf"

  def instantiate_batch(self, inputs):
    return [self(*inputs)]

  def __call__(self, indices):
    if self._embedding is None:
      with tf.variable_scope(self._vscope):
        self._embedding = (
            tf.get_variable("embedding_table",
                            [self._embedding_size, FLAGS.vector_size],
                            initializer=tf.random_uniform_initializer()))
    return tf.gather(self._embedding, indices)


class NonTerminalOp(loom.LoomOp):
  """Create a LoomOp for the non-terminals -- either a tree RNN or LSTM."""

  def __init__(self):
    super(NonTerminalOp, self).__init__([vector_type(), vector_type()],
                                        [vector_type()])
    self._weights = None
    self._bias = None
    self._vscope = "NonTerminal"

  def instantiate_batch(self, inputs):
    return [self(*inputs)]

  def tree_fc(self, left, right):
    # A simple tree RNN with a single fully connected layer.
    if self._weights is None:
      with tf.variable_scope(self._vscope):
        self._weights = tf.get_variable(
            "weights", [FLAGS.vector_size*2, FLAGS.vector_size],
            initializer=tf.uniform_unit_scaling_initializer(1.43))
        self._bias = tf.get_variable("bias", [FLAGS.vector_size],
                                     initializer=tf.zeros_initializer())
    x = tf.concat([left, right], 1)
    result = tf.add(tf.matmul(x, self._weights), self._bias)
    return tf.nn.relu(result)

  def tree_lstm(self, left, right):
    # A variation on the tree LSTM -- we add an extra hidden layer.
    if self._weights is None:
      with tf.variable_scope(self._vscope):
        self._weights_0 = tf.get_variable(
            "weights_0", [FLAGS.vector_size*2, FLAGS.vector_size],
            initializer=tf.uniform_unit_scaling_initializer(1.43))
        self._bias_0 = tf.get_variable("bias_0", [FLAGS.vector_size],
                                       initializer=tf.zeros_initializer())
        self._weights = tf.get_variable(
            "weights", [FLAGS.vector_size, FLAGS.vector_size*4],
            initializer=tf.uniform_unit_scaling_initializer(1.0))
        self._bias = tf.get_variable("bias", [FLAGS.vector_size*4],
                                     initializer=tf.zeros_initializer())
    # One hidden layer
    x = tf.concat([left, right], 1)
    h0 = tf.nn.relu(tf.add(tf.matmul(x, self._weights_0), self._bias_0))

    # Do a single matrix multiply to compute all gates
    h1 = tf.add(tf.matmul(h0, self._weights), self._bias)
    (hfl, hfr, hi, hg) = tf.split(h1, 4, axis=1)

    fl = tf.nn.sigmoid(hfl)  # forget left
    fr = tf.nn.sigmoid(hfr)  # forget right
    i = tf.nn.sigmoid(hi)    # input gate
    g = tf.nn.tanh(hg)       # computation

    ylr = tf.add(tf.multiply(fl, left), tf.multiply(fr, right))
    ygi = tf.multiply(i, g)
    y = tf.add(ylr, ygi)

    return y

  def __call__(self, left, right):
    if FLAGS.tree_lstm:
      return self.tree_lstm(left, right)
    else:
      return self.tree_fc(left, right)


class ModelBase(object):
  """Base class for the benchmark model."""

  def __init__(self, batch_size):
    # Use the tree size as the number of entries in the embedding table.
    self._embedding_size = FLAGS.tree_size
    self.batch_size = batch_size
    self._leaf_op = LeafOp(self._embedding_size)
    self._non_terminal_op = NonTerminalOp()
    self.elapsed_times = []
    self.elapsed_fd_times = []

  def random_index(self):
    """Get a random index into the embedding table."""
    return random.randint(0, self._embedding_size-1)

  def name(self):
    """Return the name of this model -- to be overridden by base classes."""
    return "Undefined."

  def build_model(self):
    """Build self._output -- to be overridden by base classes."""
    pass

  def build_model_loss(self):
    """Build model and add a loss function."""
    self.build_model()
    _logger.info("Differentiating.")
    if FLAGS.train_with_loss:
      # We don't actually care what the loss function is; we're just timing.
      self._loss = tf.nn.l2_loss(self._output)
      optr = tf.train.GradientDescentOptimizer(0.00001)
      self._train = optr.minimize(self._loss)
    else:
      self._loss = tf.reduce_sum(self._output)
      self._train = tf.constant(0.0)

  def build_feed_dict(self):
    """Build a feed dict for the model -- to be overridden by base classes."""
    return {}

  def evaluate(self, sess):
    """Run the model on random input data."""
    _logger.info("Testing for batch size %d.", self.batch_size)
    # We run the graph twice without timing to force any hidden initialization
    # and/or caching behavior to occur.
    for i in six.moves.xrange(0, 1):
      _logger.info("Burn-in %d.", i)
      fd = self.build_feed_dict()
      sess.run([self._train, self._loss], feed_dict=fd)

    # Small batch sizes have greater timing variation, so we compensate
    # by doing more batches.
    batch_size = self.batch_size
    if batch_size < 32:
      num_batches = int(32/self.batch_size) * FLAGS.num_repeats
    else:
      num_batches = FLAGS.num_repeats

    for batch in six.moves.xrange(0, num_batches):
      _logger.info("Batch: %d", batch)
      _logger.info("Build feed_dict.")
      start_time_fd = time.time()
      fd = self.build_feed_dict()
      end_time_fd = time.time()
      elapsed_fd = end_time_fd - start_time_fd
      self.elapsed_fd_times.append(elapsed_fd)

      _logger.info("Run.")
      start_time = time.time()
      [_, loss_v] = sess.run([self._train, self._loss], feed_dict=fd)
      end_time = time.time()
      elapsed = end_time - start_time
      self.elapsed_times.append(elapsed)

      _logger.info("Done.  Elapsed: %f [%f].  Loss: %f",
                   elapsed, elapsed_fd, loss_v)

  def run(self):
    """Build a graph and run the model on random input data."""
    _logger.info("Creating graph.")
    with tf.Graph().as_default():
      _logger.info("Building model.")
      self.build_model_loss()

      _logger.info("Starting session.")
      config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
      with tf.Session(config=config) as sess:
        _logger.info("Initializing variables.")
        sess.run(tf.global_variables_initializer())
        _logger.info("Starting timing test.")
        self.evaluate(sess)

      _logger.info("Ending session.")


class TfModel(ModelBase):
  """Native tensorflow tree model.

  All trees have the same shape so that they can be batched together
  without using Loom.
  """

  def __init__(self, batch_size):
    """Create a tree model that uses native TensorFlow."""
    # Use the tree size as the number of entries in the embedding table.
    super(TfModel, self).__init__(batch_size)
    self._placeholders = []

  def name(self):
    return "TensorFlow"

  def build_model(self):
    """The tensorflow model uses a tree of fixed shape."""
    tree = make_input_tree(FLAGS.tree_size)
    _logger.info("Tree: %s", str(tree))
    self._output = self.build_graph(tree)

  def build_graph(self, root):
    """Build a tensorflow computation graph with the same shape as root."""
    if isinstance(root, tuple):
      left = self.build_graph(root[0])
      right = self.build_graph(root[1])
      # We don't use the LoomOp with Loom.
      # Just call it to get the computation graph.
      return self._non_terminal_op(left, right)
    else:
      # Insert a placeholder for every leaf node.
      # We substitute random indices for the placeholders in build_feed_dict.
      indices = tf.placeholder(dtype="int32", shape=[self.batch_size])
      self._placeholders.append(indices)
      return self._leaf_op(indices)

  def build_feed_dict(self):
    """Create random indices for leaf nodes."""
    # Pass the indices via feed_dict, to ensure a fair comparison with Loom.
    # Each leaf has one index (times the number of trees in the batch).
    def rand_indices():
      return np.array(
          [self.random_index() for _ in six.moves.xrange(0, self.batch_size)],
          dtype="int32")
    return {p: rand_indices() for p in self._placeholders}


class LoomModel(ModelBase):
  """Model which uses dynamic batching with the Loom API."""

  def __init__(self, batch_size, proper_batching):
    """Create a tree model that uses the Loom API.

    Args:
      batch_size: The number of trees in the batch.
      proper_batching: If True, each tree has a different shape, otherwise
          each tree in the batch has the same shape for comparison with
          native TensorFlow.
    """
    # Use the tree size as the number of entries in the embedding table.
    super(LoomModel, self).__init__(batch_size)
    self._proper_batching = proper_batching

  def name(self):
    return "Loom"

  def build_model(self):
    """Build a model using Loom."""
    # Create a dictionary of the LoomOps that the model uses.
    named_tensors = {}
    named_ops = {
        "leaf": self._leaf_op,
        "non_terminal": self._non_terminal_op
    }
    # Make a random tree.
    self._tree = make_input_tree(FLAGS.tree_size)
    if not self._proper_batching:
      _logger.info("Tree: %s", str(self._tree))
    # Register all the of LoomOps with Loom.
    self._loom = loom.Loom(named_tensors=named_tensors,
                           named_ops=named_ops,
                           direct_feed_dict=FLAGS.direct_feed_dict)
    # Grab the TensorFlow tensor that holds the Loom output.
    self._output = self._loom.output_tensor(vector_type())

  def get_input_tree(self):
    """Get an input tree, either random or of fixed shape."""
    if self._proper_batching:
      # Make a different tree shape for each input in the batch.
      return make_input_tree(FLAGS.tree_size)
    else:
      # Use the same shape for each input in the batch.
      return self._tree

  def build_feed_dict(self):
    if FLAGS.serialize_and_merge:
      return self.build_feed_dict_with_serialize_and_merge()

    _logger.info("Traversing trees.")
    # The weaver is an object that can invoke LoomOps, and create a feed_dict.
    weaver = self._loom.make_weaver()
    # Recurse over each tree in the batch
    for _ in six.moves.xrange(0, self.batch_size):
      root = self.traverse_tree(self.get_input_tree(), weaver)
      weaver.add_output(root)

    # Now build the feed_dict, which contains both indices into the embedding
    # tables, and indices for the gather nodes inserted by Loom.
    if FLAGS.direct_feed_dict:
      _logger.info("Calling build_feed_dict in direct mode.")
    else:
      _logger.info("Calling build_feed_dict with serialization.")
    return weaver.build_feed_dict()

  def build_feed_dict_with_serialize_and_merge(self):
    """Serialize each tree with a separate weaver, and merge them together."""
    if FLAGS.direct_feed_dict:
      raise RuntimeError("Cannot serialize separately with direct_feed_dict.")

    _logger.info("Traversing and serializing trees.")
    # Recurse over each tree in the batch.
    serialized_trees = []
    for _ in six.moves.xrange(0, self.batch_size):
      weaver = self._loom.make_weaver()
      root = self.traverse_tree(self.get_input_tree(), weaver)
      weaver.add_output(root)
      serialized_trees.append(weaver.serialize())

    # Pass the serialized trees as the input tensors.
    return {self._loom.input_tensor: serialized_trees}

  def traverse_tree(self, node, weaver):
    # Recursive function to invoke a LoomOp on each node in the tree.
    if isinstance(node, tuple):
      left = self.traverse_tree(node[0], weaver)
      right = self.traverse_tree(node[1], weaver)
      # Invoke the Loom non_terminal op.
      return weaver.non_terminal(left, right)
    else:
      # Invoke the Loom leaf op, on a random index.
      idx = weaver(np.array(self.random_index(), dtype="int32"))
      return weaver.leaf(idx)


def test_model(model_class, *args):
  """Do a timing test for a model on a range of batch sizes."""
  test_results = {}
  if FLAGS.quick_run:
    batch_size_list = [1, 1024]
  else:
    batch_size_list = [1, 32, 64, 128, 256, 1024]

  for batch_size in batch_size_list:
    test_results[batch_size] = ([], [])
    for _ in six.moves.xrange(0, FLAGS.num_epochs):
      model = model_class(batch_size, *args)
      model.run()
      test_results[batch_size][0].extend(model.elapsed_times)
      test_results[batch_size][1].extend(model.elapsed_fd_times)
  return test_results


def print_results(test_results, model_name):
  """Print the results of a timing test."""
  def avg(lst):
    return sum(lst)/len(lst)

  _logger.info("Results for model %s:", model_name)
  result_list = list(six.iteritems(test_results))
  for (b, r) in sorted(result_list, reverse=True):
    (times, times_fd) = r
    tree_times = [t/b for t in times]
    tree_times_fd = [t/b for t in times_fd]

    _logger.info("Batch size: %d | per batch: %f [%f, %f] | "
                 "per tree: %f [%f, %f] | feed_dict: %f [%f, %f]",
                 b, avg(times), min(times), max(times),
                 avg(tree_times), min(tree_times), max(tree_times),
                 avg(tree_times_fd), min(tree_times_fd), max(tree_times_fd))


def compare_results(results1, results2, model_name1, model_name2):
  """Compare the results between two models."""
  def avg(lst):
    return sum(lst)/len(lst)

  _logger.info("Comparing %s to %s", model_name1, model_name2)
  rs1 = sorted(list(six.iteritems(results1)), reverse=True)
  rs2 = sorted(list(six.iteritems(results2)), reverse=True)
  for r in zip(rs1, rs2):
    ((b, (times1, _)), (_, (times2, _))) = r
    _logger.info("Batch size: %d | ratio: %f", b, avg(times2)/avg(times1))


def compare_total_speedup(test_results, baseline):
  """Get the total speedup over base line time for a single tree."""
  def avg(lst):
    return sum(lst)/len(lst)

  baseline_tree_time = avg(baseline[0])

  _logger.info("Speedup over baseline time %f.", baseline_tree_time)
  result_list = list(six.iteritems(test_results))
  for (b, r) in sorted(result_list, reverse=True):
    (times, _) = r
    tree_times = [t/b for t in times]
    avg_time = avg(tree_times)
    _logger.info("Batch size: %d | tree time: %f, speedup: %f", b,
                 avg_time, baseline_tree_time/avg_time)


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  _logger.info("Tensorflow Version: %s", str(tf.__version__))

  tf_results = test_model(TfModel)
  loom_results = test_model(LoomModel, False)
  loom_results_proper = test_model(LoomModel, True)

  if FLAGS.tree_lstm:
    model_type = "GRU"
  else:
    model_type = "FC"

  _logger.info("====================================================")
  _logger.info("Num epochs: %d; repeats per epoch %d",
               FLAGS.num_epochs, FLAGS.num_repeats)
  _logger.info("Model type: %s, %s", model_type, FLAGS.tree_type)
  _logger.info("Vector size: %d", FLAGS.vector_size)
  _logger.info("Tree size: %d", FLAGS.tree_size)

  print_results(tf_results, "TensorFlow")
  print_results(loom_results, "Loom")
  print_results(loom_results_proper, "Loom with random trees")

  compare_results(tf_results, loom_results, "TensorFlow", "Loom")
  compare_total_speedup(loom_results, tf_results[1])
  _logger.info("Finished benchmarks.")


if __name__ == "__main__":
  tf.app.run()
