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

"""A simple benchmark that builds binary trees or sequences in Loom."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import google3
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.loom import loom

tf.flags.DEFINE_integer('vector_length', 10,
                        'The size of the vectors to be added up.')
tf.flags.DEFINE_integer('depth', 10,
                        'How deep to make the binary trees of + operations.')
tf.flags.DEFINE_integer('batch_size', 1000, ''
                        'How many trees to build in the loom input.')
tf.flags.DEFINE_bool('sequence', False, ''
                     'Use sequences instead of trees.')
tf.flags.DEFINE_bool('unroll_loop', False, ''
                     'Unroll the loom\'s TF ops to a maximum depth.')
tf.flags.DEFINE_integer('num_constants', 275000,  # as of 6/15/16 takes ~1 sec
                        'How many constants to create.')

FLAGS = tf.flags.FLAGS


class BinaryLoomOp(loom.LoomOp):
  """Calls a tf op on two arguments."""

  def __init__(self, type_shape, op):
    self._type_shape = type_shape
    self._op = op
    super(BinaryLoomOp, self).__init__(
        [self._type_shape, self._type_shape], [self._type_shape])

  def instantiate_batch(self, inputs):
    return [self._op(inputs[0], inputs[1])]


def dry_build_tree(weaver, n):
  if n == 0: return None
  dry_build_tree(weaver, n-1)
  if not FLAGS.sequence:
    dry_build_tree(weaver, n-1)
  return None


def build_tree(weaver, n):
  if n == 0: return weaver.x

  left = build_tree(weaver, n-1)
  if FLAGS.sequence:
    return weaver.x
  else:
    right = build_tree(weaver, n-1)
  return weaver.add(left, right)


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.logging.info('Starting benchmark.')
  x = tf.constant(np.array(xrange(FLAGS.vector_length), dtype='float32'))
  shape = loom.TypeShape('float32', (FLAGS.vector_length,))
  ops = {'add': BinaryLoomOp(shape, tf.add)}
  the_loom = loom.Loom(max_depth=(FLAGS.depth if FLAGS.unroll_loop else None),
                       named_tensors={'x': x}, named_ops=ops)

  weaver = the_loom.make_weaver()
  tf.logging.info('Dry Recursion.')
  unused_dry_trees = [dry_build_tree(weaver, FLAGS.depth)
                      for _ in xrange(FLAGS.batch_size)]
  tf.logging.info('Building LoomOp trees.')
  trees = [build_tree(weaver, FLAGS.depth)
           for _ in xrange(FLAGS.batch_size)]
  tf.logging.info('Creating constants.')
  value = np.zeros(FLAGS.vector_length, dtype='float32')
  for _ in xrange(FLAGS.num_constants):
    weaver.constant(value)
  tf.logging.info('Building FeedDict.')
  _ = weaver.build_feed_dict(trees)

  tf.logging.info('Building LoomOp trees.')
  one_tree = the_loom.make_weaver()
  one_tree.add_output(build_tree(one_tree, FLAGS.depth))
  tf.logging.info('Serializing.')
  many_trees = [one_tree.serialize() for _ in xrange(FLAGS.batch_size)]
  tf.logging.info('Building FeedDict.')
  trees_feed_dict = {the_loom.input_tensor: many_trees}

  tf.logging.info('Running...')
  the_loom.output_tensor(shape).eval(
      session=tf.Session(),
      feed_dict=trees_feed_dict)
  tf.logging.info('Done.')


if __name__ == '__main__':
  tf.app.run()
