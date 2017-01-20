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
"""Eval for TensorFlow Fold sentiment models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
# import google3
import tensorflow as tf
from tensorflow_fold.blocks.examples.sentiment import sentiment

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_file', None, 'Model checkpoint file.')
FOLDS = ['test', 'dev', 'train']


def main(_):
  print('loading word embeddings from %s' % FLAGS.embedding_file)
  weight_matrix, word_idx = sentiment.load_embeddings(FLAGS.embedding_file)

  with tf.Session() as sess:
    print('restoring the model')
    word_embedding = sentiment.create_embedding(weight_matrix)
    compiler, metrics = sentiment.create_model(
        word_embedding, word_idx, FLAGS.lstm_num_units)
    saver = tf.train.Saver()
    saver.restore(sess, FLAGS.checkpoint_file)
    print('model restored from file: %s' % FLAGS.checkpoint_file)

    print('evaluating on trees from %s' % FLAGS.tree_dir)
    with compiler.multiprocessing_pool():
      filenames = [os.path.join(FLAGS.tree_dir, '%s.txt' % f) for f in FOLDS]
      for filename in filenames:
        trees = sentiment.load_trees(filename)
        print('file: %s, #trees: %d' % (filename, len(trees)))
        res = sorted(sess.run(metrics, compiler.build_feed_dict(trees)).items())
        print('      loss: [%s]' %
              ' '.join('%s: %.3e' % (name.rsplit('_', 1)[0], v)
                       for name, v in res if name.endswith('_loss')))
        print('  accuracy: [%s]' %
              ' '.join('%s: %.2f' % (name.rsplit('_', 1)[0], v * 100)
                       for name, v in res if name.endswith('_hits')))


if __name__ == '__main__':
  tf.app.run()
