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
"""Training for TensorFlow Fold sentiment models.

Data from: <http://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip>
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
# import google3
import tensorflow as tf
from tensorflow_fold.blocks.examples.sentiment import sentiment
import tensorflow_fold.public.blocks as td


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string(
    'checkpoint_base', None, 'Path prefix for saving checkpoints')
flags.DEFINE_integer(
    'epochs', 20, 'Number of training epochs. Zero to run forever.')
flags.DEFINE_integer(
    'batch_size', 100, 'Number of examples per batch.')
flags.DEFINE_float(
    'learning_rate', 0.05, 'Learning rate for Adagrad optimizer.')
flags.DEFINE_float(
    'embedding_learning_rate_factor', 0.1,
    'Scaling factor for gradient updates to word embedding vectors.')
flags.DEFINE_float(
    'keep_prob', 0.75, 'Keep probability for dropout.')


def main(_):
  print('loading word embeddings from %s' % FLAGS.embedding_file)
  weight_matrix, word_idx = sentiment.load_embeddings(FLAGS.embedding_file)

  train_file = os.path.join(FLAGS.tree_dir, 'train.txt')
  print('loading training trees from %s' % train_file)
  train_trees = sentiment.load_trees(train_file)

  dev_file = os.path.join(FLAGS.tree_dir, 'dev.txt')
  print('loading dev trees from %s' % dev_file)
  dev_trees = sentiment.load_trees(dev_file)

  with tf.Session() as sess:
    print('creating the model')
    keep_prob = tf.placeholder_with_default(1.0, [])
    train_feed_dict = {keep_prob: FLAGS.keep_prob}
    word_embedding = sentiment.create_embedding(weight_matrix)
    compiler, metrics = sentiment.create_model(
        word_embedding, word_idx, FLAGS.lstm_num_units, keep_prob)
    loss = tf.reduce_sum(compiler.metric_tensors['all_loss'])
    opt = tf.train.AdagradOptimizer(FLAGS.learning_rate)
    grads_and_vars = opt.compute_gradients(loss)
    found = 0
    for i, (grad, var) in enumerate(grads_and_vars):
      if var == word_embedding.weights:
        found += 1
        grad = tf.scalar_mul(FLAGS.embedding_learning_rate_factor, grad)
        grads_and_vars[i] = (grad, var)
    assert found == 1  # internal consistency check
    train = opt.apply_gradients(grads_and_vars)
    saver = tf.train.Saver()

    print('initializing tensorflow')
    sess.run(tf.global_variables_initializer())

    with compiler.multiprocessing_pool():
      print('training the model')
      train_set = compiler.build_loom_inputs(train_trees)
      dev_feed_dict = compiler.build_feed_dict(dev_trees)
      dev_hits_best = 0.0
      for epoch, shuffled in enumerate(td.epochs(train_set, FLAGS.epochs), 1):
        train_loss = 0.0
        for batch in td.group_by_batches(shuffled, FLAGS.batch_size):
          train_feed_dict[compiler.loom_input_tensor] = batch
          _, batch_loss = sess.run([train, loss], train_feed_dict)
          train_loss += batch_loss
        dev_metrics = sess.run(metrics, dev_feed_dict)
        dev_loss = dev_metrics['all_loss']
        dev_accuracy = ['%s: %.2f' % (k, v * 100) for k, v in
                        sorted(dev_metrics.items()) if k.endswith('hits')]
        print('epoch:%4d, train_loss: %.3e, dev_loss: %.3e, dev_accuracy: [%s]'
              % (epoch, train_loss, dev_loss, ' '.join(dev_accuracy)))
        dev_hits = dev_metrics['root_hits']
        if dev_hits > dev_hits_best:
          dev_hits_best = dev_hits
          save_path = saver.save(sess, FLAGS.checkpoint_base, global_step=epoch)
          print('model saved in file: %s' % save_path)

if __name__ == '__main__':
  tf.app.run()
