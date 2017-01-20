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
"""Tests for tensorflow_fold.blocks.layers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import numpy as np
import six
import tensorflow as tf
from tensorflow_fold.blocks import blocks as tdb
from tensorflow_fold.blocks import layers as tdl
from tensorflow_fold.blocks import test_lib


class LayersTest(test_lib.TestCase):

  def test_fc_linear(self):
    fc = tdl.FC(1, None, tf.constant_initializer(2.0))
    with self.test_session() as sess:
      out = [fc(tf.constant([x], 'float32')) for x in [[0, 0], [1, 0], [1, 1]]]
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[[0]], [[2]], [[4]]], sess.run(out))

  def test_fc_relu(self):
    fc = tdl.FC(2, initializer=tf.constant_initializer(3.0))
    with self.test_session() as sess:
      out = [fc(tf.constant([x], 'float32')) for x in [[-1], [1]]]
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[[0, 0]], [[3, 3]]], sess.run(out))

  def test_tf_dropout(self):
    tf.set_random_seed(123)
    with self.test_session() as sess:
      in_kp = tf.placeholder(tf.float32)
      out_kp = tf.placeholder(tf.float32)
      fc = tdl.FC(1, None, tf.constant_initializer(2.),
                  input_keep_prob=in_kp, output_keep_prob=out_kp)
      out = fc(tf.ones([1, 1]))
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[2]], sess.run(out, {in_kp: 1.0, out_kp: 1.0}))
      self.assertAllEqual([[0]], sess.run(
          out, {in_kp: 1e-10, out_kp: 1.0}))  # kp=0 -> NaN
      self.assertAllEqual([[0]], sess.run(
          out, {in_kp: 1.0, out_kp: 1e-10}))  # kp=0 -> NaN

  def test_fc_raises(self):
    six.assertRaisesRegex(
        self, TypeError, 'FC input dtype must be float32', tdl.FC(1),
        tf.constant([0], dtype='int64'))
    six.assertRaisesRegex(
        self, TypeError, 'FC input shape must be 1D', tdl.FC(1),
        tf.constant(0, dtype='float32'))
    fc = tdl.FC(1)
    fc(tf.constant([[0]], 'float32'))
    six.assertRaisesRegex(
        self, TypeError, 'Type mismatch between input type', fc,
        tf.constant([[0, 0]], 'float32'))

  def test_embedding(self):
    weights = np.array([[1, 2], [3, 4]], dtype='float32')
    embedding = tdl.Embedding(2, 2, initializer=weights)
    with self.test_session() as sess:
      embeddings = [embedding(tf.constant([x])) for x in [0, 1, 7, -5]]
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[[1, 2]], [[3, 4]], [[3, 4]], [[3, 4]]],
                          sess.run(embeddings))

  def test_embedding_nomod(self):
    weights = np.array([[1, 2], [3, 4]], dtype='float32')
    embedding = tdl.Embedding(2, 2, initializer=weights, mod_inputs=False)
    with self.test_session() as sess:
      embeddings = [embedding(tf.constant([x])) for x in [0, 1]]
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[[1, 2]], [[3, 4]]], sess.run(embeddings))
      self.assertRaises(
          Exception,  # API doesn't specify what tf.gather() throws
          sess.run, embedding(tf.constant([2])))

  def test_embedding_int8(self):
    weights = np.array([[1, 2], [3, 4]], dtype='float32')
    embedding = tdl.Embedding(2, 2, initializer=weights)
    with self.test_session() as sess:
      embeddings = [embedding(tf.constant([x], dtype=tf.int8))
                    for x in [0, 1, 7, -5]]
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[[1, 2]], [[3, 4]], [[3, 4]], [[3, 4]]],
                          sess.run(embeddings))

  def test_embedding_uint8(self):
    weights = np.array([[1, 2], [3, 4]], dtype='float32')
    embedding = tdl.Embedding(2, 2, initializer=weights)
    with self.test_session() as sess:
      embeddings = [embedding(tf.constant([x], dtype=tf.uint8))
                    for x in [0, 1, 3]]
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[[1, 2]], [[3, 4]], [[3, 4]]],
                          sess.run(embeddings))

  def test_embedding_initializer(self):
    embedding = tdl.Embedding(2, 2, initializer=tf.constant_initializer(1.0))
    with self.test_session() as sess:
      embeddings = [embedding(tf.constant([x])) for x in [0, 1, 7]]
      sess.run(tf.global_variables_initializer())
      self.assertAllEqual([[[1, 1]], [[1, 1]], [[1, 1]]], sess.run(embeddings))

  def test_embedding_raises(self):
    self.assertRaises(ValueError, tdl.Embedding, 2, 2, np.zeros([3, 3]))
    six.assertRaisesRegex(
        self, TypeError, 'Embeddings take scalar inputs.', tdl.Embedding(2, 2),
        tf.constant([[0, 0]], 'int32'))
    six.assertRaisesRegex(
        self, TypeError, 'Embeddings take integer inputs.', tdl.Embedding(2, 2),
        tf.constant([0], 'float32'))

  def test_fractalnet_smoketest(self):
    input_placeholder = tf.placeholder(tf.float32, [None, 3])
    output_placeholder = tf.placeholder(tf.float32, [None, 3])
    fractal_net = tdl.FractalNet(3, 2, lambda name: tdl.FC(3, name=name))
    result = fractal_net(input_placeholder)
    loss = tf.nn.l2_loss(result - output_placeholder)
    optr = tf.train.GradientDescentOptimizer(0.001)
    trainer = optr.minimize(loss)

    dataset = np.random.standard_normal([10, 3])
    answers = np.random.standard_normal([10, 3])

    feed_dict = {input_placeholder: dataset, output_placeholder: answers}
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      old_loss = loss.eval(feed_dict)
      for unused_iteration in range(20):
        sess.run([trainer], feed_dict)
      new_loss = loss.eval(feed_dict)
      self.assertLess(new_loss, old_loss)

  def test_fractalnet_smoketest_random_drop_path(self):
    input_placeholder = tf.placeholder(tf.float32, [None, 3])
    output_placeholder = tf.placeholder(tf.float32, [None, 3])
    fractal_net = tdl.FractalNet(3, 2, lambda name: tdl.FC(3, name=name),
                                 drop_path=True)
    result = fractal_net(input_placeholder)
    loss = tf.nn.l2_loss(result - output_placeholder)
    optr = tf.train.GradientDescentOptimizer(0.001)
    trainer = optr.minimize(loss)

    dataset = np.random.standard_normal([10, 3])
    answers = np.random.standard_normal([10, 3])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      feed_dict = {input_placeholder: dataset,
                   output_placeholder: answers}

      np.random.seed(42)
      fractal_net.drop_path = False
      old_loss = loss.eval(feed_dict)
      for unused_iteration in range(100):
        fractal_net.drop_path = True
        sess.run([trainer], feed_dict)
      fractal_net.drop_path = False
      new_loss = loss.eval(feed_dict)
      self.assertLess(new_loss, old_loss)

  def test_create_variable_scope(self):
    with tf.variable_scope('foo'):
      self.assertEqual('foo/bar', tdl.create_variable_scope('bar').name)
      with tf.variable_scope('/'):
        pass
      # tf.variable_scope pushes a new name scope internally, so
      # creating variable scope 'foo/bar' when name scope /foo/bar/
      # already exists has the side-effect of creating name scope
      # /foo/bar_1', so the numbering skips to 2 here.
      self.assertEqual('foo/bar_2', tdl.create_variable_scope('bar').name)
      self.assertEqual('foo/bar/', tdl.create_variable_scope('bar/').name)
      self.assertEqual('foo/bar/', tdl.create_variable_scope('bar/').name)
      self.assertEqual('foo//', tdl.create_variable_scope('/').name)
    with tf.variable_scope('foo'):
      self.assertEqual('foo/bar_3', tdl.create_variable_scope('bar').name)
    self.assertEqual('foo/bar_4', tdl.create_variable_scope('foo/bar').name)
    self.assertEqual('foo/', tdl.create_variable_scope('foo/').name)
    self.assertEqual('foo/', tdl.create_variable_scope('foo/').name)

  def test_create_variable_scope2(self):
    with tf.variable_scope('outer'):
      with tf.variable_scope('inner'):
        pass
      self.assertEqual('outer/inner_1', tdl.create_variable_scope('inner').name)

  def test_rshift(self):
    block = (tdl.FC(1, None, tf.constant_initializer(2.0)) >>
             tdl.FC(1, None, tf.constant_initializer(3.0)))
    with self.test_session():
      self.assertEqual([6.0], (tdb.Vector(1) >> block).eval([1.0], tolist=True))

  def test_rrshift(self):
    block = tf.constant([3.0]) >> tdl.FC(1, None, tf.constant_initializer(2.0))
    with self.test_session():
      self.assertEqual([6.0], block.eval(None, tolist=True))

if __name__ == '__main__':
  test_lib.main()
