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
"""Tests for tensorflow_fold.blocks.plan."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
# import google3
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import block_compiler
from tensorflow_fold.blocks import blocks
from tensorflow_fold.blocks import plan
from tensorflow_fold.blocks import test_lib
from tensorflow_fold.blocks.test_lib import mock

plan.define_plan_flags()


class ParseSpecTest(test_lib.TestCase):

  def test_well_formed(self):
    self.assertEqual(dict(str='str', three=3, half=0.5),
                     plan.parse_spec('str=str,three=3,half=0.5'))

  def test_malformed(self):
    six.assertRaisesRegex(
        self, ValueError, 'Clause "foo" doesn\'t contain an "="',
        plan.parse_spec, 'foo')
    six.assertRaisesRegex(
        self, ValueError, 'Clause "bar" doesn\'t contain an "="',
        plan.parse_spec, 'foo=3.0,bar')
    six.assertRaisesRegex(
        self, ValueError, 'Clause "bar" doesn\'t contain an "="',
        plan.parse_spec, 'foo=3.0,bar,baz=7')


class BuildOptimizerFromParams(test_lib.TestCase):

  def test_bad_optimizer(self):
    six.assertRaisesRegex(
        self, ValueError, 'Unrecognized optimizer: magic',
        plan.build_optimizer_from_params, 'magic')

  def test_bad_argument(self):
    six.assertRaisesRegex(
        self, ValueError,
        'The adam optimizer doesn\'t take an argument named magic_flag',
        plan.build_optimizer_from_params, magic_flag=3)

  def test_missing_argument(self):
    six.assertRaisesRegex(
        self, ValueError, 'The adagrad optimizer requires learning_rate '
        'to be set.', plan.build_optimizer_from_params, 'adagrad')

  def test_name_variants(self):
    self.assertTrue(isinstance(
        plan.build_optimizer_from_params('adagrad', learning_rate=1e-3),
        tf.train.AdagradOptimizer))
    self.assertTrue(isinstance(
        plan.build_optimizer_from_params('ada_grad', learning_rate=1e-3),
        tf.train.AdagradOptimizer))
    self.assertTrue(isinstance(
        plan.build_optimizer_from_params('ADAGRAD', learning_rate=1e-3),
        tf.train.AdagradOptimizer))


class PlanTestBase(test_lib.TestCase):

  def setUp(self):
    super(PlanTestBase, self).setUp()
    # Reset flags.
    tf.flags.FLAGS.master = ''
    tf.flags.FLAGS.mode = 'train'
    tf.flags.FLAGS.num_multiprocess_processes = None
    tf.flags.FLAGS.task = 0
    tf.flags.FLAGS.truncate_examples = 0
    # Recreate tmpdir.
    tmpdir = self.get_temp_dir()
    shutil.rmtree(tmpdir, ignore_errors=True)
    os.mkdir(tmpdir)


class PlanTest(PlanTestBase):

  def test_create(self):
    p = plan.Plan.create(plan.Plan.mode_keys.TRAIN)
    self.assertEqual(p.mode, plan.Plan.mode_keys.TRAIN)
    self.assertTrue(isinstance(p, plan.TrainPlan))

    p = plan.Plan.create(plan.Plan.mode_keys.EVAL)
    self.assertEqual(p.mode, plan.Plan.mode_keys.EVAL)
    self.assertTrue(isinstance(p, plan.EvalPlan))

    p = plan.Plan.create(plan.Plan.mode_keys.INFER)
    self.assertEqual(p.mode, plan.Plan.mode_keys.INFER)
    self.assertTrue(isinstance(p, plan.InferPlan))

    self.assertRaisesWithLiteralMatch(
        ValueError,
        'invalid mode \'foo\' not in [\'eval\', \'infer\', \'train\']',
        plan.Plan.create, 'foo')

  def test_finalize_stats_no_summaries(self):
    p = plan.Plan(None)
    p.losses['foo'] = tf.constant([1.0])
    p.metrics['bar'] = tf.constant(2)
    self.assertEqual(p.summaries, None)
    p.finalize_stats()
    with self.test_session():
      self.assertEqual(1, p.loss_total.eval())
    self.assertRaisesWithLiteralMatch(
        RuntimeError, 'finalize_stats() has already been called',
        p.finalize_stats)

  def test_finalize_stats_invalid_loss_dtype(self):
    p = plan.Plan(None)
    p.losses['foo'] = tf.constant(1)
    self.assertRaisesWithLiteralMatch(
        TypeError, 'invalid loss dtype tf.int32, must be a floating point type',
        p.finalize_stats)

  def test_finalize_stats_summaries(self):
    p = plan.Plan(None)
    p.save_summaries_secs = 42
    p.losses['foo'] = tf.constant([1.0])
    p.losses['bar'] = tf.constant([2.0, 3.0])
    p.metrics['baz'] = tf.constant(4)
    p.metrics['qux'] = tf.constant([5.0, 6.0])
    p.finalize_stats()
    with self.test_session():
      self.assertEqual(6, p.loss_total.eval({p.batch_size_placeholder: 1}))
      summary = tf.Summary()
      summary.ParseFromString(p.summaries.eval({p.batch_size_placeholder: 1}))
      qux_string = tf.summary.histogram('qux', [5, 6]).eval()
      qux_proto = tf.Summary()
      qux_proto.ParseFromString(qux_string)
      qux_histogram = qux_proto.value[0].histo
      expected_values = [
          tf.Summary.Value(tag='foo', simple_value=1),
          tf.Summary.Value(tag='bar', simple_value=5),
          tf.Summary.Value(tag='loss_total', simple_value=6),
          tf.Summary.Value(tag='baz', simple_value=4),
          tf.Summary.Value(tag='qux', histo=qux_histogram)]
      six.assertCountEqual(self, expected_values, summary.value)
      summary.ParseFromString(p.summaries.eval({p.batch_size_placeholder: 2}))
      expected_values = [
          tf.Summary.Value(tag='foo', simple_value=0.5),
          tf.Summary.Value(tag='bar', simple_value=2.5),
          tf.Summary.Value(tag='loss_total', simple_value=3),
          tf.Summary.Value(tag='baz', simple_value=4),
          tf.Summary.Value(tag='qux', histo=qux_histogram)]
      six.assertCountEqual(self, expected_values, summary.value)

  def test_finalize_stats_summaries_empty(self):
    p = plan.Plan(None)
    p.save_summaries_secs = 42
    p.finalize_stats()
    with self.test_session():
      self.assertEqual(b'', p.summaries.eval())

  def test_assert_runnable(self):
    p = plan.Plan(None)
    self.assertRaisesWithLiteralMatch(
        ValueError, 'compiler is required', p.assert_runnable)
    p.compiler = block_compiler.Compiler.create(blocks.Scalar())
    self.assertRaisesWithLiteralMatch(
        ValueError, 'logdir is required', p.assert_runnable)
    p.logdir = '/tmp/'
    p.assert_runnable()

  def test_lazy_length(self):
    it, count = plan._lazy_length(xrange(2, 5))
    self.assertEqual(count, 0)
    self.assertEqual(next(it), 2)
    self.assertEqual(count, 1)
    self.assertEqual(list(it), [3, 4])
    self.assertEqual(count, 3)


class TrainPlanTest(PlanTestBase):

  def test_create_from_flags(self):
    tf.flags.FLAGS.mode = plan.Plan.mode_keys.TRAIN
    tf.flags.FLAGS.truncate_examples = 3
    tf.flags.FLAGS.num_multiprocess_processes = 4
    tf.flags.FLAGS.master = 'foo'
    tf.flags.FLAGS.batches_per_epoch = 123
    foo = tf.get_variable('foo', [], tf.float32, tf.constant_initializer(4))
    p = plan.Plan.create_from_flags(_setup_plan(
        compiler=block_compiler.Compiler.create(blocks.Scalar()),
        losses={'foo': foo},
        examples=xrange(5)))
    self.assertEqual(p.num_multiprocess_processes, 4)
    self.assertEqual(p.master, 'foo')
    self.assertEqual(p.batches_per_epoch, 123)
    self.assertEqual(p.compute_summaries, True)
    self.assertEqual(p.is_chief_trainer, True)
    self.assertEqual(p.logdir, os.path.join('/tmp/', 'plan', 'run_0', 'train'))
    self.assertEqual(p.rundir, os.path.join('/tmp/', 'plan', 'run_0'))
    self.assertEqual(p.plandir, os.path.join('/tmp/', 'plan'))
    self.assertEqual([0, 1, 2], list(p.examples))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(4, p.loss_total.eval())
      sess.run(p.train_op)  # should make loss smaller
      self.assertLess(p.loss_total.eval(), 4)

    tf.flags.FLAGS.num_multiprocess_processes = None
    tf.flags.FLAGS.task = 42
    train_op = tf.no_op()
    p = plan.Plan.create_from_flags(_setup_plan(
        compiler=block_compiler.Compiler.create(blocks.Scalar()),
        losses={'foo': tf.constant(3.14)},
        train_op=train_op,
        examples=xrange(5)))
    self.assertEqual(p.num_multiprocess_processes, None)
    self.assertEqual(p.compute_summaries, False)
    self.assertEqual(p.is_chief_trainer, False)
    self.assertEqual(p.train_op, train_op)

  def test_create_from_params(self):
    params = plan.plan_default_params()
    params.update({
        'mode': plan.Plan.mode_keys.TRAIN,
        'truncate_examples': 3,
        'num_multiprocess_processes': 4,
        'master': 'foo',
        'batches_per_epoch': 123})
    foo = tf.get_variable('foo', [], tf.float32, tf.constant_initializer(4))
    p = plan.Plan.create_from_params(_setup_plan(
        compiler=block_compiler.Compiler.create(blocks.Scalar()),
        losses={'foo': foo},
        examples=xrange(5)), params)
    self.assertEqual(p.num_multiprocess_processes, 4)
    self.assertEqual(p.master, 'foo')
    self.assertEqual(p.batches_per_epoch, 123)
    self.assertEqual(p.compute_summaries, True)
    self.assertEqual(p.is_chief_trainer, True)
    self.assertEqual(p.logdir, os.path.join('/tmp/', 'plan', 'run_0', 'train'))
    self.assertEqual(p.rundir, os.path.join('/tmp/', 'plan', 'run_0'))
    self.assertEqual(p.plandir, os.path.join('/tmp/', 'plan'))
    self.assertEqual([0, 1, 2], list(p.examples))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual(4, p.loss_total.eval())
      sess.run(p.train_op)  # should make loss smaller
      self.assertLess(p.loss_total.eval(), 4)

    tf.flags.FLAGS.num_multiprocess_processes = None
    tf.flags.FLAGS.task = 42
    train_op = tf.no_op()
    p = plan.Plan.create_from_flags(_setup_plan(
        compiler=block_compiler.Compiler.create(blocks.Scalar()),
        losses={'foo': tf.constant(3.14)},
        train_op=train_op,
        examples=xrange(5)))
    self.assertEqual(p.num_multiprocess_processes, None)
    self.assertEqual(p.compute_summaries, False)
    self.assertEqual(p.is_chief_trainer, False)
    self.assertEqual(p.train_op, train_op)

  def test_finalize_stats_raises(self):
    self.assertRaisesWithLiteralMatch(
        ValueError, 'at least one loss is required',
        plan.TrainPlan().finalize_stats)

  def test_assert_runnable(self):
    p = plan.TrainPlan()
    self.assertRaisesWithLiteralMatch(
        ValueError, 'at least one loss is required', p.assert_runnable)
    p.losses['foo'] = tf.constant(42.0)
    self.assertRaisesWithLiteralMatch(
        ValueError, 'compiler is required', p.assert_runnable)
    p.compiler = block_compiler.Compiler.create(blocks.Scalar())
    self.assertRaisesWithLiteralMatch(
        RuntimeError, 'finalize_stats() has not been called', p.assert_runnable)
    p.finalize_stats()
    self.assertRaisesWithLiteralMatch(
        ValueError, 'logdir is required', p.assert_runnable)
    p.logdir = '/tmp/'
    self.assertRaisesWithLiteralMatch(
        ValueError, 'train_op is required', p.assert_runnable)
    p.train_op = tf.no_op()
    self.assertRaisesWithLiteralMatch(
        ValueError, 'batch_size is required', p.assert_runnable)
    p.batch_size = 10
    self.assertRaisesWithLiteralMatch(
        ValueError, 'either examples or batches_per_epoch is required',
        p.assert_runnable)
    p.examples = xrange(2)
    p.assert_runnable()
    p.examples = None
    self.assertRaises(ValueError, p.assert_runnable)
    p.batches_per_epoch = 42
    p.assert_runnable()

  def test_run_feed_dict(self):
    p = self.create_plan(loom_input_tensor=None)
    p.examples = [1] * 4
    self.check_plan(p, [])

  def test_empty_examples_raises(self):
    p = self.create_plan(loom_input_tensor=None)
    p.examples = iter([])
    self.assertRaisesWithLiteralMatch(
        ValueError, 'examples must be non-empty', self.check_plan, p, [])

  def test_exact_batch_sizes_examples_too_small_raises(self):
    p = self.create_plan(loom_input_tensor=None)
    p.examples = [1]
    p.exact_batch_sizes = True
    self.assertRaisesWithLiteralMatch(
        ValueError, 'when exact_batch_sizes is true, examples must have at '
        'least batch_size items; 1 vs. 2', self.check_plan, p, [])

  def test_run_input_tensor(self):
    total_batches = 6  # batches_per_epoch * epochs = 2 * 3 = 6
    q = tf.FIFOQueue(total_batches, [tf.string])
    p = self.create_plan(loom_input_tensor=q.dequeue())
    p.batches_per_epoch = 2
    loom_input, = list(p.compiler.build_loom_inputs([1]))
    batch = [loom_input] * p.batch_size
    setup_ops = [q.enqueue([batch]) for _ in xrange(total_batches)]
    self.check_plan(p, setup_ops)

  def create_plan(self, loom_input_tensor):
    p = plan.TrainPlan()
    foo = tf.get_variable('foo', [], tf.float32, tf.constant_initializer(12))
    p.compiler = block_compiler.Compiler.create(
        blocks.Scalar() >> blocks.Function(lambda x: x * foo),
        loom_input_tensor=loom_input_tensor)
    p.losses['foo'] = p.compiler.output_tensors[0]
    p.finalize_stats()
    p.train_op = tf.train.GradientDescentOptimizer(1.0).minimize(
        p.loss_total, global_step=p.global_step)
    p.logdir = self.get_temp_dir()
    p.dev_examples = [2]
    p.is_chief_trainer = True
    p.batch_size = 2
    p.epochs = 3
    p.print_file = six.StringIO()
    return p

  def check_plan(self, p, setup_ops):
    # We aren't using a managed session, so we need to run this ourselves.
    init_op = tf.global_variables_initializer()
    sv = p.create_supervisor()
    with self.test_session() as sess:
      sess.run(init_op)
      sess.run(setup_ops)
      self.assertEqual([12, 12], sess.run(
          p.losses['foo'], p.compiler.build_feed_dict([1, 1])).tolist())
      p.run(sv, sess)
      expected_lines = [
          'epoch:    1 train[loss: 1.100e+01] dev[loss: 1.600e+01]',
          'new best model saved in file: %s' % (
              os.path.join(p.logdir, 'model.ckpt-2')),
          'epoch:    2 train[loss: 7.000e+00] dev[loss: 8.000e+00]',
          'new best model saved in file: %s' % (
              os.path.join(p.logdir, 'model.ckpt-4')),
          'epoch:    3 train[loss: 3.000e+00] dev[loss: 0.000e+00]',
          'new best model saved in file: %s' % (
              os.path.join(p.logdir, 'model.ckpt-6'))]
      expected = '\n'.join(expected_lines) + '\n'
      log_str = p.print_file.getvalue()
      self.assertTrue(log_str.endswith(expected), msg=log_str)
      self.assertEqual([0, 0], sess.run(
          p.losses['foo'], p.compiler.build_feed_dict([1, 1])).tolist())

  def test_init_loom(self):
    p = plan.TrainPlan()
    p.compiler = block_compiler.Compiler().compile(blocks.Scalar())
    p.batch_size = 3
    p.task = 13
    p.num_dequeuers = 7

    self.assertRaisesWithLiteralMatch(
        ValueError, 'must have at least one PS task; 0', p.init_loom)

    p.ps_tasks = 5
    self.assertRaisesWithLiteralMatch(
        ValueError, 'worker_replicas must be at least num_queues + '
        'num_dequeuers; 0 vs. 5 + 7 = 12', p.init_loom)
    p.worker_replicas = 14

    # Would be best to actually create a queue and inspect it, but
    # tf.QueueBase doesn't currently expose these properties.
    self.assertEqual(p._create_queue(3, ctor=dict)['capacity'], 12)
    p.queue_capacity = 42
    q_dict = p._create_queue(3, ctor=dict)
    self.assertEqual(q_dict['capacity'], 42)
    self.assertEqual(q_dict['shared_name'], 'tensorflow_fold_plan_queue3')

    self.assertEqual(p.init_loom(), (True, False))

    p.compiler = block_compiler.Compiler().compile(blocks.Scalar())
    p.task = 3
    self.assertEqual(p.init_loom(), (False, True))

    p.compiler = block_compiler.Compiler().compile(blocks.Scalar())
    p.num_dequeuers = 0
    self.assertRaisesWithLiteralMatch(
        ValueError, 'cannot specify queue_capacity without also '
        'specifying num_dequeuers', p.init_loom)

    p.compiler = block_compiler.Compiler().compile(blocks.Scalar())
    p.queue_capacity = 0
    self.assertEqual(p.init_loom(), (True, True))

  def test_enqueue(self):
    p = plan.TrainPlan()
    p.compiler = block_compiler.Compiler().compile(blocks.Scalar())
    p.examples = [7] * 8  # two items should be ignored (8 % 3 == 2)
    p.is_chief_trainer = True
    p.batch_size = 3
    p.queue_capacity = 12
    p.num_dequeuers = 1
    p.ps_tasks = 1
    q = p._create_queue(0)
    p._setup_enqueuing([q])
    q_size = q.size()
    q_dequeue = q.dequeue_many(12)
    p.finalize_stats()
    p.logdir = self.get_temp_dir()
    p.epochs = 2
    p.print_file = six.StringIO()
    init_op = tf.global_variables_initializer()
    sv = p.create_supervisor()
    with self.test_session() as sess:
      sess.run(init_op)
      p.run(sv, sess)
      expected = '\n'.join(['running train',
                            'train_size: 6',
                            'epoch:    1 train[loss: 0.000e+00]',
                            'epoch:    2 train[loss: 0.000e+00]',
                            'final model saved in file: %s' % p.logdir])
      log_str = p.print_file.getvalue()
      self.assertIn(expected, log_str)
      self.assertEqual(12, sess.run(q_size))
      loom_inputs = sess.run(q_dequeue)
      self.assertEqual((12,), loom_inputs.shape)
      results = sess.run(p.compiler.output_tensors[0],
                         {p.compiler.loom_input_tensor: loom_inputs})
      self.assertEqual(results.tolist(), [7] * 12)

  def test_dequeue(self):
    p = plan.TrainPlan()
    p.compiler = block_compiler.Compiler().compile(blocks.Scalar())
    p.is_chief_trainer = True
    p.batch_size = 3
    p.batches_per_epoch = 2
    p.queue_capacity = 12
    p.num_dequeuers = 1
    p.ps_tasks = 1
    q = p._create_queue(0)
    p._setup_dequeuing([q])
    input_batch = list(p.compiler.build_loom_inputs([7])) * 3
    q_enqueue = q.enqueue_many([input_batch * 4])
    p.losses['foo'], = p.compiler.output_tensors
    p.train_op = tf.no_op()
    p.finalize_stats()
    p.logdir = self.get_temp_dir()
    p.epochs = 2
    p.print_file = six.StringIO()
    init_op = tf.global_variables_initializer()
    sv = p.create_supervisor()
    with self.test_session() as sess:
      sess.run(init_op)
      sess.run(q_enqueue)
      p.run(sv, sess)
    expected = '\n'.join(['running train',
                          'train_size: 6',
                          'epoch:    1 train[loss: 7.000e+00]',
                          'epoch:    2 train[loss: 7.000e+00]',
                          'final model saved in file: %s' % p.logdir])
    log_str = p.print_file.getvalue()
    self.assertIn(expected, log_str)


class EvalPlanTest(PlanTestBase):

  def test_create_from_flags(self):
    tf.flags.FLAGS.mode = plan.Plan.mode_keys.EVAL
    tf.flags.FLAGS.truncate_examples = 3
    p = plan.Plan.create_from_flags(_setup_plan(
        compiler=block_compiler.Compiler.create(blocks.Scalar()),
        losses={'foo': tf.constant(42.0)},
        examples=xrange(5)))
    self.assertEqual(p.logdir, os.path.join('/tmp/', 'plan', 'run_0', 'eval'))
    self.assertEqual(p.logdir_restore,
                     os.path.join('/tmp/', 'plan', 'run_0', 'train'))
    self.assertEqual(p.rundir, os.path.join('/tmp/', 'plan', 'run_0'))
    self.assertEqual(p.plandir, os.path.join('/tmp/', 'plan'))
    self.assertEqual([0, 1, 2], list(p.examples))
    self.assertEqual(p.compute_summaries, True)

  def test_create_from_params(self):
    params = plan.plan_default_params()
    params.update({
        'mode': plan.Plan.mode_keys.EVAL,
        'truncate_examples': 3})
    p = plan.Plan.create_from_params(_setup_plan(
        compiler=block_compiler.Compiler.create(blocks.Scalar()),
        losses={'foo': tf.constant(42.0)},
        examples=xrange(5)), params)
    self.assertEqual(p.logdir, os.path.join('/tmp/', 'plan', 'run_0', 'eval'))
    self.assertEqual(p.logdir_restore,
                     os.path.join('/tmp/', 'plan', 'run_0', 'train'))
    self.assertEqual(p.rundir, os.path.join('/tmp/', 'plan', 'run_0'))
    self.assertEqual(p.plandir, os.path.join('/tmp/', 'plan'))
    self.assertEqual([0, 1, 2], list(p.examples))
    self.assertEqual(p.compute_summaries, True)

  def test_assert_runnable(self):
    p = plan.EvalPlan()
    self.assertRaisesWithLiteralMatch(
        ValueError, 'compiler is required', p.assert_runnable)
    p.compiler = block_compiler.Compiler.create(blocks.Scalar())
    p.logdir = '/tmp/'
    self.assertRaisesWithLiteralMatch(
        ValueError, 'examples is required', p.assert_runnable)
    p.examples = xrange(5)
    self.assertRaisesWithLiteralMatch(
        ValueError, 'at least one loss is required', p.assert_runnable)
    p.losses['foo'] = tf.constant(42.0)
    p.finalize_stats()
    self.assertRaisesWithLiteralMatch(
        ValueError, 'logdir_restore is required', p.assert_runnable)
    p.logdir_restore = '/tmp/foo/'
    p.assert_runnable()

  def _make_plan(self):
    p = plan.EvalPlan()
    p.compiler = block_compiler.Compiler.create(blocks.Scalar())
    temp_dir = self.get_temp_dir()
    p.logdir = os.path.join(temp_dir, 'eval')
    p.logdir_restore = os.path.join(temp_dir, 'train')
    p.examples = [2, 4]
    p.print_file = six.StringIO()
    p.losses['loss'] = p.compiler.output_tensors[0]
    p.metrics['foo'] = tf.constant(42.0)
    p.finalize_stats()
    return p

  def test_run_once(self):
    p = self._make_plan()
    p.save_best = False
    # We aren't using a managed session, so we need to run this ourselves.
    init_op = tf.global_variables_initializer()
    sv = p.create_supervisor()
    with self.test_session() as sess:
      p.run(sv, sess)
      log_str = p.print_file.getvalue()
      self.assertTrue(
          log_str.endswith('could not restore from %s\n' % p.logdir_restore),
          msg=log_str)

      p.print_file = six.StringIO()
      sess.run(init_op)
      tf.gfile.MkDir(p.logdir_restore)
      save_path = os.path.join(p.logdir_restore, 'model')
      sv.saver.save(sess, save_path, global_step=42)
      p.run(sv, sess)
      log_str = p.print_file.getvalue()
      expected_lines = ['restoring from %s-42' % save_path,
                        'step:       0 loss: 3.000e+00 foo: 4.200e+01']
      expected = '\n'.join(expected_lines) + '\n'
      self.assertTrue(log_str.endswith(expected), msg=log_str)

  def test_run_loop(self):
    # This call to get_variable is used to mutate global state.
    tf.get_variable('global_step', [], tf.int32, tf.constant_initializer(42))
    p = self._make_plan()
    p.eval_interval_secs = 0.01
    p.should_stop = mock.Mock(side_effect=[False, False, True])
    # We aren't using a managed session, so we need to run this ourselves.
    init_op = tf.global_variables_initializer()
    sv = p.create_supervisor()
    with self.test_session() as sess:
      p.print_file = six.StringIO()
      sess.run(init_op)
      tf.gfile.MkDir(p.logdir_restore)
      save_path = os.path.join(p.logdir_restore, 'model')
      sv.saver.save(sess, save_path, global_step=42)
      m = mock.Mock()
      sv.SummaryComputed = m
      p.run(sv, sess)
      expected_summary = tf.Summary(value=[
          tf.Summary.Value(tag='foo', simple_value=42),
          tf.Summary.Value(tag='loss', simple_value=3)])
      m.assert_called_once_with(sess, expected_summary, global_step=42)
      log_str = p.print_file.getvalue()
      expected_lines = ['restoring from %s-42' % save_path,
                        'step:      42 loss: 3.000e+00 foo: 4.200e+01',
                        ('new best model saved in file: %s' %
                         os.path.join(p.logdir, 'model.ckpt-42'))]
      expected = '\n'.join(expected_lines) + '\n'
      self.assertTrue(log_str.endswith(expected), msg=log_str)

      p.should_stop = mock.Mock(side_effect=[False, False, True])
      p.examples = [4]
      p.print_file = six.StringIO()
      p.save_best = False
      p.run(sv, sess)
      log_str = p.print_file.getvalue()
      expected_lines = ['restoring from %s-42' % save_path,
                        'step:      42 loss: 4.000e+00 foo: 4.200e+01']
      expected = '\n'.join(expected_lines) + '\n'
      self.assertTrue(log_str.endswith(expected), msg=log_str)

  def test_print_metrics(self):
    p = self._make_plan()
    p.save_best = False
    p.metrics['bar'] = tf.constant(43.0)
    p.metrics['baz'] = tf.constant([44.0, 45.0])
    p.metrics['qux'] = tf.constant([46.0, 47.0])
    p.metrics['quux'] = tf.constant(48.0)
    init_op = tf.global_variables_initializer()
    sv = p.create_supervisor()
    with self.test_session() as sess:
      p.print_file = six.StringIO()
      sess.run(init_op)
      tf.gfile.MkDir(p.logdir_restore)
      save_path = os.path.join(p.logdir_restore, 'model')
      sv.saver.save(sess, save_path, global_step=42)
      p.run(sv, sess)
      log_str = p.print_file.getvalue()
      expected_lines = [
          'restoring from %s-42' % save_path,
          'step:       0 loss: 3.000e+00 foo: 4.200e+01 bar: 4.300e+01',
          'baz:',
          '[4.400e+01 4.500e+01]',
          'qux:',
          '[4.600e+01 4.700e+01]',
          'quux: 4.800e+01']
      expected = '\n'.join(expected_lines) + '\n'
      self.assertTrue(log_str.endswith(expected), msg=log_str)


class InferPlanTest(PlanTestBase):

  def test_create_from_flags(self):
    tf.flags.FLAGS.mode = plan.Plan.mode_keys.INFER
    tf.flags.FLAGS.truncate_examples = 3
    p = plan.Plan.create_from_flags(_setup_plan(
        compiler=block_compiler.Compiler.create(blocks.Scalar()),
        examples=xrange(5),
        infer_from='/foo',
        outputs=tf.constant(42)))
    self.assertEqual('/foo', p.infer_from)

  def test_assert_runnable(self):
    p = plan.InferPlan()
    self.assertRaisesWithLiteralMatch(
        ValueError, 'compiler is required', p.assert_runnable)
    p.compiler = block_compiler.Compiler.create(blocks.Scalar())
    p.logdir = '/tmp/'
    self.assertRaisesWithLiteralMatch(
        ValueError, 'examples is required', p.assert_runnable)
    p.examples = xrange(5)
    self.assertRaisesWithLiteralMatch(
        ValueError, 'outputs is required', p.assert_runnable)
    p.outputs = tf.placeholder(tf.float32)
    p.assert_runnable()
    p.batch_size = None
    self.assertRaisesWithLiteralMatch(
        ValueError, 'batch_size is required', p.assert_runnable)

  def test_run_no_key_fn(self):
    p = plan.InferPlan()
    p.compiler = block_compiler.Compiler.create(
        blocks.Scalar() >> blocks.Function(tf.negative))
    p.logdir = self.get_temp_dir()
    p.examples = xrange(5)
    p.outputs = p.compiler.output_tensors
    results = []
    p.results_fn = results.append
    p.batch_size = 3
    p.chunk_size = 2
    with self.test_session() as sess:
      p.run(session=sess)
    self.assertEqual(1, len(results))
    self.assertEqual([(0,), (-1,), (-2,), (-3,), (-4,)], list(results[0]))

  def test_run_key_fn(self):
    p = plan.InferPlan()
    p.compiler = block_compiler.Compiler.create(
        blocks.Scalar() >> blocks.Function(tf.negative))
    p.logdir = self.get_temp_dir()
    p.examples = xrange(5)
    p.outputs = p.compiler.output_tensors
    results = []
    p.results_fn = results.append
    p.key_fn = str
    p.batch_size = 3
    p.chunk_size = 2
    with self.test_session() as sess:
      p.run(session=sess)
    self.assertEqual(1, len(results))
    self.assertEqual(
        [('0', (-0,)), ('1', (-1,)), ('2', (-2,)), ('3', (-3,)), ('4', (-4,))],
        list(results[0]))


class TfNthTest(test_lib.TestCase):

  def test_single_fn(self):
    with self.test_session() as sess:
      n = tf.placeholder(tf.int64, [])
      self.assertEqual(42, sess.run(plan._tf_nth([lambda: tf.constant(42)], n),
                                    {n: 0}))
      self.assertRaises(tf.errors.InvalidArgumentError, sess.run,
                        plan._tf_nth([lambda: tf.constant(42)], n), {n: 7})

  def test_n_fns(self):
    with self.test_session() as sess:
      n = tf.placeholder(tf.int64, [])
      nth = plan._tf_nth([lambda j=i: tf.constant(j) for i in xrange(4)], n)
      for idx in xrange(4):
        self.assertEqual(idx, sess.run(nth, {n: idx}))
      self.assertRaises(tf.errors.InvalidArgumentError, sess.run, nth, {n: -1})


def _setup_plan(**kwargs):
  def setup(p):
    for k, v in six.iteritems(kwargs):
      setattr(p, k, v)
  return setup


if __name__ == '__main__':
  test_lib.main()
