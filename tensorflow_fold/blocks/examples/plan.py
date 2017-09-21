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
"""Abstractions for TensorFlow Fold training, eval, and inference.

This is a skeleton version of the original plan.py and only used by example
code. These features are removed from the old plan.py:

1. Distributed training with queues and multiple workers

2. Continuous eval with --eval_interval_secs

3. Flags for specifying optimization and learning rate algorithms
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import os
import sys
# import google3
import numpy as np
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import util


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string(
    'mode', 'train', 'One of train or eval or infer.')
# Flags for checkpoints and summaries.
flags.DEFINE_string(
    'logdir_base', '/tmp/', 'Path base for checkpoints and summaries.')
flags.DEFINE_integer(
    'run_id', 0, 'Run ID. Checkpoints and summaries are stored under '
    '`os.path.join(logdir_base, \'run_\' + run_id)`. In train '
    'mode, you can set --run_id=-1 to create a new unique new run_id.')
# Flags for training.
flags.DEFINE_integer(
    'learning_rate_decay_steps', 10000,
    'The decay steps to use in exponential learning rate decay:'
    'decayed_rate=learning_rate * decay_rate ^ (global_step / decay_steps).')
flags.DEFINE_integer(
    'batch_size', 128,
    'Number of examples per batch.')
flags.DEFINE_integer(
    'epochs', 20,
    'Number of training epochs. Zero to run forever.')
# Flags for inference.
flags.DEFINE_string(
    'infer_from', 'train',
    'Subdirectory to load the latest checkpoint from. Typically \'train\' '
    'or \'eval\'.')


class Plan(object):
  """Base class for training, evaluation, and inference plans.

  Attributes:
    mode: One of 'train', 'eval', or 'infer'.
    compiler: A `td.Compiler`, or None.
    examples: An iterable of examples, or None.
    metrics: An ordered dict from strings to real numeric tensors. These are
      used to make scalar summaries if they are scalars and histogram summaries
      otherwise.
    losses: An ordered dict from strings to tensors.
    logdir: A string; used for saving/restoring checkpoints and summaries.
    rundir: A string; the parent directory of logdir, shared between training
      and eval jobs for the same model.
  """
  mode_keys = tf.contrib.learn.ModeKeys

  @classmethod
  def create_from_flags(cls, setup_plan_fn):
    """Creates a plan from flags.

    Args:
      setup_plan_fn: A unary function accepting a plan as its argument. The
        function must assign the following attributes:
         * compiler
         * examples (excepting when batches are being read from the loom
           input tensor in train mode e.g. by a dequeuing worker)
         * losses (in train/eval mode)
         * outputs (in infer mode)

    Returns:
      A runnable plan with finalized stats.

    Raises:
      ValueError: If flags are invalid.
    """
    device_fn = tf.train.replica_device_setter()
    with tf.device(device_fn):
      cases = {Plan.mode_keys.TRAIN: TrainPlan,
               Plan.mode_keys.EVAL: EvalPlan,
               Plan.mode_keys.INFER: InferPlan}
      if FLAGS.mode not in cases:
        raise ValueError('invalid mode %r not in %s' % (FLAGS.mode,
                                                        sorted(cases)))
      plan = cases[FLAGS.mode]()

    with tf.device(device_fn):
      setup_plan_fn(plan)
      if not plan.has_finalized_stats: plan.finalize_stats()
      plan._apply_options_post()  # pylint:disable=protected-access
    plan.assert_runnable()  # internal consistency check
    return plan

  def __init__(self, mode):
    self.mode = mode
    self.compiler = None
    self.examples = None
    self.metrics = collections.OrderedDict()
    self.losses = collections.OrderedDict()
    self.logdir = None
    self.rundir = None
    self.batch_size = FLAGS.batch_size
    self.epochs = FLAGS.epochs
    self._finalized = False
    self._loss_total = None
    self._summaries = None
    self._global_step = tf.contrib.framework.get_or_create_global_step()
    self._best_loss = None
    self._batch_size_ph = None

  def _apply_options_post(self):
    """Applies post-setup flags."""
    pass

  def finalize_stats(self):
    """Finalizes metrics and losses. Gets/creates global_step if unset."""
    if self.has_finalized_stats:
      raise RuntimeError('finalize_stats() has already been called')
    self._finalized = True
    for name, tensor in six.iteritems(self.metrics):
      if tensor.get_shape().ndims == 0:
        tf.summary.scalar(name, tensor)
      else:
        tf.summary.histogram(name, tensor)
    if self.losses:
      loss_dtype = next(six.itervalues(self.losses)).dtype
      if not loss_dtype.is_floating:
        raise TypeError('invalid loss dtype %r, must be a floating point type'
                        % loss_dtype)
      self._batch_size_ph = tf.placeholder(loss_dtype, [], name='batch_size')
    loss_sums = []
    for name, tensor in six.iteritems(self.losses):
      loss_sums.append(tf.reduce_sum(tensor))
      tf.summary.scalar(name, loss_sums[-1] / self._batch_size_ph)
    if loss_sums:
      self._loss_total = tf.add_n(loss_sums)
      # computing a loss total summary is redundant if there is only one loss
      if len(loss_sums) > 1:
        tf.summary.scalar('loss_total', self._loss_total / self._batch_size_ph)
    self._summaries = tf.summary.merge_all()
    if self._summaries is None: self._summaries = tf.constant('')

  @property
  def has_finalized_stats(self):
    return self._finalized

  def init_loom(self, **loom_kwargs):
    """Initializes compilers's loom.

    The plan must have a compiler with a compiled root block and an
    uninitialized loom.

    Args:
     **loom_kwargs: Arguments to `compiler.init_loom`. In enqueuing
        or dequeuing training `loom_input_tensor` may not be specified.

    Returns:
      A pair of two bools `(needs_examples, needs_stats)`, indicating
      which of these requirements must be met in order for the plan to
      be runnable. In enqueuing training and in inference we need examples
      but not stats, whereas in dequeuing the obverse holds. In all
      other cases we need both examples and stats.

    Raises:
      ValueError: If `compiler` is missing.
      RuntimeError: If `compile()` has not been called on the compiler.
      RuntimeError: If the compiler's loom has already been initialized.

    """
    if self.compiler is None: raise ValueError('compiler is required')
    self.compiler.init_loom(**loom_kwargs)
    return True, True  # this is overridden by train and infer plans

  def assert_runnable(self):
    """Raises an exception if the plan cannot be run."""
    if self.compiler is None: raise ValueError('compiler is required')
    if (self.metrics or self.losses) and not self._finalized:
      raise RuntimeError('finalize_stats() has not been called')
    if self.logdir is None: raise ValueError('logdir is required')

  def create_supervisor(self):
    """Creates a TF supervisor for running the plan."""
    self.assert_runnable()
    self.log_and_print('creating supervisor with logdir %s' % self.logdir)
    supervisor_kwargs = dict(
        logdir=self.logdir, summary_op=None, global_step=self._global_step,
        save_summaries_secs=120,
        summary_writer=tf.train.Supervisor.USE_DEFAULT)
    supervisor_kwargs.update(self._supervisor_kwargs())
    return tf.train.Supervisor(**supervisor_kwargs)

  def _supervisor_kwargs(self):
    raise NotImplementedError

  def run(self, supervisor=None, session=None):
    """Runs the plan with `supervisor` and `session`.

    Args:
      supervisor: A TF supervisor, or None. If None, a supervisor is created by
        calling `self.create_supervisor()`.
      session: A TF session, or None. If None, a session is created by calling
        `session.managed_session()`. Will be installed as the default
        session while running the plan.

    Raises:
      ValueError: If the plan's attributes are invalid.
      RuntimeError: If the plan has metrics or losses, and `finalize_stats()`
        has not been called.
    """
    self.assert_runnable()
    if supervisor is None: supervisor = self.create_supervisor()
    if session is None:
      with supervisor.managed_session() as session:
        self.run(supervisor, session)
      return

    tf.gfile.MakeDirs(self.logdir)
    self.log_and_print('running %s' % self.mode)
    with session.as_default():
      self._run(supervisor, session)

  def _run(self, supervisor, session):
    raise NotImplementedError

  def log_and_print(self, msg):
    print(msg, file=sys.stdout)
    tf.logging.info(msg)

  def _save_best(self, session, saver, loss, step):
    if self._best_loss is None or loss < self._best_loss:
      self._best_loss = loss
      save_path = os.path.join(self.logdir, 'model.ckpt')
      save_fname = saver.save(session, save_path, global_step=step)
      self.log_and_print('new best model saved in file: %s' % save_fname)

  def _eval_batches(self, supervisor, session, batches, step, is_dev=False):
    """Runs a batchwise eval.

    Args:
      supervisor: A TF supervisor.
      session: A TF session.
      batches: An iterable of (batch_size, feed_dict) pairs.
      step: The current global step. Used for computing summaries.
      is_dev: Whether or not we are running on the dev set. If so we never
        compute summaries (because they would overwrite summaries from the
        training set).

    Returns:
      A (size, loss_total, metrics) tuple.

    Raises:
      ValueError: if batches is empty.
    """
    compute_summaries = not is_dev
    size = 0
    metrics = collections.OrderedDict.fromkeys(self.metrics, 0.0)
    loss = 0.0
    fetches = {'metrics': self.metrics, 'loss': self._loss_total}
    if compute_summaries:
      fetches['summaries'] = self._summaries
      summary_pb = tf.Summary()
      tag_value = {}
    for batch_size, feed_dict in batches:
      size += batch_size
      if compute_summaries:
        feed_dict[self._batch_size_ph] = batch_size
      results = session.run(fetches, feed_dict)
      for name, value in six.iteritems(results['metrics']):
        metrics[name] += value * batch_size
      loss += results['loss']
      if compute_summaries:
        for value in tf.Summary.FromString(results['summaries']).value:
          if value.HasField('simple_value'):
            if value.tag not in tag_value:
              tag_value[value.tag] = summary_pb.value.add(tag=value.tag)
            tag_value[value.tag].simple_value += value.simple_value * batch_size
    if size == 0:
      raise ValueError('dev_examples must be non-empty' if is_dev else
                       'examples must be non-empty')
    for name in six.iterkeys(metrics):
      metrics[name] /= size
    if compute_summaries:
      for name in six.iterkeys(tag_value):
        tag_value[name].simple_value /= size
      supervisor.SummaryComputed(session, summary_pb, global_step=step)
    return size, loss, metrics


class TrainPlan(Plan):
  """Plan class for training.

  Attributes:
    dev_examples: An iterable of development (i.e. validation) examples,
      or None.
    train_op: An TF op, e.g. `Optimizer.minimize(loss)`, or None.
    train_feeds: A dict of training feeds, e.g. keep probability for dropout.
    epochs: An integer, or None.
    batch_size: An integer, or None.

  """

  def __init__(self):
    super(TrainPlan, self).__init__(Plan.mode_keys.TRAIN)
    self.dev_examples = None
    self.train_op = None
    self.train_feeds = {}
    self.rundir = os.path.join(FLAGS.logdir_base, 'run_%d' % FLAGS.run_id)
    self.logdir = os.path.join(self.rundir, self.mode)

  def _apply_options_post(self):
    if self.train_op is None: self.build_optimizer()

  def build_optimizer(self):
    if self.train_op is not None: raise ValueError('train_op already exists')
    self.train_op = tf.train.AdamOptimizer().minimize(
        self._loss_total, global_step=self._global_step)

  def finalize_stats(self):
    if not self.losses: raise ValueError('at least one loss is required')
    super(TrainPlan, self).finalize_stats()

  def assert_runnable(self):
    if not self.losses: raise ValueError('at least one loss is required')
    super(TrainPlan, self).assert_runnable()
    if self.train_op is None: raise ValueError('train_op is required')
    if not self.batch_size: raise ValueError('batch_size is required')
    if not self.examples:
      raise ValueError('examples is required')

  def _supervisor_kwargs(self):
    return dict(is_chief=True, save_model_secs=(
        0 if self.dev_examples else 120))

  def _run(self, supervisor, session):
    train_feed_dict = self.train_feeds.copy()
    train_fetches = {'train_op': self.train_op, 'loss': self._loss_total,
                     'step': self._global_step}
    train_fetches['summaries'] = self._summaries
    epochs, train_size = self._by_feed_dict(train_feed_dict)
    if self.dev_examples:
      # Memoize a generator of batches of (size, feed_dict) pairs.
      gen_dev_batches = util.epochs(
          ((len(batch), self.compiler.build_feed_dict(batch))
           for batch in util.group_by_batches(
               self.dev_examples, self.batch_size)), shuffle=False)
      # If there is an existing checkpoint in logdir, and we are
      # saving the best model, calculate best_loss before doing any
      # training, so we don't potentially replace a better-performing
      # checkpoint with a worse one.
      ckpt = tf.train.get_checkpoint_state(self.logdir)
      if ckpt and ckpt.model_checkpoint_path:
        _, self._best_loss, _ = self._eval_batches(
            supervisor, session, next(gen_dev_batches), None, is_dev=True)
        if self._best_loss is None: return  # should_stop returned true

    for epoch, batches in enumerate(epochs, 1):
      self.log_and_print('Starting epoch %d.' % epoch)
      train_loss = 0.0
      for (k, _) in enumerate(batches):
        results = session.run(train_fetches, train_feed_dict)
        train_loss += results['loss']
        self.log_and_print('Batch %d: loss %f' % (k, results['loss']))
        supervisor.summary_computed(
            session, results['summaries'], results['step'])
      if train_size == 0:
        raise ValueError('examples must be non-empty')
      train_loss /= train_size
      log_str = 'epoch:%5d train[loss: %.3e]' % (epoch, train_loss)

      if self.dev_examples:
        dev_size, dev_loss, dev_metrics = self._eval_batches(
            supervisor, session, next(gen_dev_batches), results['step'],
            is_dev=True)
        if epoch == 1: self.log_and_print('train_size: %d dev_size: %d' %
                                          (train_size, dev_size))
        log_str += ' dev[%s]' % _eval_str(dev_size, dev_loss, dev_metrics)
        self._save_best(session, supervisor.saver, dev_loss, results['step'])
      else:
        if epoch == 1: self.log_and_print('train_size: %d' % train_size)
      self.log_and_print(log_str)

    if not self.dev_examples:
      save_path = os.path.join(self.logdir, 'model.ckpt')
      save_fname = supervisor.saver.save(
          session, save_path, global_step=results['step'])
      self.log_and_print('final model saved in file: %s' % save_fname)

  def _by_feed_dict(self, feed_dict):
    """Setup for reading training data from feed dictionaries."""
    def prepare_batches(shuffled):
      for batch in util.group_by_batches(shuffled, self.batch_size,
                                         truncate=False):
        feed_dict[self.compiler.loom_input_tensor] = batch
        feed_dict[self._batch_size_ph] = len(batch)
        yield
    examples, train_size = _lazy_length(self.examples)
    loom_inputs = self.compiler.build_loom_inputs(examples)
    epochs = map(prepare_batches, util.epochs(loom_inputs, self.epochs))
    return epochs, train_size


class _StreamingPlan(Plan):
  """Base class for eval and infer plans, which can stream their data."""

  def __init__(self, mode):
    super(_StreamingPlan, self).__init__(mode)
    self.batch_size = 10000

  def assert_runnable(self):
    super(_StreamingPlan, self).assert_runnable()
    if not self.examples and self.mode != Plan.mode_keys.INFER:
      raise ValueError('examples are required in non-infer mode.')
    if self.batch_size is None: raise ValueError('batch_size is required')

  def _supervisor_kwargs(self):
    return dict(save_model_secs=0)


class EvalPlan(_StreamingPlan):
  """Plan class for evaluation.

  Attributes:
    logdir_restore: A string or None; log directory for restoring checkpoints
      from.
    batch_size: An integer (defaults to 10,000); maximal number of
      examples to pass to a single call to `Session.run()`. When streaming,
      this is also the maximal number of examples that will be materialized
      in-memory.
  """

  def __init__(self):
    super(EvalPlan, self).__init__(Plan.mode_keys.EVAL)
    self.logdir_restore = None
    self.rundir = os.path.join(FLAGS.logdir_base, 'run_%d' % FLAGS.run_id)
    self.logdir_restore = os.path.join(self.rundir, Plan.mode_keys.TRAIN)
    self.logdir = os.path.join(self.rundir, self.mode)

  def assert_runnable(self):
    super(EvalPlan, self).assert_runnable()
    if not self.losses: raise ValueError('at least one loss is required')
    if not self.logdir_restore: raise ValueError('logdir_restore is required')

  def _run(self, supervisor, session):
    batches = (  # generates (size, feed_dict) pairs
        (len(batch), self.compiler.build_feed_dict(batch))
        for batch in util.group_by_batches(self.examples, self.batch_size))
    if self._restore(supervisor, session):
      step = tf.train.global_step(session, self._global_step)
      results = self._eval_batches(supervisor, session, batches, step)
      if results[0] is not None:
        self._report_loss_and_save_best(supervisor, session, step, *results)

  def _report_loss_and_save_best(self, supervisor, session, step,
                                 size, loss, metrics):
    self.log_and_print('step:%8d %s' % (step, _eval_str(size, loss, metrics)))
    self._save_best(session, supervisor.saver, loss, step)

  def _restore(self, supervisor, session):
    """Tries to restore a checkpoint, returns True on success."""
    ckpt = tf.train.get_checkpoint_state(self.logdir_restore)
    if ckpt and ckpt.model_checkpoint_path:
      self.log_and_print('restoring from %s' % ckpt.model_checkpoint_path)
      supervisor.saver.restore(session, ckpt.model_checkpoint_path)
      return True
    self.log_and_print('could not restore from %s' % self.logdir_restore)
    return False


class InferPlan(_StreamingPlan):
  """Plan class for inference.

  Attributes:
    key_fn: A function from examples to keys, or None.
    outputs: A list or tuple of tensors to be run to produce results, or None.
    results_fn: A function that takes an iterable of `(key, result)` pairs if
    `key_fn` is present or `result`s otherwise; by default prints to stdout.
    context_manager: A context manager for wrapping calls to `result_
    batch_size: An integer (defaults to 10,000); maximal number of
      examples to materialize in-memory.
    chunk_size: An integer (defaults to 100); chunk size for each unit
      of work, if multiprocessing.
  """

  def __init__(self):
    super(InferPlan, self).__init__(Plan.mode_keys.INFER)
    self.key_fn = None
    self.outputs = None
    self.results_fn = _default_results_fn
    self.chunk_size = 100
    self.rundir = os.path.join(FLAGS.logdir_base, 'run_%d' % FLAGS.run_id)
    self.logdir = os.path.join(self.rundir, FLAGS.infer_from)

  def init_loom(self, **loom_kwargs):
    super(InferPlan, self).init_loom(**loom_kwargs)
    return True, False  # infer plans need examples, but not stats

  def assert_runnable(self):
    super(InferPlan, self).assert_runnable()
    if self.chunk_size is None: raise ValueError('chunk_size is required')
    if self.outputs is None: raise ValueError('outputs is required')
    if self.results_fn is None: raise ValueError('results_fn is required')

  def _run(self, supervisor, session):
    return self.results_fn(itertools.chain.from_iterable(
        self._run_batch(examples, session)
        for examples in util.group_by_batches(self.examples, self.batch_size)))

  def _run_batch(self, examples, session):
    """Lazily runs `examples` with session."""
    loom_input_chunks = self.compiler.build_loom_input_batched(
        examples, self.chunk_size, ordered=True)
    result_chunks = (
        session.run(self.outputs, {self.compiler.loom_input_tensor: chunk})
        for chunk in loom_input_chunks)
    results = itertools.chain.from_iterable(
        zip(*chunk) for chunk in result_chunks)
    if self.key_fn:
      keys = map(self.key_fn, examples)  # pylint: disable=not-callable
      results = zip(keys, results)
    return results


def _default_results_fn(results):
  for result in results:
    print(*result)


def _lazy_length(iterable):
  """Lazily computes length as iteration proceeds; returns (iterator, count)."""
  count = np.zeros((), np.int64)
  def incrementing_identity(x, count=count):
    count += 1
    return x
  return map(incrementing_identity, iterable), count


def _format_items(items):
  formatter = lambda x: '%.3e' % x
  last_large_output = None
  for key, value in items:
    value = np.asarray(value)
    large_output = value.ndim >= 1
    # If there was a previous output, print a separator.
    if last_large_output is not None:
      yield '\n' if large_output or last_large_output else ' '
    format_string = '%s:\n%s' if large_output else '%s: %s'
    yield format_string % (key,
                           np.array2string(value, style=formatter,
                                           formatter={'float_kind': formatter}))
    last_large_output = large_output


def _eval_str(eval_size, loss, metrics):
  items = [('loss', loss / eval_size)]
  items.extend(metrics.items())
  return ''.join(_format_items(items))
