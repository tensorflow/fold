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
"""Abstractions for TensorFlow Fold training, eval, and inference."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import inspect
import itertools
import os
import re
import sys
import time
# import google3
import numpy as np
import six
from six.moves import map  # pylint: disable=redefined-builtin
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.blocks import util

flags = tf.app.flags
FLAGS = flags.FLAGS


_OPTIMIZER_CLASSES = {
    'adadelta': tf.train.AdadeltaOptimizer,
    'adagradda': tf.train.AdagradDAOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adam': tf.train.AdamOptimizer,
    'ftrl': tf.train.FtrlOptimizer,
    'gradientdescent': tf.train.GradientDescentOptimizer,
    'momentum': tf.train.MomentumOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
}


def parse_spec(spec):
  """Parses a list of key values pairs.

  Args:
    spec: A comma separated list of strings of the form `<key>=<value>`.

  Raises:
    ValueError: If `spec` is malformed or contains duplicate keys.

  Returns:
    A dict.
  """
  params = {}
  for clause in spec.split(','):
    clause = clause.strip()
    if not clause:
      continue
    if '=' not in clause:
      raise ValueError('Clause "%s" doesn\'t contain an "=".' % clause)
    key, value = clause.split('=', 1)
    key = key.strip()
    value = value.strip()
    if key in params: raise ValueError('duplicate key %s' % key)
    try:
      params[key] = int(value)
    except ValueError:
      try:
        params[key] = float(value)
      except ValueError:
        params[key] = value
  return params


def build_optimizer_from_params(optimizer='adam', **kwargs):
  """Constructs an optimizer from key-value pairs.

  For example

  ```python
  build_optimizer_from_params('momentum', momentum=0.9, learning_rate=1e-3)
  ```
  creates a MomentumOptimizer with momentum 0.9 and learning rate 1e-3.

  Args:
    optimizer: The name of the optimizer to construct.
    **kwargs: Arguments for the optimizer's constructor.

  Raises:
    ValueError: If `optimizer` is unrecognized.
    ValueError: If `kwargs` sets arguments that optimizer doesn't have, or
      fails to set arguments the optimizer requires.

  Returns:
    A tf.train.Optimizer of the appropriate type.
  """
  optimizer_name = re.sub('_', '', optimizer).lower()
  if optimizer_name not in _OPTIMIZER_CLASSES:
    raise ValueError('Unrecognized optimizer: %s' % optimizer_name)
  optimizer_class = _OPTIMIZER_CLASSES[optimizer_name]

  # Get the argspec, and figure out which arguments are optional vs mandatory.
  constructor_spec = inspect.getargspec(optimizer_class.__init__)
  constructor_args = set(constructor_spec.args[1:])  # 'self' doesn't count.
  constructor_optional_args = set(name for name, value in zip(
      reversed(constructor_spec.args), reversed(constructor_spec.defaults)))
  constructor_mandatory_args = constructor_args.difference(
      constructor_optional_args)

  # Make sure all the args given are things this optimizer can consume.
  for arg in kwargs:
    if arg not in constructor_args:
      raise ValueError('The %s optimizer doesn\'t take an argument named %s' % (
          optimizer_name, arg))

  # Make sure all the mandatory arguments have been provided.
  for arg in constructor_mandatory_args:
    if arg not in kwargs:
      raise ValueError('The %s optimizer requires %s to be set.' % (
          optimizer_name, arg))

  # Finally, call the constructor:
  return optimizer_class(**kwargs)


def define_plan_flags(default_plan_name='plan', blacklist=None):
  """Defines all of the flags used by `td.Plan.create_from_flags()`.

  Args:
    default_plan_name: A default value for the `--plan_name` flag.
    blacklist: A set of string flag names to not define.
  """
  blacklist = frozenset([] if blacklist is None else blacklist)
  def _maybe_def(define, name, default, docstring):
    if name not in blacklist: define(name, default, docstring)
  _register_options(_maybe_def, default_plan_name)


def plan_default_params():
  """Returns a dict from plan option parameter names to their defaults."""
  params = {}
  def register(unused_flag_def, name, default, unused_docstring):
    params[name] = default
  _register_options(register)
  return params


def _register_options(register, default_plan_name='plan'):
  """Calls a function once for each option a plan needs.

  Arguments:
    register: Register is a function which takes four arguments, namely,
       1. The function you'd use to define the option as a flag.
       2. The name of the option.
       3. The default value for the option.
       4. A doc-string for the option.
    default_plan_name: A default value for the plan_name option.
  """
  register(
      flags.DEFINE_string,
      'mode', 'train', 'One of train or eval or infer.')
  # Flags for checkpoints and summaries.
  register(
      flags.DEFINE_string,
      'logdir_base', '/tmp/', 'Path base for checkpoints and summaries.')
  register(
      flags.DEFINE_string,
      'plan_name', default_plan_name, 'Plan name.')
  register(
      flags.DEFINE_integer,
      'run_id', 0, 'Run ID. Checkpoints and summaries are stored under '
      '`os.path.join(logdir_base, plan_name, \'run_\' + run_id)`. In train '
      'mode, you can set --run_id=-1 to create a new unique new run_id.')
  register(
      flags.DEFINE_integer,
      'save_summaries_secs', 120,
      'Time interval between computations of summaries for the event log. Set '
      'to zero to disable summaries.')
  # Flags for data processing
  register(
      flags.DEFINE_integer,
      'num_multiprocess_processes', None, 'Number of worked processes to use '
      'for multiprocessing when building loom inputs. Defaults to the cpu '
      'count. Zero to disable multiprocessing')
  register(
      flags.DEFINE_integer,
      'truncate_examples', 0,
      'If non-zero, truncates datasets to have at most this many examples.')
  # Flags for replication.
  register(
      flags.DEFINE_string,
      'master', '',
      'TensorFlow master to use.')
  # Flags for training.
  register(
      flags.DEFINE_string,
      'optimizer_spec', '',
      'An optimizer spec used to construct the optimizer. Default is ADAM.')
  register(
      flags.DEFINE_integer,
      'batch_size', 128,
      'Number of examples per batch.')
  register(
      flags.DEFINE_integer,
      'epochs', 20,
      'Number of training epochs. Zero to run forever.')
  register(
      flags.DEFINE_integer,
      'task', 0,
      'Task ID of the replica running the training.')
  register(
      flags.DEFINE_integer,
      'ps_tasks', 0,
      'Number of PS Tasks in the training job.')
  register(
      flags.DEFINE_integer,
      'worker_replicas', 1, 'Number of train worker replicas. When '
      'num_dequeuers is specified, num_queues will be '
      'min(num_dequeuers, ps_tasks), and worker_replicas must be at least '
      'num_queues + num_dequeuers to ensure that there is at least one '
      'enqueuing worker per queue.')
  register(
      flags.DEFINE_integer,
      'num_dequeuers', 0, 'Number of dequeuing workers. If specified, ps_tasks '
      'must be positive, num_queues will be min(num_dequeuers, ps_tasks), '
      'worker_replicas must be at least num_queues + num_dequeuers to ensure '
      'that there is at least one enqueuing worker per queue.')
  register(
      flags.DEFINE_integer,
      'queue_capacity', 0, 'Capacity for shared queues. If unspecified and '
      'num_dequeuers is specified, defaults to 4x batch_size.')
  register(
      flags.DEFINE_integer,
      'batches_per_epoch', 1000,
      'How many batches to consider an epoch when examples are absent. Has no '
      'effect when examples is present (because an epoch is defined as '
      'a full pass through the training examples). ')
  register(
      flags.DEFINE_integer,
      'save_model_secs', 120,
      'Time interval between creation of model checkpoints. Note that if a '
      'dev set is provided then we save the best performing models, and this '
      'is ignored.')
  # Flags for eval.
  register(
      flags.DEFINE_integer,
      'eval_interval_secs', 120,
      'Time interval between eval runs (when running in a loop). Set to zero '
      'to run a single eval and then exit.')
  register(
      flags.DEFINE_bool,
      'save_best', True,
      'Whether eval mode should save a checkpoint if this model has the best '
      'loss so far. This can be disabled for example when evaluating '
      'performance on multiple data sets.')
  # Flags for inference.
  register(
      flags.DEFINE_string,
      'infer_from', 'train',
      'Subdirectory to load the latest checkpoint from. Typically \'train\' '
      'or \'eval\'.')


class _Options(object):

  def __init__(self, params):
    self.__dict__.update(params)


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
    num_multiprocess_processes: Number of worker processes to use for
      multiprocessing loom inputs. Default (None) is the CPU count.
      Set to zero to disable multiprocessing.
    is_chief_trainer: A boolean indicating whether this is the chief training
          worker.
    name: A string; defaults to 'plan'.
    logdir: A string; used for saving/restoring checkpoints and summaries.
    rundir: A string; the parent directory of logdir, shared between training
      and eval jobs for the same model.
    plandir: A string; the parent directory of rundir, shared between many runs
      of different models on the same task.
    master: A string; Tensorflow master to use.
    save_summaries_secs: An integer; set to zero to disable summaries. In
      distributed training only the chief should set this to a non-zero value.
    print_file: A file to print logging messages to; defaults to stdout.
    should_stop: A callback to check for whether the training or eval jobs
      should be stopped.
    report_loss: A callback for training and eval jobs to report losses.
    report_done: A callback called by the eval jobs when they finish.
  """
  mode_keys = tf.contrib.learn.ModeKeys

  @classmethod
  def create(cls, mode):
    """Creates a plan.

    Args:
      mode: A string; 'train', 'eval', or 'infer'.

    Raises:
      ValueError: If `mode` is invalid.

    Returns:
      A Plan.
    """
    cases = {Plan.mode_keys.TRAIN: TrainPlan,
             Plan.mode_keys.EVAL: EvalPlan,
             Plan.mode_keys.INFER: InferPlan}
    if mode not in cases:
      raise ValueError('invalid mode %r not in %s' % (mode, sorted(cases)))
    return cases[mode]()

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
    return cls._create_from_options(setup_plan_fn, FLAGS)

  @classmethod
  def create_from_params(cls, setup_plan_fn, params):
    """Creates a plan from a dictionary.

    Args:
      setup_plan_fn: A unary function accepting a plan as its argument. The
        function must assign the following attributes:
         * compiler
         * examples (excepting when batches are being read from the loom
           input tensor in train mode e.g. by a dequeuing worker)
         * losses (in train/eval mode)
         * outputs (in infer mode)
      params: a dictionary to pull options from.

    Returns:
      A runnable plan with finalized stats.

    Raises:
      ValueError: If params are invalid.
    """
    return cls._create_from_options(setup_plan_fn, _Options(params))

  @classmethod
  def _create_from_options(cls, setup_plan_fn, options):
    """Creates a plan from the attributes of some object.

    Args:
      setup_plan_fn: A unary function accepting a plan as its argument. The
        function must assign the following attributes:
         * compiler
         * examples (excepting when batches are being read from the loom
           input tensor in train mode e.g. by a dequeuing worker)
         * losses (in train/eval mode)
         * outputs (in infer mode)
      options: a bunch to pull options from.

    Returns:
      A runnable plan with finalized stats.

    Raises:
      ValueError: If options are invalid.
    """
    device_fn = tf.train.replica_device_setter(options.ps_tasks)
    with tf.device(device_fn):
      plan = Plan.create(options.mode)
    if plan.mode != Plan.mode_keys.TRAIN:
      if options.ps_tasks != 0:
        raise ValueError('Can only specify ps_tasks in train mode.')
      if options.task != 0:
        raise ValueError('Can only specify task in train mode.')
      if options.num_dequeuers != 0:
        raise ValueError('Can only specify num_dequeuers in train mode.')
      if options.queue_capacity != 0:
        raise ValueError('Can only specify queue_capacity in train mode.')

    plan.save_summaries_secs = options.save_summaries_secs
    plan.num_multiprocess_processes = options.num_multiprocess_processes
    plan.plandir = os.path.join(options.logdir_base, options.plan_name)
    plan.master = options.master

    plan._apply_options_pre(options)  # pylint:disable=protected-access
    with tf.device(device_fn):
      setup_plan_fn(plan)
      if not plan.has_finalized_stats: plan.finalize_stats()
      plan.examples = _maybe_truncate(options, plan.examples)
      plan._apply_options_post(options)  # pylint:disable=protected-access
    plan.assert_runnable()  # internal consistency check
    return plan

  def __init__(self, mode):
    self._platform_google = 'google3' in globals()
    if self._platform_google:
      # pylint: disable=protected-access
      # To get valid line numbers.
      tf.logging._skip_log_prefix('log_and_print')
      # pylint: enable=protected-access
    self.mode = mode
    self.compiler = None
    self.examples = None
    self.metrics = collections.OrderedDict()
    self.losses = collections.OrderedDict()
    self.num_multiprocess_processes = None
    self.is_chief_trainer = False
    self.logdir = None
    self.rundir = None
    self.plandir = None
    self.master = ''
    self.save_summaries_secs = 120
    self.print_file = sys.stdout
    self._finalized = False
    self._loss_total = None
    self._summaries = None
    self._global_step = tf.contrib.framework.get_or_create_global_step()
    self._best_loss = None  # TODO(moshelooks): Store this in the graph.
    self._batch_size_ph = None

    # Callbacks:
    self.should_stop = lambda: False
    self.report_loss = lambda step, loss: None
    self.report_done = lambda: None

  def _apply_options_pre(self, options):
    """Applies pre-setup flags."""
    pass

  def _apply_options_post(self, options):
    """Applies post-setup flags."""
    pass

  def finalize_stats(self):
    """Finalizes metrics and losses. Gets/creates global_step if unset."""
    if self.has_finalized_stats:
      raise RuntimeError('finalize_stats() has already been called')
    self._finalized = True
    if self.compute_summaries:
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
      if self.compute_summaries:
        self._batch_size_ph = tf.placeholder(loss_dtype, [], name='batch_size')
    loss_sums = []
    for name, tensor in six.iteritems(self.losses):
      loss_sums.append(tf.reduce_sum(tensor))
      if self.compute_summaries:
        tf.summary.scalar(name, loss_sums[-1] / self._batch_size_ph)
    if loss_sums:
      self._loss_total = tf.add_n(loss_sums)
      # computing a loss total summary is redundant if there is only one loss
      if self.compute_summaries and len(loss_sums) > 1:
        tf.summary.scalar('loss_total', self.loss_total / self._batch_size_ph)
    if self.compute_summaries:
      self._summaries = tf.summary.merge_all()
      if self._summaries is None: self._summaries = tf.constant('')

  @property
  def compute_summaries(self):
    """A bool; whether or not summaries are being computed."""
    return bool(self.save_summaries_secs)

  @property
  def global_step(self):
    """The global step tensor."""
    return self._global_step

  @property
  def has_finalized_stats(self):
    return self._finalized

  @property
  def batch_size_placeholder(self):
    """A placeholder for normalizing loss summaries.

    Returns:
      A scalar placeholder if there are losses and finalize_stats() has been
      called, else None.
    """
    return self._batch_size_ph

  @property
  def loss_total(self):
    """A scalar tensor, or None.

    Returns:
      The total loss if there are losses and finalize_stats() has been called,
      else None.
    """
    return self._loss_total

  @property
  def summaries(self):
    """A scalar string tensor, or None.

    Returns:
      Merged summaries if compute_summaries is true and finalize_stats
      has been called, else None.
    """
    return self._summaries

  def init_loom(self, **loom_kwargs):
    """Initializes compilers's loom.

    The plan must have a compiler with a compiled root block and an
    uninitialized loom.

    In training mode this sets up enqueuing/dequeuing if num_dequeuers is
    non-zero. When enqueuing, no actual training is performed; the
    train op is to enqueue batches of loom inputs from `train_set`,
    typically for some other training worker(s) to dequeue from. When
    dequeuing, batches are read using a dequeue op, typically from a
    queue that some other training worker(s) are enqueuing to.

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
        logdir=self.logdir, summary_op=None, global_step=self.global_step,
        save_summaries_secs=self.save_summaries_secs,
        summary_writer=(tf.train.Supervisor.USE_DEFAULT
                        if self.compute_summaries else None))
    supervisor_kwargs.update(self._supervisor_kwargs())
    return tf.train.Supervisor(**supervisor_kwargs)

  def _supervisor_kwargs(self):
    raise NotImplementedError

  def _should_stop(self, supervisor):
    return supervisor.should_stop() or self.should_stop()

  def run(self, supervisor=None, session=None):
    """Runs the plan with `supervisor` and `session`.

    Args:
      supervisor: A TF supervisor, or None. If None, a supervisor is created by
        calling `self.create_supervisor()`.
      session: A TF session, or None. If None, a session is created by calling
        `session.managed_session(self.master)`. Will be installed as the default
        session while running the plan.

    Raises:
      ValueError: If the plan's attributes are invalid.
      RuntimeError: If the plan has metrics or losses, and `finalize_stats()`
        has not been called.
    """
    self.assert_runnable()
    if supervisor is None: supervisor = self.create_supervisor()
    if session is None:
      with supervisor.managed_session(self.master) as session:
        self.run(supervisor, session)
      return

    tf.gfile.MakeDirs(self.logdir)
    self.log_and_print('running %s' % self.mode)
    with session.as_default():
      if self.num_multiprocess_processes == 0:
        self._run(supervisor, session)
        return
      with self.compiler.multiprocessing_pool(
          self.num_multiprocess_processes):
        self._run(supervisor, session)

  def _run(self, supervisor, session):
    raise NotImplementedError

  def log_and_print(self, msg):
    print(msg, file=self.print_file)
    tf.logging.info(msg)
    if self._platform_google: tf.logging.flush()

  def _save_best(self, session, saver, loss, step):
    # TODO(moshelooks): Store best_loss for train/eval in the graph.
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
      step: The current global step.
      is_dev: Whether or not we are running on the dev set. If so we never
        compute summaries (because they would overwrite summaries from the
        training set).

    Returns:
      A (size, loss_total, metrics) tuple, or (None, None, None) if should_stop
      return True before the eval was completed.

    Raises:
      ValueError: if batches is empty.
    """
    compute_summaries = self.compute_summaries and not is_dev
    size = 0
    metrics = collections.OrderedDict.fromkeys(self.metrics, 0.0)
    loss = 0.0
    fetches = {'metrics': self.metrics, 'loss': self.loss_total}
    if compute_summaries:
      fetches['summaries'] = self.summaries
      summary_pb = tf.Summary()
      tag_value = {}
    for batch_size, feed_dict in batches:
      size += batch_size
      if compute_summaries:
        feed_dict[self.batch_size_placeholder] = batch_size
      if self._should_stop(supervisor): return None, None, None
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

  There are two primary training modes. When `examples` is present,
  batches are created in Python and passed to TF as feeds. In this
  case, `examples` must be non-empty. When `examples` is absent,
  batches are read directly from the compiler's loom_input_tensor. In
  the latter case, each batch must have exactly batch_size elements.

  Attributes:
    batches_per_epoch: An integer, or None; how many batches to
      consider an epoch when `examples` is absent. Has no effect on
      training when `examples` is present (because an epoch is defined
      as a full pass through the training set).
    dev_examples: An iterable of development (i.e. validation) examples,
      or None.
    train_op: An TF op, e.g. `Optimizer.minimize(loss)`, or None.
    train_feeds: A dict of training feeds, e.g. keep probability for dropout.
    epochs: An integer, or None.
    batch_size: An integer, or None.
    save_model_secs: An integer. Note that if a dev_examples is provided then we
      save the best performing models, and this is ignored.
    task: An integer. This is a different integer from the rest;
      ps_tasks, num_dequeuers, queue_capacity all indicate some form of
      capacity, whereas this guy is task ID.
    worker_replicas: An integer.
    ps_tasks: An integer.
    num_dequeuers: An integer.
    queue_capacity: An integer.
    optimizer_params: a dictionary mapping strings to optimizer arguments.
      Used only if train_op is not provided.
    exact_batch_sizes: A bool; if true, `len(examples) % batch_size`
      items from the training set will be dropped each epoch to ensure
      that all batches have exactly `batch_size` items. Default is false.
      Has no effect if batches are being read from the compiler's loom
      input tensor. Otherwise, if true, `examples` must have at least
      `batch_size` items (to ensure that the training set is non-empty).

  """

  def __init__(self):
    super(TrainPlan, self).__init__(Plan.mode_keys.TRAIN)
    self.dev_examples = None
    self.train_op = None
    self.train_feeds = {}
    self.epochs = None
    self.batch_size = None
    self.save_model_secs = 120
    self.task = 0
    self.worker_replicas = 0
    self.ps_tasks = 0
    self.num_dequeuers = 0
    self.queue_capacity = 0
    self.batches_per_epoch = None
    self.optimizer_params = {}
    self.exact_batch_sizes = False

  def _setup_enqueuing(self, queues, **loom_kwargs):
    """Sets up enqueuing to the approx. smallest (least full) of `queues`."""
    self.compiler.init_loom(loom_input_tensor=None, **loom_kwargs)
    input_tensor = self.compiler.loom_input_tensor
    fns = [lambda r=q: r.enqueue_many([input_tensor]) for q in queues]
    self.train_op = _tf_nth(fns, tf.argmin(_noised_q_sizes(queues), axis=0))
    self.losses.clear()
    self.losses['dummy'] = tf.constant(0.0)
    self.save_summaries_secs = 0
    self.dev_examples = None
    self.train_feeds.clear()
    self.save_model_secs = 0
    self.exact_batch_sizes = True

  def _setup_dequeuing(self, queues, **loom_kwargs):
    """Sets up dequeuing from the approx. largest (most full) of `queues`."""
    fns = [lambda r=q: r.dequeue_many(self.batch_size) for q in queues]
    q_op = _tf_nth(fns, tf.argmax(_noised_q_sizes(queues), axis=0))
    self.compiler.init_loom(loom_input_tensor=q_op, **loom_kwargs)
    self.examples = None
    if not self.dev_examples: self.num_multiprocess_processes = 0

  def _apply_options_pre(self, options):
    self.epochs = options.epochs
    self.batch_size = options.batch_size
    self.batches_per_epoch = options.batches_per_epoch
    self.save_model_secs = options.save_model_secs
    self.optimizer_params = parse_spec(options.optimizer_spec)

    run_id = options.run_id
    if run_id == -1:
      if tf.gfile.Exists(self.plandir):
        existing = set(tf.gfile.ListDirectory(self.plandir))
        for run_id in itertools.count():
          if 'run_%d' % run_id not in existing: break
      else:
        run_id = 0
    self.rundir = os.path.join(self.plandir, 'run_%d' % run_id)
    self.logdir = os.path.join(self.rundir, self.mode)
    self.task = options.task
    self.is_chief_trainer = (options.task == 0)
    if not self.is_chief_trainer: self.save_summaries_secs = 0
    self.worker_replicas = options.worker_replicas
    self.ps_tasks = options.ps_tasks
    self.num_dequeuers = options.num_dequeuers
    self.queue_capacity = options.queue_capacity

  def _create_queue(self, queue_id, ctor=tf.RandomShuffleQueue):
    # The enqueuing workers transform inputs into serialized loom
    # weaver messages, which are represented as strings.
    return ctor(
        capacity=self.queue_capacity or 4 * self.batch_size,
        min_after_dequeue=0, dtypes=[tf.string], shapes=[tf.TensorShape([])],
        shared_name='tensorflow_fold_plan_queue%s' % queue_id)

  def _num_queues(self):
    return min(self.ps_tasks, self.num_dequeuers)

  def _apply_options_post(self, options):
    if self.train_op is None: self.build_optimizer()
    self.dev_examples = _maybe_truncate(options, self.dev_examples)

  def build_optimizer(self):
    if self.train_op is not None: raise ValueError('train_op already exists')
    self.train_op = build_optimizer_from_params(
        **self.optimizer_params).minimize(self.loss_total,
                                          global_step=self.global_step)

  def finalize_stats(self):
    if not self.losses: raise ValueError('at least one loss is required')
    super(TrainPlan, self).finalize_stats()

  def init_loom(self, **loom_kwargs):
    if not self.num_dequeuers:
      if self.queue_capacity:
        raise ValueError('cannot specify queue_capacity without also '
                         'specifying num_dequeuers')
      return super(TrainPlan, self).init_loom(**loom_kwargs)
    if self.ps_tasks < 1:
      raise ValueError('must have at least one PS task; %s' % self.ps_tasks)
    min_worker_replicas = self._num_queues() + self.num_dequeuers
    if self.worker_replicas < min_worker_replicas:
      raise ValueError(
          'worker_replicas must be at least num_queues + num_dequeuers; '
          '%s vs. %s + %s = %s' % (self.worker_replicas, self._num_queues(),
                                   self.num_dequeuers, min_worker_replicas))
    if self.compiler is None: raise ValueError('compiler is required')
    if not self.batch_size: raise ValueError('batch_size is required')

    # The queue needs to live on PS to be shared across workers. All
    # workers create the same num_queues (this is min(ps_tasks,
    # num_dequeuers)) distinct queues, with are assigned sequentially
    # to the first num_queues PS tasks. At every training step, each
    # enqueuing worker selects a queue with approx. minimal size to
    # enqueue to (to balance data across queues). Similarly, each
    # dequeuing worker selects a queue with approx. maximal size to
    # dequeue from.
    queues = []
    for queue_id in xrange(self._num_queues()):
      with tf.device(tf.DeviceSpec(job='ps', task=queue_id)):
        queues.append(self._create_queue(queue_id))

    if self.compute_summaries:
      for q in queues:
        self.metrics['queue_sizes/%s' % q.name] = q.size()

    # First num_dequeuers tasks dequeue, the remainder enqueue.
    if self.task < self.num_dequeuers:
      self._setup_dequeuing(queues, **loom_kwargs)
      return False, True  # dequeuers need stats, but not examples
    self._setup_enqueuing(queues, **loom_kwargs)
    return True, False  # enqueuers need examples, but not stats

  def assert_runnable(self):
    if not self.losses: raise ValueError('at least one loss is required')
    super(TrainPlan, self).assert_runnable()
    if self.train_op is None: raise ValueError('train_op is required')
    if not self.batch_size: raise ValueError('batch_size is required')
    if not (self.examples or self.batches_per_epoch):
      raise ValueError('either examples or batches_per_epoch is required')

  def _supervisor_kwargs(self):
    return dict(is_chief=self.is_chief_trainer, save_model_secs=(
        0 if self.dev_examples else self.save_model_secs))

  def _run(self, supervisor, session):
    train_feed_dict = self.train_feeds.copy()
    train_fetches = {'train_op': self.train_op, 'loss': self.loss_total,
                     'step': self.global_step}
    if self.compute_summaries: train_fetches['summaries'] = self.summaries
    # The training loop is essentially the same regardless of whether
    # we are passing batches by feed dict or by loom input
    # tensor. There are a few minor differences:
    #
    # 1. By feed dict, we compute the size of the training set lazily,
    #    as we iterate over it in the first epoch. By input tensor, we
    #    calculate train_size as batch_size * batches_per_epoch.
    #
    # 2. By feed dict, we get the size of each batch by calling len()
    #    on it (since the last batch in the epoch may have less than
    #    batch_size elements). By input tensor, we require that every
    #    batch have exactly batch_size elements.
    #
    # 3. By feed dict we need to create batches of inputs, and feed
    #    them every time we run the train op (obviously).
    if self.examples:
      epochs, train_size = self._by_feed_dict(train_feed_dict)
    else:
      epochs, train_size = self._by_input_tensor(train_feed_dict)
    if self.dev_examples:
      # Memoize a generator of batches of (size, feed_dict) pairs.
      gen_dev_batches = util.epochs(
          ((len(batch), self.compiler.build_feed_dict(batch))
           for batch in util.group_by_batches(
               self.dev_examples, self.batch_size)), shuffle=False)

    for epoch, batches in enumerate(epochs, 1):
      train_loss = 0.0
      for _ in batches:
        if self._should_stop(supervisor): return
        results = session.run(train_fetches, train_feed_dict)
        train_loss += results['loss']
        if self.compute_summaries:
          supervisor.summary_computed(
              session, results['summaries'], results['step'])
      if train_size == 0:
        raise ValueError('examples must be non-empty')
      if self.exact_batch_sizes and epoch == 1:
        if train_size < self.batch_size:
          raise ValueError('when exact_batch_sizes is true, examples must have '
                           'at least batch_size items; %s vs. %s' % (
                               train_size, self.batch_size))
        train_size -= train_size % self.batch_size
      train_loss /= train_size
      self.report_loss(results['step'], train_loss)
      log_str = 'epoch:%5d train[loss: %.3e]' % (epoch, train_loss)
      if self.dev_examples:
        dev_size, dev_loss, dev_metrics = self._eval_batches(
            supervisor, session, next(gen_dev_batches), results['step'],
            is_dev=True)
        if dev_size is None: return  # should_stop returned true
        if epoch == 1: self.log_and_print('train_size: %d dev_size: %d' %
                                          (train_size, dev_size))
        log_str += ' dev[%s]' % _eval_str(dev_size, dev_loss, dev_metrics)
        self.log_and_print(log_str)
        self._save_best(session, supervisor.saver, dev_loss, results['step'])
      else:
        if epoch == 1: self.log_and_print('train_size: %d' % train_size)
        self.log_and_print(log_str)
    if not self.dev_examples and self.is_chief_trainer:
      save_path = os.path.join(self.logdir, 'model.ckpt')
      save_fname = supervisor.saver.save(
          session, save_path, global_step=results['step'])
      self.log_and_print('final model saved in file: %s' % save_fname)

  def _by_feed_dict(self, feed_dict):
    """Setup for reading training data from feed dictionaries."""
    def prepare_batches(shuffled):
      for batch in util.group_by_batches(shuffled, self.batch_size,
                                         truncate=self.exact_batch_sizes):
        feed_dict[self.compiler.loom_input_tensor] = batch
        if self.compute_summaries:
          feed_dict[self.batch_size_placeholder] = len(batch)
        yield
    examples, train_size = _lazy_length(self.examples)
    loom_inputs = self.compiler.build_loom_inputs(examples)
    epochs = map(prepare_batches, util.epochs(loom_inputs, self.epochs))
    return epochs, train_size

  def _by_input_tensor(self, feed_dict):
    """Setup for reading training data from the loom input tensor."""
    if self.compute_summaries:
      feed_dict[self.batch_size_placeholder] = self.batch_size
    counter = xrange(self.epochs) if self.epochs else itertools.count()
    epochs = (xrange(self.batches_per_epoch) for _ in counter)
    train_size = self.batches_per_epoch * self.batch_size
    return epochs, train_size


class _StreamingPlan(Plan):
  """Base class for eval and infer plans, which can stream their data."""

  def __init__(self, mode):
    super(_StreamingPlan, self).__init__(mode)
    self.batch_size = 10000

  def assert_runnable(self):
    super(_StreamingPlan, self).assert_runnable()
    if not self.examples: raise ValueError('examples is required')
    if self.batch_size is None: raise ValueError('batch_size is required')

  def _supervisor_kwargs(self):
    return dict(save_model_secs=0)


class EvalPlan(_StreamingPlan):
  """Plan class for evaluation.

  Attributes:
    eval_interval_secs: Time interval between eval runs (when running in a
      loop). Set to zero or None to run a single eval and then exit; in this
      case data will be streamed. Otherwise, data must fit in memory.
    save_best: A boolean determining whether to save a checkpoint if this model
      has the best loss so far.
    logdir_restore: A string or None; log directory for restoring checkpoints
      from.
    batch_size: An integer (defaults to 10,000); maximal number of
      examples to pass to a single call to `Session.run()`. When streaming,
      this is also the maximal number of examples that will be materialized
      in-memory.
  """

  def __init__(self):
    super(EvalPlan, self).__init__(Plan.mode_keys.EVAL)
    self.eval_interval_secs = None
    self.save_best = True
    self.logdir_restore = None

  def _apply_options_pre(self, options):
    self.eval_interval_secs = options.eval_interval_secs
    self.save_best = options.save_best
    self.rundir = os.path.join(self.plandir, 'run_%d' % options.run_id)
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
    if self.eval_interval_secs:
      gen_batches = util.epochs(batches, shuffle=False)  # memoize batches
      max_reported_step = 0
      while not (self._should_stop(supervisor) and max_reported_step > 0):
        start_time = time.time()
        if self._restore(supervisor, session):
          step = tf.train.global_step(session, self.global_step)
          if step > max_reported_step:
            max_reported_step = step
            results = self._eval_batches(
                supervisor, session, next(gen_batches), step)
            if results[0] is None: break  # should_stop returned true
            self._report_loss_and_save_best(supervisor, session, step, *results)
          else:
            self.log_and_print('not running eval because step=%s' % step)
        sleep_time = self.eval_interval_secs - (time.time() - start_time)
        if sleep_time > 0: time.sleep(sleep_time)
    elif self._restore(supervisor, session):
      step = tf.train.global_step(session, self.global_step)
      results = self._eval_batches(supervisor, session, batches, step)
      if results[0] is not None:
        self._report_loss_and_save_best(supervisor, session, step, *results)
    self.report_done()

  def _report_loss_and_save_best(self, supervisor, session, step,
                                 size, loss, metrics):
    self.log_and_print('step:%8d %s' % (step, _eval_str(size, loss, metrics)))
    self.report_loss(step, loss / size)
    if self.save_best: self._save_best(session, supervisor.saver, loss, step)

  def _restore(self, supervisor, session):
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

  def _apply_options_pre(self, options):
    self.rundir = os.path.join(self.plandir, 'run_%d' % options.run_id)
    self.logdir = os.path.join(self.rundir, options.infer_from)

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


def _maybe_truncate(options, iterable):
  if options.truncate_examples and iterable:
    return itertools.islice(iterable, options.truncate_examples)
  return iterable


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


def _tf_nth(fns, n):
  """Runs only the nth element of fns, where n is a scalar integer tensor."""
  cases = [(tf.equal(tf.constant(i, n.dtype), n), fn)
           for i, fn in enumerate(fns)]
  final_pred, final_fn = cases.pop()
  def default():
    with tf.control_dependencies([
        tf.Assert(final_pred, [n, len(fns)], name='nth_index_error')]):
      return final_fn()
  if len(fns) == 1: return default()
  return tf.case(cases, default)


def _noised_q_sizes(queues):
  """Returns the sizes of queues as a tensor + random std normal noise."""
  q_sizes = tf.stack([q.size() for q in queues])
  return tf.cast(q_sizes, tf.float32) + tf.random_normal(q_sizes.shape)
