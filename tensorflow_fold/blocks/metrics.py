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
"""Metrics for TensorFlow Fold."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import six
import tensorflow_fold.blocks.blocks as tdb
import tensorflow_fold.blocks.result_types as tdt


class Metric(tdb.Block):
  """A block that computes a metric.

  Metrics are used in Fold when the size of a model's output is not
  fixed, but varies as a function of the input data. They are also
  handy for accumulating results across sequential and recursive
  computations without having the thread them through explicitly as
  return values.

  For example, to create a block `y` that takes a (label, prediction)
  as input, adds an L2 `'loss'` metric, and returns the prediction as
  its output, you could say:

  ```python
  y = Composition()
  with y.scope():
    label = y.input[0]
    prediction = y.input[1]
    l2 = (Function(tf.sub) >> Function(tf.nn.l2_loss)).reads(label, prediction)
    Metric('loss').reads(l2)
    y.output.reads(prediction)
  ```

  The input type of the block must be a `TensorType`, or a
  `(TensorType, PyObjectType)` tuple.
  The output type is always `VoidType`. In the tuple input case, the
  second item of the tuple becomes a label for the tensor value, which
  can be used to identify where the value came from in a nested data
  structure and/or batch of inputs.

  For example:

  ```python
  sess = tf.InteractiveSession()
  # We pipe Map() to Void() because blocks with sequence output types
  # cannot be compiled.
  block = td.Map(td.Scalar() >> td.Metric('foo')) >> td.Void()
  compiler = td.Compiler.create(block)
  sess.run(compiler.metric_tensors['foo'],
           compiler.build_feed_dict([range(3), range(4)])) =>
    array([ 0.,  1.,  2.,  0.,  1.,  2.,  3.], dtype=float32)
  ```

  Or with labels:

  ```python
  sess = tf.InteractiveSession()
  block = td.Map((td.Scalar(), td.Identity()) >> td.Metric('bar')) >> td.Void()
  compiler = td.Compiler.create(block)
  feed_dict, metric_labels = compiler.build_feed_dict(
      [[(0, 'zero'), (1, 'one')], [(2, 'two')]],
      metric_labels=True)
  metric_labels  =>  {'bar': ['zero', 'one', 'two']}
  sess.run(compiler.metric_tensors['bar'], feed_dict)  =>
      array([ 0.,  1.,  2.], dtype=float32)
  ```
  """

  def __init__(self, metric_name):
    if not isinstance(metric_name, six.string_types):
      raise TypeError('metric_name must be a string: %s' % (metric_name,))
    self._metric_name = metric_name
    super(Metric, self).__init__(name=str(metric_name),
                                 output_type=tdt.VoidType())

  _expected_input_types = (tdt.TensorType, tdt.TupleType)

  def _update_input_type(self):
    if isinstance(self.input_type, tdt.TupleType):
      if len(self.input_type) != 2:
        raise TypeError('metric tuple input must have 2 items: %s' %
                        self.input_type)
      if not isinstance(self.input_type[0], tdt.TensorType):
        raise TypeError('expected a tensor type, saw: %s' % self.input_type[0])
      if not isinstance(self.input_type[1], tdt.PyObjectType):
        raise TypeError('expected a pyobj type, saw: %s' % self.input_type[1])
      self._evaluate = self._evaluate_labeled
      self._metric_type = self.input_type[0]
    else:
      self._evaluate = self._evaluate_unlabeled
      self._metric_type = self.input_type

  def _compile(self, compiler_ctx):
    compiler_ctx.register_metric_op(self._metric_name, self._metric_type)

  def _evaluate_labeled(self, eval_ctx, x):
    eval_ctx.add_output(eval_ctx.op(self._metric_name, [x[0]])[0])
    eval_ctx.metric_labels[self._metric_name].append(x[1])

  def _evaluate_unlabeled(self, eval_ctx, x):
    eval_ctx.add_output(eval_ctx.op(self._metric_name, [x])[0])
