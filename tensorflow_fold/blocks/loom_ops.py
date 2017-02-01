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
"""Loom Ops used by Fold high-level API. This is an internal module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# import google3
import tensorflow_fold.blocks.result_types as tdt
from tensorflow_fold.public import loom


def _get_typeshapes(tensor_ts):
  return [t._type_shape for t in tensor_ts]  # pylint: disable=protected-access


class TaggingPassThroughOp(loom.LoomOp):
  """A pass-through op that adds a tag to its output type.

  When constructing a Fold Compiler, its root output and metrics
  are routed through a tagging pass-through. This is necessary to
  ensure that output tensors of the same type are uniquely identifiable.
  """

  def __init__(self, passthrough_types, tags):
    self.passthrough_types = passthrough_types
    in_ts = _get_typeshapes(passthrough_types)
    out_ts = [loom.TypeShape(ts.dtype, ts.shape, tag)
              for ts, tag in zip(in_ts, tags)]
    super(TaggingPassThroughOp, self).__init__(in_ts, out_ts)

  def instantiate_batch(self, inputs):
    return inputs


class FuncallOp(loom.LoomOp):
  """Loom Op that wraps a Function."""

  def __init__(self, tf_fn, input_type, output_type):
    self.tf_fn = tf_fn
    in_ts = _get_typeshapes(input_type.terminal_types())
    out_ts = _get_typeshapes(output_type.terminal_types())
    super(FuncallOp, self).__init__(in_ts, out_ts)
    self._unflatten_inputs = None
    self._flatten_outputs = None
    if (isinstance(input_type, tdt.TupleType) and
        any(isinstance(t, tdt.TupleType) for t in input_type)):
      self._unflatten_inputs = input_type.unflatten
    if (not isinstance(output_type, tdt.TupleType) or
        any(isinstance(t, tdt.TupleType) for t in output_type)):
      self._flatten_outputs = output_type.flatten

  def instantiate_batch(self, inputs):
    if self._unflatten_inputs:
      inputs = self._unflatten_inputs(iter(inputs), None)
    outputs = self.tf_fn(*inputs)
    return self._flatten_outputs(outputs) if self._flatten_outputs else outputs
