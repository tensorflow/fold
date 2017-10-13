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

"""High-level Blocks API for [TensorFlow Fold](../index.md).

## Compiler

@@Compiler

## Blocks for input

@@Tensor
@@Scalar
@@Vector
@@InputTransform
@@SerializedMessageToTree
@@OneHot
@@OneHotFromList
@@Optional

## Blocks for composition

@@Composition
@@Pipe
@@Record
@@AllOf

## Blocks for tensors

@@FromTensor
@@Function
@@Concat
@@Zeros

## Blocks for sequences

@@Map
@@Fold
@@RNN
@@Reduce
@@Sum
@@Min
@@Max
@@Mean
@@Broadcast
@@Zip
@@ZipWith
@@NGrams
@@Nth
@@GetItem
@@Length
@@Slice

## Other blocks

@@ForwardDeclaration
@@OneOf
@@Metric
@@Identity
@@Void

## Layers

@@FC
@@Embedding
@@FractalNet
@@ScopedLayer

## Types

@@TensorType
@@VoidType
@@PyObjectType
@@TupleType
@@SequenceType
@@BroadcastSequenceType

## Plans

@@Plan
@@TrainPlan
@@EvalPlan
@@InferPlan
@@define_plan_flags
@@plan_default_params

## Conversion functions

@@convert_to_block
@@convert_to_type
@@canonicalize_type

## Utilities

@@EdibleIterator
@@group_by_batches
@@epochs
@@parse_spec
@@build_optimizer_from_params
@@create_variable_scope

## Abstract classes

@@IOBase
@@Block
@@Layer
@@ResultType
"""

# This is the entrypoint for importing the TensorFlow Fold Blocks library.
# We suggest importing it as:
#   import tensorflow_fold.public.blocks as td

## Regenerating the Docs
#
# Fold's API docs are extracted from the toplevel docstring of
# third_party.tensorflow_fold.public.blocks and docstrings from the other
# files that it refers to.
#

# pylint: disable=wildcard-import, unused-import
from tensorflow_fold.blocks.block_compiler import *
from tensorflow_fold.blocks.blocks import *
from tensorflow_fold.blocks.layers import *
from tensorflow_fold.blocks.metrics import *
from tensorflow_fold.blocks.plan import *
from tensorflow_fold.blocks.result_types import *
from tensorflow_fold.blocks.util import *
# pylint: enable=wildcard-import, unused-import
