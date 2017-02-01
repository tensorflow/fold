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
"""DeserializingWeaver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

# import google3
import tensorflow as tf
from tensorflow_fold.loom.weaver_op_base import RegisterWeaverOp

# This is a gross hack that (apparently) prevents Python from
# occasionally segfaulting at shutdown when unlinking dynamic
# libraries, possibly related to <https://goo.gl/aSx6Bi>.  We need to
# call some function tf.pywrap_tensorflow that munges protos, *before*
# we dlopen the deserializing weaver library (which also munges protos).
tf.pywrap_tensorflow.list_devices()

_deserializing_weaver = tf.load_op_library(os.path.join(
    tf.resource_loader.get_data_files_path(), '_deserializing_weaver_op.so'))
deserializing_weaver = _deserializing_weaver.deserializing_weaver

RegisterWeaverOp('DeserializingWeaver')
