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

"""This is the low-level Loom API for [TensorFlow Fold](../index.md).

As a simple example here's a loom that lets you evaluate arbitrary trees of
element-wise adds and multiplies on floating point vectors of length 3.

```python
class BinaryLoomOp(loom.LoomOp):

  def __init__(self, type_shape, op):
    self._op = op
    super(BinaryLoomOp, self).__init__(
      [type_shape, type_shape], [type_shape])

  def instantiate_batch(self, inputs):
    return [self._op(inputs[0], inputs[1])]

# Set up some loom ops:

x_var = tf.Variable(tf.zeros(3, dtype='float64'), name='x')
y_var = tf.Variable(tf.zeros(3, dtype='float64'), name='y')
vec_3 = loom.TypeShape('float64', (3,))
vec_loom = loom.Loom(
  named_tensors={'x': x_var, 'y': y_var},
  named_ops={'add': BinaryLoomOp(vec_3, tf.add),
             'mul': BinaryLoomOp(vec_3, tf.mul)})

vec_3_out = vec_loom.output_tensor(vec_3)

loss = tf.nn.l2_loss(vec_3_out)

# Then, when parsing a particular example we can do something like:
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  # Loop over examples (the expression we build should depend on the example but
  # for simplicity we'll just build x + [1, 5, 8])
  weaver = vec_loom.make_weaver()
  x = weaver.x
  c = weaver(np.array([1, 5, 8], dtype='float64'))
  x_plus_c = weaver.add(x, c)
  # In this case vec_3_out will contain a 1x3 matrix whose rows is x+c
  print("loss=", loss.eval(feed_dict=weaver.build_feed_dict([x_plus_c])))

  # Note: you could also evaluate multiple expressions at once. For example:
  weaver = vec_loom.make_weaver()
  x = weaver.x
  y = weaver.y
  c_squared = weaver.add(weaver.mul(x, x), weaver.mul(y, y))
  x_plus_y = weaver.add(x, y)
  x_plus_y_squared = weaver.mul(x_plus_y, x_plus_y)
  print("loss=", loss.eval(
      feed_dict=weaver.build_feed_dict([c_squared, x_plus_y_squared])))
  # In this case vec_3_out will contain a 2x3 matrix whose rows are x^2+y^2 and
  # (x+y)^2 (with multiplication being component-wise.)
```

@@TypeShape
@@LoomOp
@@PassThroughLoomOp
@@Loom
@@Weaver
"""

# This is the entrypoint for importing the TensorFlow Fold Loom library.
# We suggest importing it as:
#   import tensorflow_fold.public.loom

## Regenerating the Docs
#
# Fold's API docs are extracted from the toplevel docstring of
# third_party.tensorflow_fold.public.blocks and docstrings from the other
# files that it refers to.
#

# pylint: disable=wildcard-import, unused-import
from tensorflow_fold.loom.loom import *
# pylint: enable=wildcard-import, unused-import
