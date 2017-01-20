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
"""Common methods for testing TensorFlow Fold."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
# import google3
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow_fold.util import proto_tools

# pylint: disable=g-import-not-at-top,unused-import
if six.PY3:
  import unittest.mock as mock
else:
  import mock
# pylint: enable=g-import-not-at-top,unused-import

# Make sure SerializedMessageToTree can see our proto files.
proto_tools.map_proto_source_tree_path(
    '', os.getcwd())  # Tests run in the bazel root directory.
proto_tools.import_proto_file('tensorflow_fold/util/test.proto')
proto_tools.import_proto_file('tensorflow_fold/util/test3.proto')


class TestCase(tf.test.TestCase):

  def assertRaisesWithLiteralMatch(self, exception, literal, callable_obj,
                                   *args, **kwargs):
    with self.assertRaises(exception) as ctx:
      callable_obj(*args, **kwargs)
    self.assertEqual(str(ctx.exception), literal)

  # Open-sourced here:
  # <github.com/google/google-apputils/blob/master/google/apputils/basetest.py>
  def assertSameStructure(self, a, b, aname='a', bname='b', msg=None):
    """Asserts that two values contain the same structural content.

    The two arguments should be data trees consisting of trees of dicts and
    lists. They will be deeply compared by walking into the contents of dicts
    and lists; other items will be compared using the == operator.
    If the two structures differ in content, the failure message will indicate
    the location within the structures where the first difference is found.
    This may be helpful when comparing large structures.

    Args:
      a: The first structure to compare.
      b: The second structure to compare.
      aname: Variable name to use for the first structure in assertion messages.
      bname: Variable name to use for the second structure.
      msg: Additional text to include in the failure message.
    """
    # Accumulate all the problems found so we can report all of them at once
    # rather than just stopping at the first
    problems = []

    _walk_structure_for_problems(a, b, aname, bname, problems)

    # Avoid spamming the user toooo much
    max_problems_to_show = self.maxDiff // 80
    if len(problems) > max_problems_to_show:
      problems = problems[0:max_problems_to_show-1] + ['...']

    if problems:
      failure_message = '; '.join(problems)
      if msg:
        failure_message += (': ' + msg)
      self.fail(failure_message)


# Open-sourced here:
# <github.com/google/google-apputils/blob/master/google/apputils/basetest.py>
def _walk_structure_for_problems(a, b, aname, bname, problem_list):
  """The recursive comparison behind assertSameStructure."""
  if type(a) != type(b):  # pylint: disable=unidiomatic-typecheck
    problem_list.append('%s is a %r but %s is a %r' %
                        (aname, type(a), bname, type(b)))
    # If they have different types there's no point continuing
    return

  if isinstance(a, collections.Mapping):
    for k in a:
      if k in b:
        _walk_structure_for_problems(
            a[k], b[k], '%s[%r]' % (aname, k), '%s[%r]' % (bname, k),
            problem_list)
      else:
        problem_list.append('%s has [%r] but %s does not' % (aname, k, bname))
    for k in b:
      if k not in a:
        problem_list.append('%s lacks [%r] but %s has it' % (aname, k, bname))

  # Strings are Sequences but we'll just do those with regular !=
  elif (isinstance(a, collections.Sequence) and
        not isinstance(a, six.string_types)):
    minlen = min(len(a), len(b))
    for i in xrange(minlen):
      _walk_structure_for_problems(
          a[i], b[i], '%s[%d]' % (aname, i), '%s[%d]' % (bname, i),
          problem_list)
    for i in xrange(minlen, len(a)):
      problem_list.append('%s has [%i] but %s does not' % (aname, i, bname))
    for i in xrange(minlen, len(b)):
      problem_list.append('%s lacks [%i] but %s has it' % (aname, i, bname))

  else:
    if a != b:
      problem_list.append('%s is %r but %s is %r' % (aname, a, bname, b))


def main():
  tf.test.main()
