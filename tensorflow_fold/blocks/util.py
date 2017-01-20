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
"""Util code for TensorFlow Fold."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import random
# import google3
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin


class EdibleIterator(collections.Iterator):
  """A wrapper around an iterator that lets it be used as a TF feed value.

  Edible iterators are useful when you have an expensive computation
  running asynchronously that you want to feed repeatedly. For
  example:

  ```python
  items = my_expensive_function()  # returns an iterator, doesn't block
  fd = {x: list(items)} # blocks here
  while training:
    do_stuff()  # doesn't run until my_expensive_function() completes
    sess.run(fetches, fd)
  ```

  With an edible iterator you can instead say:

  ```python
  items = my_expensive_function()  # returns an iterator, doesn't block
  fd = {x: EdibleIterator(items)} # doesn't block
  while training:
    do_stuff()  # runs right away
    sess.run(fetches, fd)  # blocks here
  ```

  Python iterators are only good for a single traversal. This means
  that if you call `next()` before a feed dict is created, then that
  value will be lost. When an edible iterator gets fed to TF the base
  iterator is exhausted, but the results are cached, so the same
  edible may be consumed repeatedly.

  Implementation details: TF consumes feed values by converting them
  to NumPy arrays. NumPy doesn't like iterators (otherwise we could
  e.g use tee(), which would be simpler), so we use `__array__` to
  tell NumPy how to do the conversion.
  """

  def __init__(self, iterable):
    self._iterator = iter(iterable)
    self._value = None

  def __next__(self):
    return next(self._iterator)

  next = __next__

  def __array__(self, dtype=None):
    """NumPy array protocol; returns iterator values as an ndarray."""
    if self._value is None:
      # Call fromiter if we can; it is faster and avoids the extra
      # copy, but doesn't support object types and requires a dtype.
      if dtype is None or dtype.hasobject:
        self._value = np.array(list(self._iterator), dtype)
      else:
        self._value = np.fromiter(self._iterator, dtype)
    return self._value

  @property
  def value(self):
    """Returns iterator values as an ndarray if it exists, else None."""
    return self._value


def group_by_batches(iterable, batch_size, truncate=False):
  """Yields successive batches from an iterable, as lists.

  Args:
    iterable: An iterable.
    batch_size: A positive integer.
    truncate: A bool (default false). If true, then the last
      `len_iterable % batch_size` items are not yielded, ensuring that
      all batches have exactly `batch_size` items.

  Yields:
    Successive batches from `iterable`, as lists of at most
    `batch_size` items.

  Raises:
    ValueError: If `batch_size` is non-positive.

  """
  iterator = iter(iterable)
  if batch_size <= 0:
    raise ValueError('batch_size must be positive: %s' % (batch_size,))
  if truncate:
    for batch in zip(*(iterator,) * batch_size):
      yield list(batch)
  else:
    batch_size -= 1
    for x in iterator:
      yield list(itertools.chain((x,), itertools.islice(iterator, batch_size)))


def epochs(items, n=None, shuffle=True, prng=None):
  """Yields the items of an iterable repeatedly.

  This function is particularly useful when `items` is expensive to compute
  and you want to memoize it without blocking. For example:

  ```python
  for items in epochs((my_expensive_function(x) for x in inputs), n):
    for item in items:
      f(item)
  ```

  This lets `f(item)` run as soon as the first item is ready.

  As an optimization, when n == 1 items itself is yielded without
  memoization.

  Args:
    items: An iterable.
    n: How many times to yield; zero or None (the default) means loop forever.
    shuffle: Whether or not to shuffle the items after each yield. Shuffling is
      performed in-place. We don't shuffle before the first yield because this
      would require us to block until all of the items were ready.
    prng: Nullary function returning a random float in [0.0, 1.0); defaults
      to `random.random`.

  Yields:
    An iterable of `items`, `n` times.

  Raises:
    TypeError: If `items` is not an iterable.
    ValueError: If `n` is negative.

  """
  if not isinstance(items, collections.Iterable):
    raise TypeError('items must be an iterable: %s' % (items,))
  if n is not None and n < 0:
    raise ValueError('n must be non-negative: %s' % (n,))
  if n == 1:
    yield items
    return
  first, second = itertools.tee(items)
  yield first
  items = list(second)
  if shuffle: random.shuffle(items, prng)
  for _ in xrange(n - 1) if n else itertools.count():
    yield items
    if shuffle: random.shuffle(items, prng)
