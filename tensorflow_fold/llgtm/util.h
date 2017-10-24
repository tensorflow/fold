/* Copyright 2017 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_FOLD_LLGTM_UTIL_H_
#define TENSORFLOW_FOLD_LLGTM_UTIL_H_

#include <iterator>
#include <limits>
#include <memory>
#include "tensorflow_fold/llgtm/tensor.h"

namespace llgtm {

// Wraps an iterator by applying a function to each element.
// The function is called on-demand, essentially implementing a lazy
// functional-style map over a collection of elements.  The class F must
// support an operator() that returns a reference to T.
// Note that the wrapped iterator is an input iterator only.
template<typename T, typename IterType, typename F>
class IteratorWrapper : public std::iterator<std::input_iterator_tag, T> {
 public:
  // Creates an IteratorWrapper from an iterator and a function f.
  explicit IteratorWrapper(IterType iter, F f = F())
      : iter_(iter), function_(f) {}

  IteratorWrapper() = default;
  IteratorWrapper(const IteratorWrapper& p) = default;
  IteratorWrapper(IteratorWrapper&& p) = default;

  IteratorWrapper& operator=(const IteratorWrapper& p) = default;
  IteratorWrapper& operator=(IteratorWrapper&& p) = default;

  T& operator*() { return function_(*iter_); }
  const T& operator*() const { return function_(*iter_); }

  T* operator->() { return &function_(*iter_); }
  const T* operator->() const { return &function_(*iter_); }

  IteratorWrapper& operator++() {
    ++iter_;
    return *this;
  }

  IteratorWrapper operator++(int) {
    return IteratorWrapper(iter_++, function_);
  }

  bool operator==(IteratorWrapper other) { return iter_ == other.iter_; }
  bool operator!=(IteratorWrapper other) { return iter_ != other.iter_; }

 private:
  IterType iter_;
  F function_;
};

// A functor that calls get() on a unique_ptr.
// For use with IteratorWrapper, to produce iterators over containers that
// don't expose the implementation details (ownership) of the container.
template<class T>
struct UniquePtrGetter {
  T& operator()(std::unique_ptr<T>& ptr) { return *ptr; }  // NOLINT
  const T& operator()(const std::unique_ptr<T>& ptr) { return *ptr; }
};


// A helper class that can calculate mean, stddev, and variance of a
// distribution. Uses Welford's algorithm.
// See also: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance.
template<typename DT>
class OnlineStats {
 public:
  // Initialize with data.
  explicit OnlineStats(const DT* data, size_t num_elements) {
    reset();

    for (int i = 0; i < num_elements; ++i) {
      this->add(data[i]);
    }
  }

  OnlineStats() : OnlineStats(/*data=*/ nullptr, 0) {}

  void add(DT element) {
    ++n_;

    // Welford's algorithm (to compute variance).
    double d1 = element - mean_;
    mean_ += d1 / n_;
    double d2 = element - mean_;
    m2_ += d1 * d2;
  }

  void reset() {
    n_ = 0;
    mean_ = m2_ = 0.0;
    min_ = std::numeric_limits<DT>::max();
    max_ = std::numeric_limits<DT>::min();
  }

  double stddev() {
    return std::sqrt(this->variance());
  }

  double variance() {
    return m2_ / n_;
  }

  double mean() {
    return mean_;
  }

  DT min() {
    return min_;
  }

  DT max() {
    return max_;
  }

 private:
  int n_;
  double mean_;
  double m2_;
  DT min_;
  DT max_;
};

}  // end namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_UTIL_H_
