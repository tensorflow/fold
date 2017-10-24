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

#ifndef TENSORFLOW_FOLD_LLGTM_VARIABLE_INITIALIZERS_H_
#define TENSORFLOW_FOLD_LLGTM_VARIABLE_INITIALIZERS_H_

#include "tensorflow_fold/llgtm/dimensions.h"
#include "tensorflow_fold/llgtm/graph.h"

namespace llgtm {


// Initialize all elements of a variable with arbitrary dimensions to zero.
template<typename DT>
class ZerosInitializer {
 public:
  ZerosInitializer() {}

  Tensor<DT> operator()(Graph* g, const Dimensions& dims, DeviceID device) {
    return g->Zeros<DT>(dims, device);
  }
};


// Initialize all elements of a variable with arbitrary dimensions to a given
// scalar value.
template<typename DT>
class ScalarInitializer_ {
 public:
  explicit ScalarInitializer_(DT val) : val_(val) {}

  Tensor<DT> operator()(Graph* g, const Dimensions& dims, DeviceID device) {
    return g->ConstantFromScalar(dims, val_, device);
  }

 private:
  DT val_;
};

// Separate entry point for automatic template argument deduction.
template<typename DT>
ScalarInitializer_<DT> ScalarInitializer(DT val) {
  return ScalarInitializer_<DT>(val);
}


// Initialize all elements of a tensor variable using an array of values. The
// array must be at least as big as the variable.
template<typename DT>
class TensorInitializer_ {
 public:
  explicit TensorInitializer_(DT *f) : val_(f) {}

  Tensor<DT> operator()(Graph *g, const Dimensions& dims, DeviceID device) {
    return g->Value(dims, val_, device);
  }

 private:
  DT *val_;
};

// Separate entry point for automatic template argument deduction.
template<typename DT>
TensorInitializer_<DT> TensorInitializer(DT *val) {
  return TensorInitializer_<DT>(val);
}


// Initialize all elements of a variable with arbitrary dimensions using a
// function.
template<typename DT, int R, typename F>
class FunctionInitializer_ {
 public:
  explicit FunctionInitializer_(F f) : function_(f) {}

  Tensor<DT> operator()(Graph *g, const Dimensions& dims, DeviceID device) {
    return g->ConstantFromFunction<DT, R>(dims, function_, device);
  }

 private:
  F function_;
};

// Separate entry point for automatic template argument deduction of DT.
template<typename DT, int R, typename F>
FunctionInitializer_<DT, R, F> FunctionInitializer(F f) {
  return FunctionInitializer_<DT, R, F>(f);
}


// Initialize all elements of a vector variable with uniformly distributed
// random numbers.
template<typename DT>
class UniformRandomInitializer {
 public:
  explicit UniformRandomInitializer(uint64_t seed)
      : seed_(seed), has_seed_(true) {}

  UniformRandomInitializer() : has_seed_(false) {}

  Tensor<DT> operator()(Graph *g, const Dimensions& dims, DeviceID device) {
    return g->UniformRandom<DT>(dims, has_seed_ ? seed_ : 0, device);
  }

 private:
  uint64_t seed_;
  bool has_seed_;
};


// Initialize all elements of a vector variable with normally distributed
// random numbers.
template<typename DT>
class NormalRandomInitializer {
 public:
  explicit NormalRandomInitializer(uint64_t seed, DT mean = 0.0f,
                                   DT stddev = 1.0f)
      : seed_(seed), has_seed_(true), mean_(mean), stddev_(stddev) {}

  explicit NormalRandomInitializer(DT mean = 0.0f, DT stddev = 1.0f)
      : has_seed_(false), mean_(mean), stddev_(stddev) {}

  Tensor<DT> operator()(Graph *g, const Dimensions& dims, DeviceID device) {
    Tensor<DT> rand = g->NormalRandom<DT>(dims, has_seed_ ? seed_ : 0, device);

    // TODO(delesley): Generalize.  This only works if DT is float or double.
    auto mean = g->ConstantFromScalar<DT>(dims, mean_);
    auto stddev = g->ConstantFromScalar<DT>(dims, stddev_);
    return g->Add(mean, g->Multiply(rand, stddev));
  }

 private:
  uint64_t seed_;
  bool has_seed_;
  DT mean_;
  DT stddev_;
};


}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_VARIABLE_INITIALIZERS_H_
