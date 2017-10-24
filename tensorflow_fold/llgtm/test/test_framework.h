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

// Definitions that are common to all tests.

#ifndef TENSORFLOW_FOLD_LLGTM_TEST_TEST_FRAMEWORK_H_
#define TENSORFLOW_FOLD_LLGTM_TEST_TEST_FRAMEWORK_H_

#include <cmath>
#include <gtest/gtest.h>
#include "tensorflow_fold/llgtm/device.h"
#include "tensorflow_fold/llgtm/gradients.h"
#include "tensorflow_fold/llgtm/graph.h"
#include "tensorflow_fold/llgtm/layers.h"
#include "tensorflow_fold/llgtm/tensor_ops_impl.h"
#include "tensorflow_fold/llgtm/trainer.h"
#include "tensorflow_fold/llgtm/util.h"
#include "tensorflow_fold/llgtm/variable_initializers.h"

#if defined(LLGTM_PLATFORM_GOOGLE)
#include "testing/base/public/gmock.h"
#else
#include <gmock/gmock-generated-matchers.h>
#include <gmock/gmock-matchers.h>
#endif

#ifdef GOOGLE_CUDA
#include <vector>
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime.h"
#endif

namespace llgtm {

using testing::FloatNear;
using testing::Not;

template<class DT>
void ExpectEqData(const DT* a_data, const DT* b_data, size_t num_elements) {
  for (int i = 0, n = num_elements; i < n; ++i) {
    EXPECT_EQ(a_data[i], b_data[i]);
  }
}

template<class DT>
void ExpectNearData(const DT* a_data, const DT* b_data,
                    size_t num_elements, float tolerance) {
  for (int i = 0, n = num_elements; i < n; ++i) {
    // EXPECT_FLOAT_EQ is too unprecise for some TF tests.
    EXPECT_NEAR(a_data[i], b_data[i], tolerance);
  }
}

template<class DT>
void ExpectNotNearData(const DT* a_data, const DT* b_data,
                       size_t num_elements, float tolerance) {
  for (int i = 0, n = num_elements; i < n; ++i) {
    EXPECT_THAT(a_data[i], Not(FloatNear(b_data[i], tolerance)));
  }
}


// Instantiations of this template are used as type parameter for
// DeviceAwareTest. The first template parameter is the evaluator, the second
// one specifies the device that should be used for the tests.
template<typename E, DeviceID D>
struct TestConfiguration {
  using Evaluator = E;
  static constexpr DeviceID kDevice = D;
  static constexpr float kFloatTolerance = 1.0e-6;
};


template<typename Configuration>
class DeviceAwareTest : public ::testing::Test {
 public:
  template<class DT>
  void ExpectEq(const Tensor<DT>& a, const Tensor<DT>& b) {
    EXPECT_EQ(a.dimensions(), b.dimensions());
    if (a.dimensions() != b.dimensions()) {
      return;
    }

    ExpectEqData(tensor_data(a), tensor_data(b),
                 a.dimensions().num_elements());
  }

  template<class DT>
  void ExpectNear(const Tensor<DT>& a, const Tensor<DT>& b) {
    EXPECT_EQ(a.dimensions(), b.dimensions());
    if (a.dimensions() != b.dimensions()) {
      return;
    }

    ExpectNearData(tensor_data(a), tensor_data(b),
                   a.dimensions().num_elements(),
                   Configuration::kFloatTolerance);
  }

  template<class DT>
  void ExpectNotNear(const Tensor<DT>& a, const Tensor<DT>& b) {
    EXPECT_EQ(a.dimensions(), b.dimensions());
    if (a.dimensions() != b.dimensions()) {
      return;
    }

    ExpectNotNearData(tensor_data(a), tensor_data(b),
                      a.dimensions().num_elements(),
                      Configuration::kFloatTolerance);
  }

  template<typename DT>
  const DT* tensor_data(const Tensor<DT>& t) {
    switch (t.device()) {
      case kDeviceIDCPU:
        return t.result_data();
#ifdef GOOGLE_CUDA
      case kDeviceIDGPU: {
        // TODO(matthiasspringer): This transfer should ideally be done via the
        // graph evaluator.
        // TODO(matthiasspringer): Handle multiple GPUs.
        size_t bytes = t.dimensions().num_elements()*sizeof(DT);
        DT* data = reinterpret_cast<DT*>(malloc(bytes));
        cudaMemcpy(data, t.result_data(), bytes, cudaMemcpyDeviceToHost);
        cpu_allocations_.push_back(data);
        return data;
      }
#endif
      default:
        LOG(FATAL) << "Device " << device_name(t.device())
                   << " not supported.";
    }
  }

 protected:
  void TearDown() override {
#ifdef GOOGLE_CUDA
    for (void* d : cpu_allocations_) {
      free(d);
    }
#endif
  }

 private:
#ifdef GOOGLE_CUDA
  std::vector<void*> cpu_allocations_;
#endif
};


// Compute softmax(input). Helper function for unit tests only.
template<uint64_t N>
std::array<float, N> softmax(const std::array<float, N>& input) {
  double sum = 0.0;
  for (int i = 0; i < N; ++i) {
    sum += exp(input[i]);
  }

  std::array<float, N> result;
  for (int i = 0; i < N; ++i) {
    result[i] = exp(input[i]) / sum;
  }

  return result;
}


// Compute cross-entropy of two probability distributions. Helper function for
// tests only.
template<uint64_t S>
float cross_entropy(const std::array<float, S>& labels,
                    const std::array<float, S>& probabilities) {
  double sum = 0.0;

  for (int i = 0; i < S; ++i) {
    sum -= labels[i] * std::log(probabilities[i]);
  }

  return sum;
}


// For debugging use only.  Dump a tensor of rank 2 to stderr.
void DumpTensor2(const char* name, Tensor<float> a) {
  CHECK_EQ(a.device(), kDeviceIDCPU);

  std::cerr << name << ": ";
  float* data = a.result_data();
  int rows = a.dimension(0);
  int cols = a.dimension(1);
  int k = 0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      std::cerr << data[k] << " ";
      ++k;
    }
    std::cerr << "\n";
  }
}


// For debugging use only.  Dump a tensor to stderr.
void DumpTensor(const char* name, Tensor<float> a) {
  CHECK_EQ(a.device(), kDeviceIDCPU);

  if (a.rank() == 2) {
    DumpTensor2(name, a);
    return;
  }
  std::cerr << name << ": ";
  float* data = a.result_data();
  int sz = a.dimensions().num_elements();
  for (int i = 0; i < sz; ++i) {
    std::cerr << data[i] << " ";
  }
  std::cerr << "\n";
}


}  // namespace llgtm


#endif  // TENSORFLOW_FOLD_LLGTM_TEST_TEST_FRAMEWORK_H_
