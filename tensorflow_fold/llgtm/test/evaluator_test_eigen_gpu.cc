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

#include "tensorflow_fold/llgtm/backend/eigen_evaluator_client.h"
#include "tensorflow_fold/llgtm/test/evaluator_test.h"
#include "tensorflow_fold/llgtm/test/test_framework.h"

#ifndef GOOGLE_CUDA
#error "LLGTM built without CUDA support. Use --config=cuda."
#endif  // GOOGLE_CUDA

#ifndef NDEBUG
// TODO(matthiasspringer): Investigate why this is not working.
// Workarounds: a) Run in optimized mode.
//              b) Possibly use --config=nvcc8 (does not compile).
#error "Cannot run Eigen with CUDA support in debug mode. Run in opt mode."
#endif  // NDEBUG

namespace llgtm {
namespace {
using Configuration = TestConfiguration<EigenEvaluator, kDeviceIDGPU>;

INSTANTIATE_TYPED_TEST_CASE_P(Eigen, EvaluatorTest, Configuration);

class EvaluatorTestEigenGPUExtra : public DeviceAwareTest<void> {};

TEST_F(EvaluatorTestEigenGPUExtra, TestCopyToDevice) {
  EigenEvaluator evaluator;
  Dimensions dims;

  float one = 1.0;

  {
    Graph g = evaluator.NewGraph(kDeviceIDGPU);
    auto v_one = g.Value(dims, &one);
    auto v_one_cpu = g.Value(dims, &one, kDeviceIDCPU);

    // Calculate on GPU, copy result back to CPU.
    auto result = g.CopyToDevice(v_one, kDeviceIDCPU);
    g.Eval();

    this->ExpectEq(result, v_one_cpu);
  }
}

TEST_F(EvaluatorTestEigenGPUExtra, TestDevicePromotion) {
  EigenEvaluator evaluator;
  Dimensions dims;

  float one = 1.0;
  float two = 2.0;
  float five = 5.0;

  {
    Graph g = evaluator.NewGraph(kDeviceIDCPU);
    auto v_one = g.Value(dims, &one);                     // unspecified dev.
    auto v_two = g.Value(dims, &two);                     // unspecified dev.
    auto v_two_gpu = g.Value(dims, &two, kDeviceIDGPU);
    auto v_five = g.Value(dims, &five);

    // Calculate on GPU, copy result back to CPU.
    auto sum = g.Add(g.Add(v_one, v_two), v_two_gpu);
    EXPECT_EQ(v_one.device(), kDeviceIDGPU);

    auto result = g.CopyToDevice(sum, kDeviceIDCPU);
    g.Eval();

    this->ExpectEq(result, v_five);
  }
}

}  // namespace
}  // namespace llgtm
