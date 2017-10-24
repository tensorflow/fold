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
#include "tensorflow_fold/llgtm/test/gradients_test.h"
#include "tensorflow_fold/llgtm/test/test_framework.h"

#ifndef GOOGLE_CUDA
#error "LLGTM built without CUDA support. Use --config=cuda."
#endif  // GOOGLE_CUDA

#ifndef NDEBUG
// See evaluator_test_eigen_gpu.cc
#error "Cannot run Eigen with CUDA support in debug mode. Run in opt mode."
#endif  // NDEBUG

namespace llgtm {
namespace {
using Configuration = TestConfiguration<EigenEvaluator, kDeviceIDGPU>;

INSTANTIATE_TYPED_TEST_CASE_P(EigenGPU, GradientsTest, Configuration);

class GradientsTestExtra : public DeviceAwareTest<void> {};

TEST_F(GradientsTestExtra, TestCopyToDevice) {
  // This test is based on GradientsTest::TestNegative.
  Dimensions dims;

  EigenEvaluator evaluator;
  VariableSet model(&evaluator);

  VarNameSpace* a_space = model.NewNameSpace("a_layer");
  VarNameSpace* b_space = model.NewNameSpace("b_layer", a_space);

  auto* va = model.NewVariable<float>("a", dims, a_space,
                                      ScalarInitializer(4.0f));
  auto* vb = model.NewVariable<float>("b", dims, b_space,
                                      ScalarInitializer(5.0f));

  {
    Graph g = evaluator.NewGraph(kDeviceIDCPU);

    auto a = g.Variable(va);
    auto b = g.Variable(vb);

    // Promote multiply to GPU.
    auto multiply_gpu = g.Add(g.Zeros<float>(dims, kDeviceIDGPU),
                              g.Multiply(a, g.Negative(b)));
    auto expr = g.Add(g.CopyToDevice(g.Negative(a), kDeviceIDCPU),
                      g.CopyToDevice(multiply_gpu, kDeviceIDCPU));

    Gradients grads(&model);
    g.ComputeGradients(&grads, expr);

    auto a_grad = g.Gradient(grads, va);
    auto b_grad = g.Gradient(grads, vb);

    // a = 4, b = 5
    // expr = -a + a*(-b)
    // grad_a = -1 - 5 = -6
    // grad_b = 0 - 4 = -4
    auto a_grad_expected = g.ConstantFromScalar(dims, -6.0f);
    auto b_grad_expected = g.ConstantFromScalar(dims, -4.0f);

    g.Eval();

    this->ExpectEq(a_grad, a_grad_expected);
    this->ExpectEq(b_grad, b_grad_expected);
  }
}

}  // namespace
}  // namespace llgtm
