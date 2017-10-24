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

#ifndef TENSORFLOW_FOLD_LLGTM_TEST_EVALUATOR_TEST_H_
#define TENSORFLOW_FOLD_LLGTM_TEST_EVALUATOR_TEST_H_

#include "tensorflow_fold/llgtm/test/test_framework.h"

namespace llgtm {

template <typename T>
class EvaluatorTest : public DeviceAwareTest<T> {};

TYPED_TEST_CASE_P(EvaluatorTest);


TYPED_TEST_P(EvaluatorTest, TestScalarArithmetic) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims;

  float one alignas(16) = 1.0;
  float two alignas(16) = 2.0;
  float six alignas(16) = 6.0;

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);
    auto v_one = g.Value(dims, &one);
    auto v_two = g.Value(dims, &two);
    auto v_six = g.Value(dims, &six);

    auto result = g.Multiply(g.Add(v_one, v_two), v_two);
    g.Eval();

    this->ExpectEq(result, v_six);
  }
}


TYPED_TEST_P(EvaluatorTest, TestScalarArithmeticUnaligned) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims;

  float one = 1.0;
  float two = 2.0;
  float three = 3.0;
  float six = 6.0;

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);
    auto v_one = g.Value(dims, &one);
    auto v_two = g.Value(dims, &two);
    auto v_three = g.Value(dims, &three);
    auto v_six = g.Value(dims, &six);

    auto result = g.Multiply(v_one, g.Multiply(v_two, v_three));
    g.Eval();

    this->ExpectEq(result, v_six);
  }
}


TYPED_TEST_P(EvaluatorTest, TestVectorArithmetic) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(16);

  // Initialize an array of floats.
  // This test case is CPU-only, but on the GPU data would have to be
  // allocated and deleted manually, so we emulate that here.
  float* data = new float[16];
  for (int i = 0; i < 16; ++i) data[i] = i;

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto zero = g.Zeros<float>(dims);
    auto val = g.Value(dims, data);
    auto one = g.Ones<float>(dims);
    auto range = g.ConstantFromFunction<float, 1>(
        dims, [](int i) -> float { return i; });

    // Test that Zeros and One produce the correct value.
    auto z_val = g.Add(zero, val);
    auto z_one = g.Add(zero, one);

    // Evaluate nodes up to z_one.
    g.Eval();

    this->ExpectEq(z_val, val);
    this->ExpectEq(z_one, one);
    this->ExpectEq(range, val);

    // Test that Add works correctly.
    auto val_one = g.Add(z_val, z_one);
    auto range_one = g.ConstantFromFunction<float, 1>(
        dims, [](int i) -> float { return i+1; });

    // Test that Multiply works correctly.
    auto val_sqr = g.Multiply(z_val, val);
    auto range_sqr = g.ConstantFromFunction<float, 1>(
        dims, [](int i) -> float { return i*i; });

    // Evaluate remaining nodes -- ensure nothing gets evaluated twice.
    int ne = g.Eval();
    ASSERT_EQ(ne, 4);

    this->ExpectEq(val_one, range_one);
    this->ExpectEq(val_sqr, range_sqr);
  }

  // Ownership check.
  // This will fail in debug mode if the Graph assumes ownership of data.
  delete[] data;
}


TYPED_TEST_P(EvaluatorTest, TestMatrixArithmetic) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(16, 16);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto zero = g.Zeros<float>(dims);
    auto c1 = g.ConstantFromFunction<float, 2>(
        dims, [](int i, int j) -> float { return i+j; });
    auto c2 = g.ConstantFromFunction<float, 2>(
        dims, [](int i, int j) -> float { return 2*(i+j); });
    auto sum = g.Add(zero, g.Add(c1, g.Add(c1, zero)));

    g.Eval();
    this->ExpectEq(sum, c2);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixRandomUniform) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(16, 16);
  Dimensions dims_big(2048, 2048);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto r1 = g.UniformRandom<float>(dims, /*seed=*/ 15);
    auto r2 = g.UniformRandom<float>(dims, /*seed=*/ 15);
    auto r3 = g.UniformRandom<float>(dims, /*seed=*/ 16);

    g.Eval();
    this->ExpectNear(r1, r2);
    this->ExpectNotNear(r1, r3);

    const auto* r1_data = this->tensor_data(r1);
    const auto* r2_data = this->tensor_data(r2);
    const auto* r3_data = this->tensor_data(r3);

    // Check range of values
    for (int i = 0; i < 16; ++i) {
      for (int j = 0; j < 16; ++j) {
        EXPECT_GE(r1_data[i*16 + j], 0);
        EXPECT_GE(r2_data[i*16 + j], 0);
        EXPECT_GE(r3_data[i*16 + j], 0);
        EXPECT_LT(r1_data[i*16 + j], 1);
        EXPECT_LT(r2_data[i*16 + j], 1);
        EXPECT_LT(r3_data[i*16 + j], 1);
      }
    }
  }

  {
    // Check variance for uniform distribution.
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto r1 = g.UniformRandom<float>(dims_big, /*seed=*/ 15);
    auto r2 = g.UniformRandom<float>(dims_big, /*seed=*/ 16);
    g.Eval();

    size_t r1_size = r1.dimensions().num_elements();
    size_t r2_size = r2.dimensions().num_elements();
    auto r1_variance = OnlineStats<float>(this->tensor_data(r1), r1_size)
        .variance();
    auto r2_variance = OnlineStats<float>(this->tensor_data(r2), r2_size)
        .variance();
    EXPECT_NEAR(r1_variance, 1.0/12.0, 0.001f);
    EXPECT_NEAR(r2_variance, 1.0/12.0, 0.001f);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixRandomNormal) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(16, 16);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto r1 = g.NormalRandom<float>(dims, /*seed=*/ 15);
    auto r2 = g.NormalRandom<float>(dims, /*seed=*/ 15);
    auto r3 = g.NormalRandom<float>(dims, /*seed=*/ 16);

    g.Eval();
    this->ExpectNear(r1, r2);
    this->ExpectNotNear(r1, r3);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixRandomSeed) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(16, 16);

  {
    // Test initial seed via graph construction.
    Graph g1 = evaluator.NewGraph(TypeParam::kDevice, /*seed=*/ 28071990);
    auto r1 = g1.UniformRandom<float>(dims);
    auto r2 = g1.UniformRandom<float>(dims);

    Graph g2 = evaluator.NewGraph(TypeParam::kDevice, /*seed=*/ 28071990);
    auto r3 = g2.UniformRandom<float>(dims);
    auto r4 = g2.UniformRandom<float>(dims);

    Graph g3 = evaluator.NewGraph(TypeParam::kDevice, /*seed=*/ 4071992);
    auto r5 = g3.UniformRandom<float>(dims);
    auto r6 = g3.UniformRandom<float>(dims);

    g1.Eval();
    g2.Eval();
    g3.Eval();

    this->ExpectNear(r1, r3);
    this->ExpectNear(r2, r4);
    this->ExpectNotNear(r1, r5);
  }

  {
    // Test random numbers from different graphs. Should have different seeds
    // and therefore different random numbers.
    Graph g1 = evaluator.NewGraph(TypeParam::kDevice);
    Graph g2 = evaluator.NewGraph(TypeParam::kDevice);

    auto r1 = g1.NormalRandom<float>(dims);
    auto r2 = g2.NormalRandom<float>(dims);

    g1.Eval();
    g2.Eval();
    this->ExpectNotNear(r1, r2);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixNegative) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(16, 16);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto one = g.Ones<float>(dims);
    auto two = g.ConstantFromScalar(dims, 2.0f);
    auto three = g.ConstantFromScalar(dims, 3.0f);
    auto result = g.Add(three, g.Negative(two));
    g.Eval();

    this->ExpectEq(result, one);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixReciprocal) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(16, 16);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto two = g.ConstantFromScalar(dims, 2.0f);
    auto three = g.ConstantFromScalar(dims, 3.0f);
    auto result = g.Multiply(
        g.Add(two, g.Add(two, two)), g.Reciprocal(two));

    g.Eval();

    this->ExpectNear(result, three);
  }
}


TYPED_TEST_P(EvaluatorTest, TestScalarSubtract) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims;

  float one alignas(16) = 1.0;
  float two alignas(16) = 2.0;
  float three alignas(16) = 3.0;

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);
    auto v_one = g.Value(dims, &one);
    auto v_two = g.Value(dims, &two);
    auto v_three = g.Value(dims, &three);

    auto result = g.Subtract(v_three, v_two);
    g.Eval();

    this->ExpectEq(result, v_one);
  }
}


TYPED_TEST_P(EvaluatorTest, TestScalarDivide) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims;

  float two alignas(16) = 2.0;
  float four alignas(16) = 4.0;
  float eight alignas(16) = 8.0;

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);
    auto v_two = g.Value(dims, &two);
    auto v_four = g.Value(dims, &four);
    auto v_eight = g.Value(dims, &eight);

    auto result = g.Divide(v_eight, v_two);
    g.Eval();

    this->ExpectNear(result, v_four);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixSigmoid) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(2, 2);

  alignas(16) float data[4] =
    { 0.0f, std::log(3.0f), std::log(9.0f), -std::log(9.0f) };
  alignas(16) float data_expected[4] = { 0.5f, 0.75f, 0.9f, 0.1f };

  {
     Graph g = evaluator.NewGraph(TypeParam::kDevice);

     auto v_data = g.Value(dims, data);
     auto v_data_expected = g.Value(dims, data_expected);
     auto result = g.Sigmoid(v_data);

     g.Eval();
     this->ExpectNear(result, v_data_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixTanh) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(2, 2);

  alignas(16) float data[4] = { 0.0f, 0.5f, 1.0f, -0.5f };
  alignas(16) float data_expected[4] = { 0.0f, std::tanh(0.5f),
                                         std::tanh(1.0f), std::tanh(-0.5f) };

  {
     Graph g = evaluator.NewGraph(TypeParam::kDevice);

     auto v_data = g.Value(dims, data);
     auto v_data_expected = g.Value(dims, data_expected);
     auto result = g.Tanh(v_data);

     g.Eval();
     this->ExpectNear(result, v_data_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixRelu) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(2, 2);

  alignas(16) float data[4] = { 0.0f, -1.5f, 1.0f, 2.0f };
  alignas(16) float data_expected[4] = { 0.0f, 0.0f, 1.0f, 2.0f };

  {
     Graph g = evaluator.NewGraph(TypeParam::kDevice);

     auto v_data = g.Value(dims, data);
     auto v_data_expected = g.Value(dims, data_expected);
     auto result = g.Relu(v_data);

     g.Eval();
     this->ExpectNear(result, v_data_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixReluInt) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(2, 2);

  alignas(16) int data[4] = { 0, -2, 1, 2 };
  alignas(16) int data_expected[4] = { 0, 0, 1, 2 };

  {
     Graph g = evaluator.NewGraph(TypeParam::kDevice);

     auto v_data = g.Value(dims, data);
     auto v_data_expected = g.Value(dims, data_expected);
     auto result = g.Relu(v_data);

     g.Eval();
     this->ExpectEq(result, v_data_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestReshape) {
  Dimensions s_dims;
  Dimensions m_dims(1, 1);

  float one_f alignas(16) = 1.0f;
  float two_f alignas(16) = 2.0f;
  float three_f alignas(16) = 3.0f;

  typename TypeParam::Evaluator evaluator;
  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto one_s = g.Value(s_dims, &one_f);
  auto two_m = g.Value(m_dims, &two_f);
  auto three_m = g.Value(m_dims, &three_f);

  auto expr = g.Add(g.Reshape(one_s, m_dims), two_m);

  g.Eval();

  this->ExpectEq(three_m, expr);
}


TYPED_TEST_P(EvaluatorTest, TestReshapeExpandDims) {
  Dimensions dims_1(16);
  Dimensions dims_2(2, 4, 2);

  typename TypeParam::Evaluator evaluator;
  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto one_1 = g.ConstantFromFunction<float, 1>(dims_1,
      [](int i) -> float { return 1.0f; });
  auto two_2 = g.ConstantFromFunction<float, 3>(dims_2,
      [](int i, int j, int k) -> float { return 2.0f; });
  auto three_2 = g.ConstantFromFunction<float, 3>(dims_2,
      [](int i, int j, int k) -> float { return 3.0f; });

  auto expr = g.Add(g.Reshape(one_1, dims_2), two_2);
  g.Eval();

  this->ExpectEq(expr, three_2);
}


TYPED_TEST_P(EvaluatorTest, TestReshapeSqueeze) {
  Dimensions dims_1(16);
  Dimensions dims_2(2, 4, 2);

  typename TypeParam::Evaluator evaluator;
  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto one_2 = g.ConstantFromFunction<float, 3>(dims_2,
      [](int i, int j, int k) -> float { return 1.0f; });
  auto two_1 = g.ConstantFromFunction<float, 1>(dims_1,
      [](int i) -> float { return 2.0f; });
  auto three_1 = g.ConstantFromFunction<float, 1>(dims_1,
      [](int i) -> float { return 3.0f; });

  auto expr = g.Add(g.Reshape(one_2, dims_1), two_1);
  g.Eval();

  this->ExpectEq(expr, three_1);
}


TYPED_TEST_P(EvaluatorTest, TestBroadcast) {
  Dimensions dims(16, 16);
  Dimensions dims_a(1, 16);
  Dimensions dims_b(16, 1);
  Dimensions dims_c(1, 1);

  float one_f alignas(16) = 1.0f;
  float two_f alignas(16) = 2.0f;

  typename TypeParam::Evaluator evaluator;
  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto one_s = g.Value(dims_c, &one_f);
  auto two_s = g.Value(dims_c, &two_f);

  auto one_v = g.ConstantFromScalar(dims_a, 1.0f);
  auto two_v = g.ConstantFromScalar(dims_b, 2.0f);
  auto three = g.ConstantFromScalar(dims, 3.0f);

  auto one_sb = g.Broadcast<float>(one_s, dims);
  auto two_sb = g.Broadcast<float>(two_s, dims);
  auto one_vb = g.Broadcast<float>(one_v, dims);
  auto two_vb = g.Broadcast<float>(two_v, dims);

  auto three_ss = g.Add(one_sb, two_sb);
  auto three_vs = g.Add(one_vb, two_sb);
  auto three_sv = g.Add(one_sb, two_vb);
  auto three_vv = g.Add(one_vb, two_vb);

  g.Eval();
  this->ExpectEq(three, three);
  this->ExpectEq(three, three_sv);
  this->ExpectEq(three, three_vs);
  this->ExpectEq(three, three_vv);
}


TYPED_TEST_P(EvaluatorTest, TestReduceSum) {
  Dimensions dims(16, 16);
  Dimensions dims_a(1, 16);
  Dimensions dims_b(16, 1);
  Dimensions dims_c(1, 1);

  typename TypeParam::Evaluator evaluator;
  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto mat = g.ConstantFromFunction<float, 2>(dims,
      [](int i, int j) -> float { return 2*i + j; });
  auto v0 = g.ConstantFromFunction<float, 2>(dims_a,
      [](int i, int j) -> float { return 240 + 16*j; });
  auto v1 = g.ConstantFromFunction<float, 2>(dims_b,
      [](int i, int j) -> float { return 32*i + 120; });

  auto vc = g.ConstantFromScalar(dims_c, 5760.0f);

  auto r0 = g.ReduceSum<float>(mat, { 0 });
  auto r1 = g.ReduceSum<float>(mat, { 1 });
  auto rc = g.ReduceSum<float>(mat, { 0, 1 });

  g.Eval();
  this->ExpectEq(v0, r0);
  this->ExpectEq(v1, r1);
  this->ExpectEq(vc, rc);
}


TYPED_TEST_P(EvaluatorTest, TestMatrixTranspose) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(10, 20);
  Dimensions dims_t(20, 10);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c1 = g.ConstantFromFunction<float, 2>(
        dims, [](int i, int j) -> float { return i+100*j; });
    auto c2 = g.ConstantFromFunction<float, 2>(
        dims_t, [](int i, int j) -> float { return 100*i+j; });
    auto result = g.Transpose(c1);

    g.Eval();
    this->ExpectEq(result, c2);
  }
}


TYPED_TEST_P(EvaluatorTest, TestTensorTranspose) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(9, 7, 5);
  Dimensions dims_t(5, 9, 7);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto ones = g.ConstantFromScalar(dims_t, 1.0f);
    auto c1 = g.ConstantFromFunction<float, 3>(
        dims, [](int i, int j, int k) -> float { return i + 10*j + 100*k; });
    auto c2 = g.ConstantFromFunction<float, 3>(
        dims_t,
        [](int i, int j, int k) -> float { return 100*i + j + 10*k + 1; });
    auto result = g.Add(g.Transpose<float>(c1, {2, 0, 1}), ones);

    g.Eval();
    this->ExpectEq(result, c2);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixMatrixMul) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims_a(2, 3);
  Dimensions dims_b(3, 4);
  Dimensions dims_r(2, 4);

  alignas(16) float r_expected[8] = { 5000.0f, 5300.0f, 5600.0f, 5900.0f,
                                      35000.0f, 38300.0f, 41600.0f, 44900.0f };

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c1 = g.ConstantFromFunction<float, 2>(
        dims_a, [](int i, int j) -> float { return 10*i + j; });
    auto c2 = g.ConstantFromFunction<float, 2>(
        dims_b, [](int i, int j) -> float { return 1000*i + 100*j; });
    auto result = g.Matmul(c1, c2);

    auto v_r_expected = g.Value(dims_r, r_expected);

    g.Eval();
    this->ExpectEq(result, v_r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixVectorMul) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims_a(2, 3);
  Dimensions dims_b(3);
  Dimensions dims_b_matrix(3, 1);
  Dimensions dims_r(2, 1);

  alignas(16) float r_expected[2] = { 500.0f, 3500.0f };

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c1 = g.ConstantFromFunction<float, 2>(
        dims_a, [](int i, int j) -> float { return 10*i + j; });
    auto c2 = g.ConstantFromFunction<float, 1>(
        dims_b, [](int i) -> float { return 100*i; });
    auto result = g.Matmul(c1, g.Reshape(c2, dims_b_matrix));

    auto v_r_expected = g.Value(dims_r, r_expected);

    g.Eval();
    this->ExpectEq(result, v_r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestVectorSoftmax) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(5);

  alignas(16) std::array<float, 5> input = {{ 0.0f, 1.0f, 2.0f, 3.0f, 4.0f }};
  alignas(16) std::array<float, 5> result = softmax(input);

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c1 = g.Value(dims, input.data());
    auto v_result = g.Softmax(c1);
    auto v_r_expected = g.Value<float>(dims, result.data());

    g.Eval();
    this->ExpectNear(v_result, v_r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixSoftmax) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(2, 3);

  alignas(16) std::array<std::array<float, 3>, 2> input =
      {{ {{ 0.1f, 1.0f, 3.9f }},
         {{ 2.4f, 1.0f, 0.0f }} }};
  alignas(16) std::array<std::array<float, 3>, 2> result =
      {{ softmax(input[0]), softmax(input[1]) }};

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c1 = g.Value(dims, input[0].data());
    auto v_result = g.Softmax(c1);
    auto v_r_expected = g.Value(dims, result[0].data());

    g.Eval();
    this->ExpectNear(v_result, v_r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixSoftmaxCrossEntropy) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(2, 3);
  Dimensions dims_r(2, 1);

  alignas(16) std::array<std::array<float, 3>, 2> input =
      {{ {{ 0.1f, 1.0f, 3.9f }},
         {{ 2.4f, 1.0f, 0.0f }} }};
  alignas(16) std::array<std::array<float, 3>, 2> labels =
      {{ {{ 1.0f, 0.0f, 0.0f }},
         {{ 0.0f, 1.0f, 0.0f }} }};
  alignas(16) std::array<float, 2> result =
      {{ cross_entropy(labels[0], softmax(input[0])),
         cross_entropy(labels[1], softmax(input[1])) }};

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto v_input = g.Value<float>(dims, input[0].data());
    auto v_labels = g.Value<float>(dims, labels[0].data());
    auto v_result = g.CrossEntropyLoss(v_labels, g.Softmax(v_input));
    auto v_r_expected = g.Value(dims_r, result.data());

    g.Eval();
    this->ExpectNear(v_result, v_r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixSoftmaxSparseCrossEntropy) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(2, 3);
  Dimensions dims_r(2, 1);
  Dimensions dims_l(2);

  alignas(16) std::array<std::array<float, 3>, 2> input =
      {{ {{ 0.1f, 1.0f, 3.9f }},
         {{ 2.4f, 1.0f, 0.0f }} }};
  alignas(16) std::array<std::array<float, 3>, 2> labels =
      {{ {{ 1.0f, 0.0f, 0.0f }},
         {{ 0.0f, 0.0f, 1.0f }} }};
  alignas(16) std::array<int, 2> sparse_labels = {{ 0, 2 }};
  alignas(16) std::array<float, 2> result =
      {{ cross_entropy(labels[0], softmax(input[0])),
         cross_entropy(labels[1], softmax(input[1])) }};

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto v_input = g.Value<float>(dims, input[0].data());
    auto v_labels = g.Value<int>(dims_l, sparse_labels.data());
    auto v_result = g.SparseCrossEntropyLoss(v_labels, g.Softmax(v_input));
    auto v_r_expected = g.Value<float>(dims_r, result.data());

    g.Eval();
    this->ExpectNear(v_result, v_r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestVectorSoftmaxSparseCrossEntropy) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(3);
  Dimensions dims_s;

  alignas(16) std::array<float, 3> input = {{ 0.1f, 1.0f, 3.9f }};
  alignas(16) std::array<float, 3> labels = {{ 0.0f, 0.0f, 1.0f }};
  alignas(16) int sparse_label = 2;
  alignas(16) float result = cross_entropy(labels, softmax(input));

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto v_input = g.Value<float>(dims, input.data());
    auto v_labels = g.Value<int>(dims_s, &sparse_label);
    auto v_result = g.Reshape(g.SparseCrossEntropyLoss(v_labels,
                                                       g.Softmax(v_input)),
                              dims_s);
    auto v_r_expected = g.Value<float>(dims_s, &result);

    g.Eval();
    this->ExpectNear(v_result, v_r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixConcat2) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims_1(4, 16);
  Dimensions dims_2(12, 16);
  Dimensions dims_3(16, 4);
  Dimensions dims_4(16, 12);
  Dimensions dims_r(16, 16);

  {
    // Test axis 0.
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c1 = g.ConstantFromFunction<float, 2>(
        dims_1, [](int i, int j) -> float { return 100*i + j; });
    auto c2 = g.ConstantFromFunction<float, 2>(
        dims_2, [](int i, int j) -> float { return 100*(i + 4) + j; });
    auto r_expected = g.ConstantFromFunction<float, 2>(
        dims_r, [](int i, int j) -> float { return 100*i + j; });

    auto result = g.Concat<float>({c1, c2}, 0);

    g.Eval();
    this->ExpectEq(result, r_expected);
  }

  {
    // Test axis 1.
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c3 = g.ConstantFromFunction<float, 2>(
        dims_3, [](int j, int i) -> float { return 100*i + j; });
    auto c4 = g.ConstantFromFunction<float, 2>(
        dims_4, [](int j, int i) -> float { return 100*(i + 4) + j; });
    auto r_expected = g.ConstantFromFunction<float, 2>(
        dims_r, [](int j, int i) -> float { return 100*i + j; });

    auto result = g.Concat<float>({c3, c4}, 1);

    g.Eval();
    this->ExpectEq(result, r_expected);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixConcatMulti) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims_1(4, 100);

  // Test concatenation with 2-30 parts. This test covers
  // nodes::Concat<DT>::EigenKernel<-1, R>, including dummy tensors.
  for (int p = 2; p < 30; ++p)
  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto c1 = g.ConstantFromFunction<float, 2>(
        dims_1, [](int i, int j) -> float { return 1000 + j; });
    auto split_output = g.Split(c1, p, 1);

    std::vector<Tensor<float>> parts(p);

    for (int i = 0; i < p; ++i) {
      parts[i] = g.GetOutput(split_output, i).template as<float>();
    }

    auto result = g.Concat<float>(parts, 1);

    g.Eval();
    this->ExpectEq(result, c1);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixSplit) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims_1(4, 16);
  Dimensions dims_2(10, 16);
  Dimensions dims_3(2, 16);
  Dimensions dims_r(16, 16);

  {
    // Test axis 0.
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto r_expected_1 = g.ConstantFromFunction<float, 2>(
        dims_1, [](int i, int j) -> float { return 100*i + j; });
    auto r_expected_2 = g.ConstantFromFunction<float, 2>(
        dims_2, [](int i, int j) -> float { return 100*(i + 4) + j; });
    auto r_expected_3 = g.ConstantFromFunction<float, 2>(
        dims_3, [](int i, int j) -> float { return 100*(i + 4 + 10) + j; });
    auto c = g.ConstantFromFunction<float, 2>(
        dims_r, [](int i, int j) -> float { return 100*i + j; });

    auto split_output = g.Split(c, {4, 10, 2}, 0);
    auto r1 = g.GetOutput(split_output, 0).template as<float>();
    auto r2 = g.GetOutput(split_output, 1).template as<float>();
    auto r3 = g.GetOutput(split_output, 2).template as<float>();

    g.Eval();
    this->ExpectEq(r1, r_expected_1);
    this->ExpectEq(r2, r_expected_2);
    this->ExpectEq(r3, r_expected_3);
  }
}


TYPED_TEST_P(EvaluatorTest, TestMatrixSplitEqualSizes) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims_1(16, 5);
  Dimensions dims_2(16, 5);
  Dimensions dims_3(16, 5);
  Dimensions dims_r(16, 15);

  {
    // Test axis 1.
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto r_expected_1 = g.ConstantFromFunction<float, 2>(
        dims_1, [](int j, int i) -> float { return 100*i + j; });
    auto r_expected_2 = g.ConstantFromFunction<float, 2>(
        dims_2, [](int j, int i) -> float { return 100*(i + 5) + j; });
    auto r_expected_3 = g.ConstantFromFunction<float, 2>(
        dims_3, [](int j, int i) -> float { return 100*(i + 5 + 5) + j; });
    auto c = g.ConstantFromFunction<float, 2>(
        dims_r, [](int j, int i) -> float { return 100*i + j; });

    auto split_output = g.Split(c, 3, 1);
    auto r1 = g.GetOutput(split_output, 0).template as<float>();
    auto r2 = g.GetOutput(split_output, 1).template as<float>();
    auto r3 = g.GetOutput(split_output, 2).template as<float>();

    g.Eval();
    this->ExpectEq(r1, r_expected_1);
    this->ExpectEq(r2, r_expected_2);
    this->ExpectEq(r3, r_expected_3);
  }
}


TYPED_TEST_P(EvaluatorTest, TestGatherScatter) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims_a(5, 4);
  Dimensions dims_b(3, 4);

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto c1 = g.ConstantFromFunction<float, 2>(
      dims_a, [](int i, int j) -> float { return 100*i + j; });

  auto c2 = g.ConstantFromFunction<float, 2>(
      dims_b, [](int i, int j) -> float {
        if (i == 0 || i == 1) {
          return 100 + j;
        } else if (i == 2) {
          return 300 + j;
        } else {
          return 0;
        }
      });

  auto c3 = g.ConstantFromFunction<float, 2>(
      dims_a, [](int i, int j) -> float {
        if (i == 1) {
          return 2*(100*i + j);  // Sum of duplicates.
        } else if (i == 3) {
          return 100*i + j;
        } else {
          return 0;
        }
      });

  int32_t indices[3] = { 1, 1, 3 };  // Test duplicate indices.
  auto idx = g.Value<int32_t>(Dimensions(3), indices);

  auto r_gather = g.Gather(idx, c1);
  auto r_scatter = g.Scatter(dims_a[0], idx, r_gather);

  g.Eval();
  this->ExpectEq(r_gather, c2);
  this->ExpectEq(r_scatter, c3);
}


TYPED_TEST_P(EvaluatorTest, TestAssignAdd) {
  typename TypeParam::Evaluator evaluator;
  Dimensions dims(5, 4);

  VariableSet model(&evaluator);
  auto* var = model.NewVariable<float>("a", dims, nullptr,
                                       ZerosInitializer<float>());

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto v = g.Variable(var, TypeParam::kDevice);
  auto c1 = g.ConstantFromFunction<float, 2>(
      dims, [](int i, int j) -> float { return 100*i + j; });

  g.AssignAdd(var, c1);
  g.Eval();
  this->ExpectEq(v, c1);

  auto c2 = g.ConstantFromFunction<float, 2>(
      dims, [](int i, int j) -> float { return 200*i + 2*j; });

  g.AssignAdd(var, c1);
  g.Eval();
  this->ExpectEq(v, c2);
}


REGISTER_TYPED_TEST_CASE_P(EvaluatorTest,
                           TestScalarArithmetic,
                           TestScalarArithmeticUnaligned,
                           TestVectorArithmetic,
                           TestMatrixArithmetic,
                           TestMatrixRandomUniform,
                           TestMatrixRandomNormal,
                           TestMatrixRandomSeed,
                           TestMatrixNegative,
                           TestMatrixReciprocal,
                           TestScalarSubtract,
                           TestScalarDivide,
                           TestMatrixSigmoid,
                           TestReshape,
                           TestReshapeExpandDims,
                           TestReshapeSqueeze,
                           TestBroadcast,
                           TestReduceSum,
                           TestMatrixTanh,
                           TestMatrixRelu,
                           TestMatrixReluInt,
                           TestMatrixTranspose,
                           TestTensorTranspose,
                           TestMatrixMatrixMul,
                           TestMatrixVectorMul,
                           TestVectorSoftmax,
                           TestMatrixSoftmax,
                           TestMatrixSoftmaxCrossEntropy,
                           TestMatrixSoftmaxSparseCrossEntropy,
                           TestVectorSoftmaxSparseCrossEntropy,
                           TestMatrixConcat2,
                           TestMatrixConcatMulti,
                           TestMatrixSplit,
                           TestMatrixSplitEqualSizes,
                           TestGatherScatter,
                           TestAssignAdd);

}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_TEST_EVALUATOR_TEST_H_
