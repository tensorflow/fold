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

#ifndef TENSORFLOW_FOLD_LLGTM_TEST_GRADIENTS_TEST_H_
#define TENSORFLOW_FOLD_LLGTM_TEST_GRADIENTS_TEST_H_

#include "tensorflow_fold/llgtm/test/test_framework.h"

namespace llgtm {

template <typename T>
class GradientsTest : public DeviceAwareTest<T> {};

TYPED_TEST_CASE_P(GradientsTest);


TYPED_TEST_P(GradientsTest, TestScalarGradients) {
  Dimensions dims;

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  VarNameSpace* a_space = model.NewNameSpace("a_layer");
  VarNameSpace* b_space = model.NewNameSpace("b_layer", a_space);

  auto* va = model.NewVariable<float>("a", dims, a_space,
                                      ScalarInitializer(10.0f));
  auto* vb = model.NewVariable<float>("b", dims, b_space,
                                      ScalarInitializer(20.0f));

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto a = g.Variable(va);
    auto b = g.Variable(vb);

    auto two = g.ConstantFromScalar(dims, 2.0f);
    auto three = g.ConstantFromScalar(dims, 3.0f);

    auto expr = g.Add(g.Add(g.Multiply(two, a),
                            g.Multiply(three, b)),
                      g.Multiply(a, b));

    Gradients grads(&model);
    g.ComputeGradients(&grads, expr);

    auto a_grad = g.Gradient(grads, va);
    auto b_grad = g.Gradient(grads, vb);

    // a = 10, b = 20
    // expr = 2*a + 3*b + a*b
    // grad_a = 2 + 20 = 22
    // grad_b = 3 + 10 = 13
    auto a_grad_expected = g.ConstantFromScalar<float>(dims, 22.0f);
    auto b_grad_expected = g.ConstantFromScalar<float>(dims, 13.0f);

    g.Eval();

    this->ExpectEq(a_grad, a_grad_expected);
    this->ExpectEq(b_grad, b_grad_expected);
  }
}


TYPED_TEST_P(GradientsTest, TestNegative) {
  Dimensions dims;

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  VarNameSpace* a_space = model.NewNameSpace("a_layer");
  VarNameSpace* b_space = model.NewNameSpace("b_layer", a_space);

  auto* va = model.NewVariable<float>("a", dims, a_space,
                                      ScalarInitializer(4.0f));
  auto* vb = model.NewVariable<float>("b", dims, b_space,
                                      ScalarInitializer(5.0f));

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto a = g.Variable(va);
    auto b = g.Variable(vb);

    auto expr = g.Add(g.Negative(a), g.Multiply(a, g.Negative(b)));

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


TYPED_TEST_P(GradientsTest, TestReciprocal) {
  Dimensions dims;

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  VarNameSpace* a_space = model.NewNameSpace("a_layer");
  VarNameSpace* b_space = model.NewNameSpace("b_layer", a_space);

  auto* va = model.NewVariable<float>("a", dims, a_space,
                                      ScalarInitializer(2.0f));
  auto* vb = model.NewVariable<float>("b", dims, b_space,
                                      ScalarInitializer(8.0f));

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto a = g.Variable(va);
    auto b = g.Variable(vb);

    auto expr = g.Multiply(a, g.Reciprocal(g.Add(a, b)));

    Gradients grads(&model);
    g.ComputeGradients(&grads, expr);

    auto a_grad = g.Gradient(grads, va);
    auto b_grad = g.Gradient(grads, vb);

    // a = 2, b = 8
    // expr = a * (1 / (a + b)) = a / (a + b)
    // grad_a = b / (a + b)**2 = 0.08
    // grad_b = -a / (a + b)**2 = -0.02
    auto a_grad_expected = g.ConstantFromScalar<float>(dims, 0.08f);
    auto b_grad_expected = g.ConstantFromScalar<float>(dims, -0.02f);

    g.Eval();

    this->ExpectNear(a_grad, a_grad_expected);
    this->ExpectNear(b_grad, b_grad_expected);
  }
}


TYPED_TEST_P(GradientsTest, TestSigmoid) {
  Dimensions dims;

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  float a = 0.5f;
  float b = 0.75f;

  auto* var_a = model.NewVariable<float>("a", dims, nullptr,
                                         ScalarInitializer(a));
  auto* var_b = model.NewVariable<float>("b", dims, nullptr,
                                         ScalarInitializer(b));

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto a_node = g.Variable(var_a);
    auto b_node = g.Variable(var_b);

    auto expr = g.Add(b_node, g.Multiply(g.Sigmoid(g.Multiply(a_node, b_node)),
                                         a_node));

    Gradients grads(&model);
    g.ComputeGradients(&grads, expr);

    auto a_grad = g.Gradient(grads, var_a);
    auto b_grad = g.Gradient(grads, var_b);

    // Gradients calculated manually.
    float e_ab = std::exp(a*b);
    float a_expected_value = e_ab * (a*b + e_ab + 1) / (e_ab + 1) / (e_ab + 1);
    float b_expected_value = a*a*e_ab / (e_ab + 1) / (e_ab + 1) + 1;

    auto a_grad_expected = g.ConstantFromScalar(dims, a_expected_value);
    auto b_grad_expected = g.ConstantFromScalar(dims, b_expected_value);

    g.Eval();

    this->ExpectNear(a_grad, a_grad_expected);
    this->ExpectNear(b_grad, b_grad_expected);
  }
}


TYPED_TEST_P(GradientsTest, TestTanh) {
  Dimensions dims;

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  float a = 0.5f;
  float b = 0.75f;

  auto* var_a = model.NewVariable<float>("a", dims, nullptr,
                                         ScalarInitializer(a));
  auto* var_b = model.NewVariable<float>("b", dims, nullptr,
                                         ScalarInitializer(b));

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto a_node = g.Variable(var_a);
    auto b_node = g.Variable(var_b);

    auto expr = g.Multiply(g.Tanh(a_node), b_node);

    Gradients grads(&model);
    g.ComputeGradients(&grads, expr);

    auto a_grad = g.Gradient(grads, var_a);
    auto b_grad = g.Gradient(grads, var_b);

    // Gradients calculated manually.
    auto a_grad_expected = g.ConstantFromScalar<float>(
        dims, 2*b / (std::cosh(2*a) + 1));
    auto b_grad_expected = g.ConstantFromScalar<float>(dims, std::tanh(a));

    g.Eval();

    this->ExpectNear(a_grad, a_grad_expected);
    this->ExpectNear(b_grad, b_grad_expected);
  }
}


TYPED_TEST_P(GradientsTest, TestRelu) {
  Dimensions dim_s;
  Dimensions dim_v(4);

  // 4 test cases: Different combinations of positive and negative values.
  alignas(16) float data_a[4] = { 20.0f, -20.0f, -20.0f, -20.0f };
  alignas(16) float data_b[4] = { 300.0f, 300.0f, 0.0f, -300.0f };
  alignas(16) float data_grad_a[4] = { 1500.0f, 0.0f, 0.0f, -1500.0f };
  alignas(16) float data_grad_b[4] = { 101.0f, 1.0f, 1.0f, -99.0f };

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  auto* va = model.NewVariable<float>(
      "a", dim_v, nullptr, TensorInitializer(data_a));
  auto* vb = model.NewVariable<float>(
      "b", dim_v, nullptr, TensorInitializer(data_b));

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto a = g.Variable(va);
  auto b = g.Variable(vb);

  auto expr_relu = g.Relu(
      g.Multiply(g.ConstantFromScalar(dim_v, 5.0f),
                 g.Multiply(a, b)));
  auto expr = g.Reshape(
    g.ReduceSum<float>(g.Add(expr_relu, b), { 0 }), dim_s);

  Gradients grads(&model);
  g.ComputeGradients(&grads, expr);

  auto a_grad = g.Gradient(grads, va);
  auto b_grad = g.Gradient(grads, vb);

  // d/da relu(5*a*b) + b = (5*a*b > 0) * 5b
  // d/db relu(5*a*b) + b = (5*a*b > 0) * 5a + 1
  auto a_grad_expected = g.Value(dim_v, data_grad_a);
  auto b_grad_expected = g.Value(dim_v, data_grad_b);

  g.Eval();

  this->ExpectEq(a_grad, a_grad_expected);
  this->ExpectEq(b_grad, b_grad_expected);
}


TYPED_TEST_P(GradientsTest, TestBroadcastReduceSum) {
  Dimensions s_dims0;
  Dimensions s_dims(1, 1);
  Dimensions v_dims(1, 16);
  Dimensions dims(16, 16);

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  VarNameSpace* vspace = model.NewNameSpace("default");
  auto* va = model.NewVariable<float>("a", s_dims, vspace,
                                      ZerosInitializer<float>());
  auto* vb = model.NewVariable<float>("b", v_dims, vspace,
                                      ZerosInitializer<float>());

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto a = g.Variable(va);
  auto b = g.Variable(vb);

  auto mat = g.ConstantFromFunction<float, 2>(dims,
      [](int i, int j) -> float { return 2*i + j; });

  auto expr = g.Add(
    g.ReduceSum<float>(g.Multiply(mat, g.Broadcast<float>(a, dims)), {0, 1}),
    g.ReduceSum<float>(g.Multiply(mat, g.Broadcast<float>(b, dims)), {0, 1}));
  auto loss = g.Reshape(expr, s_dims0);

  Gradients grads(&model);
  g.ComputeGradients(&grads, loss);

  auto a_grad = g.Gradient(grads, va);
  auto b_grad = g.Gradient(grads, vb);

  auto a_grad_expected = g.ConstantFromScalar(s_dims, 5760.0f);
  auto b_grad_expected = g.ConstantFromFunction<float, 2>(
      v_dims, [](int i, int j) -> float { return 240 + 16*j; });

  g.Eval();

  this->ExpectEq(a_grad, a_grad_expected);
  this->ExpectEq(b_grad, b_grad_expected);
}


TYPED_TEST_P(GradientsTest, TestTranspose) {
  Dimensions s_dims0;
  Dimensions s_dims(1, 1);
  Dimensions dims_1(2, 3);
  Dimensions dims_2(3, 2);

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  VarNameSpace* vspace = model.NewNameSpace("default");
  auto* vb = model.NewVariable<float>("b", dims_2, vspace,
      FunctionInitializer<float, 2>(
          [](int i, int j) -> float { return 200.0*i + 100.0*j; }));

  Graph g = evaluator.NewGraph(TypeParam::kDevice);
  auto a = g.ConstantFromFunction<float, 2>(
      dims_1, [](int i, int j) -> float { return 30.0*i + 10.0*j; });
  auto b = g.Variable(vb);
  auto expr = g.ReduceSum<float>(g.Multiply(a, g.Transpose(b)), {0, 1});
  auto loss = g.Reshape(expr, s_dims0);

  Gradients grads(&model);
  g.ComputeGradients(&grads, loss);
  auto b_grad = g.Gradient(grads, vb);

  // Gradient: transpose(a)
  auto b_grad_expected = g.ConstantFromFunction<float, 2>(
      dims_2, [](int i, int j) -> float { return 10.0*i + 30.0*j; });

  alignas(16) float result_value = 50000.0f;
  auto v_result = g.Value(s_dims0, &result_value);

  g.Eval();

  this->ExpectEq(loss, v_result);
  this->ExpectEq(b_grad, b_grad_expected);
}


TYPED_TEST_P(GradientsTest, TestMatmul) {
  Dimensions dim_s;
  Dimensions dim_x_m(1, 3);
  Dimensions dim_w(3, 2);
  Dimensions dim_b(2);

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  VarNameSpace* vspace = model.NewNameSpace("default");
  auto* vW = model.NewVariable<float>(
      "W", dim_w, vspace, ZerosInitializer<float>());
  auto* vb = model.NewVariable<float>(
      "b", dim_b, vspace, ZerosInitializer<float>());

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  auto x = g.ConstantFromFunction<float, 2>(
      dim_x_m, [](int i, int j) -> float { return j; });
  auto W = g.Variable(vW);
  auto b = g.Variable(vb);

  // expr = x * W + b
  auto expr = g.Reshape(
      g.ReduceSum<float>(
          g.Add(g.Reshape(g.Matmul(x, W), dim_b), b), { 0 }),
      dim_s);

  Gradients grads(&model);
  g.ComputeGradients(&grads, expr);

  auto W_grad = g.Gradient(grads, vW);
  auto b_grad = g.Gradient(grads, vb);

  // dexpr/dW = broadcast(x)
  auto W_grad_expected = g.Broadcast<float>(
      g.Transpose(g.Reshape(x, dim_x_m)), dim_w);
  // dexpr/db = 1
  auto b_grad_expected = g.ConstantFromScalar(dim_b, 1.0f);

  g.Eval();

  this->ExpectEq(W_grad, W_grad_expected);
  this->ExpectEq(b_grad, b_grad_expected);
}


TYPED_TEST_P(GradientsTest, TestRandomInitializer) {
  Dimensions dims(16, 16);

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  auto* va = model.NewVariable<float>(
      "a", dims, nullptr, UniformRandomInitializer<float>());
  auto* vb = model.NewVariable<float>(
      "b", dims, nullptr, UniformRandomInitializer<float>());
  auto* vc = model.NewVariable<float>(
      "c", dims, nullptr, NormalRandomInitializer<float>());

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto a = g.Variable(va);
    auto b = g.Variable(vb);
    auto c = g.Variable(vc);
    auto z = g.Zeros<float>(dims);

    g.Eval();

    this->ExpectNotNear(a, b);
    this->ExpectNotNear(a, c);
    this->ExpectNotNear(a, z);
    this->ExpectNotNear(b, z);
    this->ExpectNotNear(c, z);
  }
}


TYPED_TEST_P(GradientsTest, TestSoftmax) {
  Dimensions dim_s;
  Dimensions dim_v(3);

  alignas(16) std::array<float, 3> d = {{ 0.2f, 0.3f, 1.0f }};  // input data
  alignas(16) std::array<float, 3> s = softmax(d);

  // Compute d/dx of expr = ReduceSum(x * softmax(x))
  alignas(16) float grads_expected[3] = {
      static_cast<float>(s[0]*(1 + d[0] - d[0]*s[0] - d[1]*s[1] - d[2]*s[2])),
      static_cast<float>(s[1]*(1 + d[1] - d[1]*s[1] - d[0]*s[0] - d[2]*s[2])),
      static_cast<float>(s[2]*(1 + d[2] - d[2]*s[2] - d[0]*s[0] - d[1]*s[1]))};

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  auto* v = model.NewVariable<float>(
      "v", dim_v, nullptr, TensorInitializer(d.data()));

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  // expr = ReducesSum(x * softmax(x))
  auto v_node = g.Variable(v);
  auto expr = g.Reshape(g.ReduceSum(g.Multiply(v_node, g.Softmax(v_node)),
                                    { 0 }),
                        dim_s);

  Gradients grads(&model);
  g.ComputeGradients(&grads, expr);

  auto v_grad = g.Gradient(grads, v);
  auto v_grad_expected = g.Value(dim_v, grads_expected);

  g.Eval();
  this->ExpectNear(v_grad, v_grad_expected);
}


TYPED_TEST_P(GradientsTest, TestSoftmaxCrossEntropy) {
  Dimensions dim_s;
  Dimensions dim_m(2, 3);

  alignas(16) std::array<std::array<float, 3>, 2> input_data =
      {{ {{ 0.2f, 0.3f, 1.0f }},
         {{ 0.9f, 0.1f, 0.0f }} }};
  alignas(16) std::array<std::array<float, 3>, 2> labels_data =
      {{ {{ 1.0f, 0.0f, 0.0f }},
         {{ 0.1f, 0.2f, 0.7f }} }};
  alignas(16) std::array<std::array<float, 3>, 2> softmax_data =
      {{ softmax(input_data[0]), softmax(input_data[1]) }};

  // Compute d/dx of expr = cross_entropy(softmax(x)).
  float grads_expected[6] = {
      // First element in batch.
      static_cast<float>(softmax_data[0][0] - labels_data[0][0]),
      static_cast<float>(softmax_data[0][1] - labels_data[0][1]),
      static_cast<float>(softmax_data[0][2] - labels_data[0][2]),
      // Second element in batch.
      static_cast<float>(softmax_data[1][0] - labels_data[1][0]),
      static_cast<float>(softmax_data[1][1] - labels_data[1][1]),
      static_cast<float>(softmax_data[1][2] - labels_data[1][2])};

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  auto* v = model.NewVariable<float>(
      "v", dim_m, nullptr, TensorInitializer(&input_data[0][0]));

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  // expr = cross_entropy(labels, softmax(x))
  auto v_node = g.Variable(v);
  auto labels_node = g.Value<float>(dim_m, &labels_data[0][0]);
  auto cross_entropy = g.CrossEntropyLoss(labels_node, g.Softmax(v_node));
  auto expr = g.Reshape(g.ReduceSum(cross_entropy, { 0 }), dim_s);

  Gradients grads(&model);
  g.ComputeGradients(&grads, expr);

  auto v_grad = g.Gradient(grads, v);
  auto v_grad_expected = g.Value<float>(dim_m, grads_expected);

  g.Eval();
  this->ExpectNear(v_grad, v_grad_expected);
}


TYPED_TEST_P(GradientsTest, TestSoftmaxSparseCrossEntropy) {
  Dimensions dim_s;
  Dimensions dim_m(2, 3);
  Dimensions dim_v(2);

  alignas(16) std::array<std::array<float, 3>, 2> input_data
      {{ {{ 0.2f, 0.3f, 1.0f }},
         {{ 0.0f, 0.1f, 0.8f }} }};
  alignas(16) std::array<std::array<float, 3>, 2> labels_data =
      {{ {{ 0.0f, 0.0f, 1.0f }},
         {{ 0.0f, 1.0f, 0.0f }} }};
  alignas(16) std::array<int, 2> sparse_labels = {{ 2, 1 }};

  typename TypeParam::Evaluator evaluator;

  // Expected value: Calculate using (dense) cross-entropy.
  VariableSet model_1(&evaluator);
  auto* v_1 = model_1.NewVariable<float>(
      "v1", dim_m, nullptr, TensorInitializer(&input_data[0][0]));

  Graph g_1 = evaluator.NewGraph(TypeParam::kDevice);

  auto v_node_1 = g_1.Variable(v_1);
  auto labels_node_1 = g_1.Value<float>(dim_m, &labels_data[0][0]);
  auto cross_entropy_1 = g_1.CrossEntropyLoss(labels_node_1,
                                              g_1.Softmax(v_node_1));
  auto expr_1 = g_1.Reshape(g_1.ReduceSum(cross_entropy_1, { 0 }), dim_s);

  Gradients grads_1(&model_1);
  g_1.ComputeGradients(&grads_1, expr_1);
  auto v_grad_1 = g_1.Gradient(grads_1, v_1);
  g_1.Eval();

  // Actual value: Compute sparse cross-entropy.
  VariableSet model_2(&evaluator);
  auto* v_2 = model_2.NewVariable<float>(
      "v2", dim_m, nullptr, TensorInitializer(&input_data[0][0]));

  Graph g_2 = evaluator.NewGraph(TypeParam::kDevice);

  auto v_node_2 = g_2.Variable(v_2);
  auto labels_node_2 = g_2.Value<int>(dim_v, sparse_labels.data());
  auto cross_entropy_2 = g_2.SparseCrossEntropyLoss(labels_node_2,
                                                    g_2.Softmax(v_node_2));
  auto expr_2 = g_2.Reshape(g_2.ReduceSum(cross_entropy_2, { 0 }), dim_s);

  Gradients grads_2(&model_2);
  g_2.ComputeGradients(&grads_2, expr_2);
  auto v_grad_2 = g_2.Gradient(grads_2, v_2);
  g_2.Eval();

  // Check result.
  this->ExpectNear(v_grad_2, v_grad_1);
}


TYPED_TEST_P(GradientsTest, TestConcatSplit) {
  Dimensions dims(16, 64);
  Dimensions dims0(16, 10);
  Dimensions dims1(16, 10);
  Dimensions dims2(16, 15);
  Dimensions dims3(16, 15);
  Dimensions dims4(16, 14);
  Dimensions dims_s;

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  auto* va = model.NewVariable<float>("a", dims, nullptr,
                                      ScalarInitializer(10.0f));

  {
    Graph g = evaluator.NewGraph(TypeParam::kDevice);

    auto a = g.Variable(va);
    auto two = g.ConstantFromScalar(dims0, 2.0f);
    auto three = g.ConstantFromScalar(dims1, 3.0f);
    auto four = g.ConstantFromScalar(dims2, 4.0f);
    auto five = g.ConstantFromScalar(dims3, 5.0f);
    auto six = g.ConstantFromScalar(dims4, 6.0f);

    auto split = g.Split(g.Variable(va), { 10, 10, 15, 15, 14 }, 1);
    auto s0 = g.Multiply(two, g.GetOutput(split, 0).template as<float>());
    auto s1 = g.Multiply(three, g.GetOutput(split, 1).template as<float>());
    auto s2 = g.Multiply(four, g.GetOutput(split, 2).template as<float>());
    auto s3 = g.Multiply(five, g.GetOutput(split, 3).template as<float>());
    auto s4 = g.Multiply(six, g.GetOutput(split, 4).template as<float>());

    auto loss = g.Reshape(g.ReduceSum(g.Concat<float>({ s0, s1, s2, s3, s4 },
                                                      1),
                                      { 0, 1 }),
                          dims_s);

    Gradients grads(&model);
    g.ComputeGradients(&grads, loss);

    auto a_grad = g.Gradient(grads, va);
    auto a_grad_expected = g.Concat<float>({ two, three, four, five, six }, 1);

    g.Eval();
    this->ExpectEq(a_grad, a_grad_expected);
  }
}


TYPED_TEST_P(GradientsTest, TestGatherScatter) {
  Dimensions dims_a(5, 4);
  Dimensions dims_b(2, 4);

  typename TypeParam::Evaluator evaluator;
  VariableSet model(&evaluator);

  auto va = model.NewVariable<float>("a", dims_a, nullptr,
                                     ZerosInitializer<float>());
  auto vb = model.NewVariable<float>("b", dims_b, nullptr,
                                     ZerosInitializer<float>());

  Graph g = evaluator.NewGraph(TypeParam::kDevice);

  // Gather from va.
  auto a = g.Variable(va);
  auto c = g.ConstantFromFunction<float, 2>(
      dims_a, [](int i, int j) -> float { return 100*i + j; });
  auto y = g.Multiply(a, c);

  int32_t indices[2] = { 1, 3 };
  auto idx = g.Value<int32_t>(Dimensions(2), indices);
  auto r_gather  = g.Gather(idx, y);

  // Scatter from vb.
  auto b = g.Variable(vb);
  auto b_scatter = g.Scatter(dims_a[0], idx, b);
  auto r_scatter = g.Multiply(b_scatter, c);

  // Get loss.
  auto rsum_a = g.ReduceSum<float>(r_gather,  Dimensions(1, 1));
  auto rsum_b = g.ReduceSum<float>(r_scatter, Dimensions(1, 1));
  auto loss = g.Reshape(g.Add(rsum_a, rsum_b), Dimensions());

  Gradients grads(&model);
  g.ComputeGradients(&grads, loss);

  auto a_grad = g.Gradient(grads, va);
  auto b_grad = g.Gradient(grads, vb);

  auto a_grad_expected = g.ConstantFromFunction<float, 2>(dims_a,
      [](int i, int j) -> float {
        if (i == 1 || i == 3) {
          return 100*i + j;
        } else {
          return 0;
        }
      });
  auto b_grad_expected = g.ConstantFromFunction<float, 2>(dims_b,
      [](int i, int j) -> float { return 100 + 200*i + j; });

  g.Eval();

  this->ExpectEq(a_grad, a_grad_expected);
  this->ExpectEq(b_grad, b_grad_expected);
}


template<typename Configuration, typename TrainerType>
void TestTrainerHelper() {
  typename Configuration::Evaluator evaluator;
  VariableSet model(&evaluator);
  // Use a small value to avoid test flakiness;
  // small steps should always decrease loss.
  TrainerType trainer(0.0001f);

  Dimensions x_dims(1, 8);

  VarNameSpace* name1 = model.NewNameSpace("Layer1");
  FullyConnectedLayer layer1(name1, 8);

  VarNameSpace* name2 = model.NewNameSpace("Layer2");
  FullyConnectedLayer layer2(name2, 8, FullyConnectedLayer::kLinear);

  float old_loss = 1e+9;
  for (int k = 0; k < 50; ++k) {
    Graph g = evaluator.NewGraph(Configuration::kDevice);

    // Two-layer feed forward net has a simple solution.
    auto x = g.Ones<float>(x_dims);
    auto y1 = layer1(&g, x);
    auto y2 = layer2(&g, y1);
    auto l2_loss = g.Multiply(y2, y2);
    auto loss = g.Reshape(g.ReduceSum(l2_loss, {0, 1}), Dimensions());

    // Copy loss to CPU (may already be CPU).
    auto loss_host = g.CopyToDevice(loss, kDeviceIDCPU);

    Gradients grads(&model);
    g.ComputeGradients(&grads, loss);
    trainer.ApplyGradients(grads);
    g.Eval();

    float new_loss = *loss_host.result_data();
    EXPECT_LT(new_loss, old_loss);
    old_loss = new_loss;
  }
}


TYPED_TEST_P(GradientsTest, TestSGDTrainer) {
  TestTrainerHelper<TypeParam, SGDTrainer>();
}


TYPED_TEST_P(GradientsTest, TestMomentumTrainer) {
  TestTrainerHelper<TypeParam, MomentumTrainer>();
}


REGISTER_TYPED_TEST_CASE_P(GradientsTest,
                           TestScalarGradients,
                           TestNegative,
                           TestReciprocal,
                           TestSigmoid,
                           TestTanh,
                           TestRelu,
                           TestBroadcastReduceSum,
                           TestTranspose,
                           TestMatmul,
                           TestRandomInitializer,
                           TestSoftmax,
                           TestSoftmaxCrossEntropy,
                           TestSoftmaxSparseCrossEntropy,
                           TestConcatSplit,
                           TestGatherScatter,
                           TestSGDTrainer,
                           TestMomentumTrainer);

}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_TEST_GRADIENTS_TEST_H_
