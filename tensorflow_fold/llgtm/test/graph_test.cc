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

#include <sstream>
#include "absl/memory/memory.h"
#include "tensorflow_fold/llgtm/backend/eigen_evaluator_client.h"
#include "tensorflow_fold/llgtm/test/test_framework.h"


namespace llgtm {
namespace {


TEST(GraphTest, TestDimensions) {
  Dimensions dims1(6, 7);
  EXPECT_EQ(dims1.num_elements(), 42);

  llgtm::Dimensions dims2(3, 4, 5);
  int off = 0;
  for (int x = 0; x < 3; ++x) {
    for (int y = 0; y < 4; ++y) {
      for (int z = 0; z < 5; ++z) {
        ASSERT_EQ(dims2.offset(x, y, z), off);
        ++off;
      }
    }
  }
}


TEST(GraphTest, TestDimensionsToString) {
  Dimensions dims0;
  Dimensions dims1(42);
  Dimensions dims2(16, 12);

  EXPECT_EQ(dims0.str(), "()");
  EXPECT_EQ(dims1.str(), "(42)");
  EXPECT_EQ(dims2.str(), "(16, 12)");
}


TEST(GraphTest, TestTensorTypeAlignment) {
  EXPECT_EQ(llgtm::TensorType::aligned_size(0, 8), 0);
  EXPECT_EQ(llgtm::TensorType::aligned_size(40, 8), 40);
  EXPECT_EQ(llgtm::TensorType::aligned_size(64, 16), 64);

  EXPECT_EQ(llgtm::TensorType::aligned_size(1, 16), 16);
  EXPECT_EQ(llgtm::TensorType::aligned_size(17, 16), 32);
  EXPECT_EQ(llgtm::TensorType::aligned_size(115, 16), 128);

  TensorType mytype1(kDTint8, Dimensions(1, 16));
  EXPECT_EQ(mytype1.aligned_memory_size(), 16);

  TensorType mytype2(kDTint8, Dimensions(1, 17));
  EXPECT_EQ(mytype2.aligned_memory_size(), 32);

  TensorType mytype3(kDTint8, Dimensions(10, 102));
  EXPECT_EQ(mytype3.aligned_memory_size(), 1024);
}


TEST(GraphTest, TestIteratorWrapper) {
  struct WInt {
    WInt(int k) : k_(k) {}
    int k_;
  };
  using WIntVec = std::vector<std::unique_ptr<WInt>>;
  using Wrapped = IteratorWrapper<WInt, WIntVec::iterator,
                                  UniquePtrGetter<WInt>>;
  WIntVec v;
  for (int i = 0; i < 10; ++i)  v.emplace_back(absl::make_unique<WInt>(i));

  // prefix test
  int j = 0;
  for (Wrapped it = Wrapped(v.begin()), e = Wrapped(v.end()); it != e; ++it) {
    EXPECT_EQ((*it).k_, j);
    EXPECT_EQ(it->k_, j);
    EXPECT_EQ(&(*it), v[j].get());
    ++j;
  }

  // postfix test
  j = 0;
  for (Wrapped it = Wrapped(v.begin()), e = Wrapped(v.end()); it != e; it++) {
    EXPECT_EQ((*it).k_, j);
    EXPECT_EQ(it->k_, j);
    EXPECT_EQ(&(*it), v[j].get());
    ++j;
  }
}


TEST(GraphTest, TestDump) {
  Dimensions dims;
  EigenEvaluator evaluator;
  float f = 1.0f;

  Graph g = evaluator.NewGraph();

  auto zero = g.Zeros<float>(dims);
  auto one = g.Value(dims, &f);
  auto expr = g.Add(zero, g.Multiply(zero, one));

  std::ostringstream dumpstr;
  g.Dump(dumpstr);
  EXPECT_EQ(dumpstr.str(),
            "n_0 = Zeros() [3]\n"
            "n_1 = Value() [2]\n"
            "n_2 = Multiply(n_0, n_1) [1]\n"
            "n_3 = Add(n_0, n_2) [1]\n");
}


TEST(GraphTest, TestIsDifferentiable) {
  Dimensions dims;
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  auto* va = model.NewVariable<float>(
      "a", dims, nullptr, ZerosInitializer<float>());

  Graph g = evaluator.NewGraph();

  auto zero = g.Zeros<float>(dims);
  auto one = g.ConstantFromScalar(dims, 1.0f);
  auto a = g.Variable(va);

  auto nondiff = g.Sigmoid(g.Add(g.Multiply(one, g.Add(one, one)), zero));
  auto diff1 = g.Add(a, nondiff);
  auto diff2 = g.Sigmoid(g.Multiply(nondiff, a));

  EXPECT_EQ(nondiff.get()->is_differentiable(), false);
  EXPECT_EQ(diff1.get()->is_differentiable(), true);
  EXPECT_EQ(diff2.get()->is_differentiable(), true);

  int graph_size = g.size();

  Gradients grads(&model);
  g.ComputeGradients(&grads, diff1);

  // The only node that should have been added is the initial error.
  EXPECT_EQ(g.size(), graph_size + 1);
}


TEST(GraphTest, TestLayer) {
  Dimensions dims(2, 16);
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  VarNameSpace* name1 = model.NewNameSpace("Layer1");
  VarNameSpace* name2 = model.NewNameSpace("Layer2");
  VarNameSpace* name3 = model.NewNameSpace("Layer3");
  VarNameSpace* name4 = model.NewNameSpace("Layer4");
  VarNameSpace* foo_name = model.NewNameSpace("Foo");

  FullyConnectedLayer layer1(name1, 32);
  FullyConnectedLayer layer2(name2, 10, FullyConnectedLayer::kLinear);
  FullyConnectedLayer layer3(name3, 16, FullyConnectedLayer::kSigmoid);
  FullyConnectedLayer layer4(name4, 64, FullyConnectedLayer::kTanh);

  auto foo_layer = MakeLayerFromFunction(foo_name, 1,
      [](Layer* self, Graph* g,
         const Layer::InputList& inputs, DeviceID device) {
    auto& x = inputs[0].as<float>();
    return g->Add(x, x);
  });

  Graph g = evaluator.NewGraph();
  auto x  = g.Zeros<float>(dims);
  auto y1 = layer1(&g, x);
  auto y2 = layer2(&g, y1);
  auto y3 = layer3(&g, y2);
  auto y4 = layer4(&g, y3);

  auto foo = (*foo_layer)(&g, { y2 });

  EXPECT_EQ(name1->size(), 2);
  EXPECT_EQ(name2->size(), 2);
  EXPECT_EQ(name3->size(), 2);
  EXPECT_EQ(name4->size(), 2);
  EXPECT_EQ(foo_name->size(), 0);

  EXPECT_EQ(y1.dimensions(), Dimensions(2, 32));
  EXPECT_EQ(y2.dimensions(), Dimensions(2, 10));
  EXPECT_EQ(y3.dimensions(), Dimensions(2, 16));
  EXPECT_EQ(y4.dimensions(), Dimensions(2, 64));

  EXPECT_EQ(foo.as<float>().dimensions(), Dimensions(2, 10));

  EXPECT_EQ(y1.get()->opcode(), kOpRelu);
  EXPECT_EQ(y2.get()->opcode(), kOpAdd);     // Linear = w*x + b
  EXPECT_EQ(y3.get()->opcode(), kOpSigmoid);
  EXPECT_EQ(y4.get()->opcode(), kOpTanh);
}


TEST(GraphTest, InputTypeCheckSuccess) {
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  FullyConnectedLayer layer(model.NewNameSpace("Layer"), 32);

  Graph g = evaluator.NewGraph();
  auto x = g.Zeros<float>(Dimensions(1, 32));
  auto y1 = layer(&g, x);
  auto y2 = layer(&g, y1);
  auto y3 = layer(&g, y2);
}


TEST(GraphTest, InputTypeCheckFailureShapeDeathTest) {
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  FullyConnectedLayer layer(model.NewNameSpace("Layer"), 32);

  Graph g = evaluator.NewGraph();
  auto x = g.Zeros<float>(Dimensions(1, 16));
  auto y1 = layer(&g, x);

  EXPECT_DEATH(layer(&g, y1),
               "Type mismatch on layer invocation: expected "
               "\\(float32, \\(1, 16\\)\\), got \\(float32, \\(1, 32\\)\\)");
}


class IdentityLayer : public Layer {
 public:
  explicit IdentityLayer(VarNameSpace* nspace) : Layer(nspace, 1) {}

  TensorBase Invoke(Graph* g, InputList inputs, DeviceID device) override {
    return g->Tuple(inputs);
  }
};

TEST(GraphTest, InputTypeCheckFailureDtypeDeathTest) {
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  IdentityLayer layer(model.NewNameSpace("Layer"));

  Graph g = evaluator.NewGraph();
  auto x1 = g.Zeros<float>(Dimensions(1, 16));
  auto x2 = g.Zeros<int32_t>(Dimensions(1, 16));

  g.Layer(&layer, { x1 });
  EXPECT_DEATH(g.Layer(&layer, { x2 }),
               "Type mismatch on layer invocation: expected "
               "\\(float32, \\(1, 16\\)\\), got \\(int32, \\(1, 16\\)\\)");
}


TEST(GraphTest, InputTypeCheckFailureNumInputsDeathTest) {
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  IdentityLayer layer(model.NewNameSpace("Layer"));

  Graph g = evaluator.NewGraph();
  auto x1 = g.Zeros<float>(Dimensions(1, 16));
  auto x2 = g.Zeros<int32_t>(Dimensions(1, 16));

  g.Layer(&layer, { x1 });
  EXPECT_DEATH(g.Layer(&layer, { x1, x2 }),
               "\\(2 vs. 1\\) ?Layer called with wrong number of arguments");
}


TEST(GraphTest, InputTypeCheckFailureTupleDeathTest) {
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  IdentityLayer layer(model.NewNameSpace("Layer"));

  Graph g = evaluator.NewGraph();
  auto x1 = g.Zeros<float>(Dimensions(1, 16));
  auto x2 = g.Zeros<int32_t>(Dimensions(1, 16));
  auto tup = g.Tuple({ x1, x2 });

  EXPECT_DEATH(g.Layer(&layer, { tup }),
               "Check failed: tensor.is_tensor");
}


class MultiOutputTypeLayer : public Layer {
 public:
  explicit MultiOutputTypeLayer(VarNameSpace* nspace) : Layer(nspace, 1) {}

  TensorBase Invoke(Graph* g, InputList inputs, DeviceID device) override {
    // Returns a different sized output each time.
    ++counter_;
    return g->Zeros<float>(Dimensions(1, counter_*10));
  }

 private:
  int counter_ = 0;
};

TEST(GraphTest, OutputTypeCheckFailureDeathTest) {
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  MultiOutputTypeLayer layer(model.NewNameSpace("Layer1"));

  Graph g = evaluator.NewGraph();
  auto x = g.Zeros<float>(Dimensions(1, 16));
  auto y1 = g.Layer(&layer, { x });

  EXPECT_DEATH(g.Layer(&layer, { x }),
               "Type mismatch on layer invocation: expected "
               "\\(float32, \\(1, 10\\)\\), got \\(float32, \\(1, 20\\)\\)");
}


class MultiOutputNumLayer : public Layer {
 public:
  explicit MultiOutputNumLayer(VarNameSpace* nspace) : Layer(nspace, 1) {}

  TensorBase Invoke(Graph* g, InputList inputs, DeviceID device) override {
    // Returns a different number of outputs each time.
    ++counter_;
    if (counter_ == 1) {
      return g->Tuple({ inputs[0] });
    } else {
      return g->Tuple({ inputs[0], inputs[0] });
    }
  }

 private:
  int counter_ = 0;
};

TEST(GraphTest, OutputTypeCheckFailureNumOutputsDeathTest) {
  EigenEvaluator evaluator;

  VariableSet model(&evaluator);
  MultiOutputNumLayer layer(model.NewNameSpace("Layer1"));

  Graph g = evaluator.NewGraph();
  auto x = g.Zeros<float>(Dimensions(1, 16));
  auto y = g.Layer(&layer, { x });

  EXPECT_DEATH(g.Layer(&layer, { x }),
               "\\(2 vs. 1\\) ?Layer returns wrong number of outputs");
}


}  // namespace
}  // namespace llgtm

