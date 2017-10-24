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

// Interface for creating Layers in LLGTM.
//
// A Layer is a function over tensors, which typically has some variables
// that are associated with it.  Common examples are a fully-connected layer,
// or an LSTM cell.
//
// The operations in a layer can be fused together into a single kernel,
// and scheduled as a block, which makes them much faster than adding ops
// to the graph individually.

#ifndef TENSORFLOW_FOLD_LLGTM_LAYERS_H_
#define TENSORFLOW_FOLD_LLGTM_LAYERS_H_

#include "tensorflow_fold/llgtm/device.h"
#include "tensorflow_fold/llgtm/gradients.h"
#include "tensorflow_fold/llgtm/graph.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "absl/memory/memory.h"

namespace llgtm  {

// Base class for Layers.
// Derived classes should override Invoke() to define the layer, and provide
// an implementation of operator().  See FullyConnectedLayer for an example.
class Layer {
 public:
  // InputList is used to pass a list of inputs to Invoke().
  // It is guaranteed to support operator[], and be constructable from
  // std::initializer_list.  Clients should not rely on other operations.
  using InputList = GraphImplementation::InputList;

  Layer() = delete;
  Layer(const Layer& layer) = delete;

  Layer(VarNameSpace* nspace, int num_inputs)
      : name_space_(nspace), num_inputs_(num_inputs) {}

  virtual ~Layer() {}

  Layer& operator=(const Layer& layer) = delete;

  // Invokes the layer on graph g.
  // Inputs is an array of input Tensors.
  // Returns either a single output Tensor, or a Tuple of outputs.
  // This function should not be called directly; use Graph::Layer instead.
  // All variables will be allocated on the given device, if specified.
  virtual TensorBase Invoke(Graph* g, InputList inputs, DeviceID device) = 0;

  VariableSet* model() { return name_space_->variable_set(); }
  VarNameSpace* name_space() { return name_space_; }

  int num_inputs() const { return num_inputs_; }

  // Returns a number that uniquely identifies this layer with the
  // GraphEvaluator.
  int layer_id() const { return layer_id_; }

  // Flag used by layers to initialize variables.
  // Returns false on the first invocation, true thereafter.
  bool initialized() const { return layer_id_ >= 0; }

 private:
  friend class GraphImplementation;

  // Registers this layer with GraphEvaluator, and sets initialized() to true.
  // Subsequent invocations must match the type signature of the first
  // invocation.
  void Initialize(GraphEvaluator* evaluator, InputList inputs,
                  const TensorBase& output);

  // Checks that inputs are of the expected type. The input types are inferred
  // the first time the layer is invoked, and must stay the same thereafter.
  // The type of a tensor includes both the shape of the tensor and its dtype,
  // e.g. (float32, [bsize, 128]). By convention, the first dimension of a
  // tensor (bsize) is assumed to be the batch size. Layers are polymorphic
  // with respect to batch size; that is to say, a layer may be invoked with
  // different batch sizes, so long as the rest of the dimensions and dtypes
  // match. However, all inputs and outputs must have the same batch size on
  // each invocation. This requirement allows us to implement dynamic batching
  // for layers.
  int CheckInputs(InputList inputs);

  // Checks that outputs are of the expected type, and have the same batch size
  // as the inputs.
  void CheckOutputs(const TensorBase& output, int batch_size);

  VarNameSpace* const name_space_;
  const int num_inputs_;

  int layer_id_ = -1;
  bool has_multiple_outputs_ = false;
  std::vector<TensorType> input_types_;
  std::vector<TensorType> output_types_;
};


// A fully connected layer.
// Currently supports only float32 inputs.
// TODO(delesley): Make FCLayer automatically concat multiple inputs.
class FullyConnectedLayer : public Layer {
 public:
  enum ActivationFunction {
    kLinear, kRelu, kSigmoid, kTanh
  };

  FullyConnectedLayer(VarNameSpace* nspace, int num_hidden,
                      ActivationFunction act = kRelu)
    : Layer(nspace, 1), num_hidden_(num_hidden), activation_(act) {}

  TensorBase Invoke(Graph* g, InputList inputs, DeviceID device) override;

  // Layers are functions over tensors, and can be called as such.
  // Using operator() is preferred over calling g->Layer() directly, because
  // type-checking of the arguments can be done at compile-time.
  // Every layer should define operator() according to this pattern.
  Tensor<float> operator()(Graph* g, Tensor<float> x) {
    return g->Layer(this, { std::move(x) }).as<float>();
  }

  Variable<float>* weights() { return weights_; }
  Variable<float>* bias() { return bias_; }

  ActivationFunction activation() const { return activation_; }
  int num_hidden() const { return num_hidden_; }

 private:
  const int num_hidden_;
  const ActivationFunction activation_;

  Variable<float>* weights_ = nullptr;
  Variable<float>* bias_ = nullptr;
};


// Utility class to create layers from lambdas.
template<typename F>
class LayerFromFunction : public Layer {
 public:
  // Creates a layer from a lambda expression.
  // The function f is assumed to have the same signature as Invoke(),
  // with this (of type Layer*) passed as the first argument.
  LayerFromFunction(VarNameSpace* nspace, int num_inputs, F f)
      : Layer(nspace, num_inputs), function_(f) {}

  TensorBase Invoke(Graph* g, InputList inputs, DeviceID device) override {
    return function_(this, g, inputs, device);
  }

  // We don't know anything about F, so operator() defaults to passing
  // inputs as an array.
  TensorBase operator()(Graph* g, InputList inputs,
                        DeviceID device = kDeviceIDUnspecified) {
    return g->Layer(this, inputs, device);
  }

 private:
  F function_;
};


// Helper function to infer the type of F.
template<typename F>
std::unique_ptr<LayerFromFunction<F>> MakeLayerFromFunction(
    VarNameSpace* nspace, int num_inputs, F f) {
  return absl::make_unique<LayerFromFunction<F>>(nspace, num_inputs, f);
}


}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_LAYERS_H_
