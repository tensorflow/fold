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

#include "tensorflow_fold/llgtm/layers.h"
#include "tensorflow_fold/llgtm/device.h"
#include "tensorflow_fold/llgtm/graph.h"
#include "tensorflow_fold/llgtm/graph_evaluator.h"
#include "tensorflow_fold/llgtm/tensor_ops_impl.h"
#include "tensorflow_fold/llgtm/variable_initializers.h"

namespace llgtm {

namespace {

// Check that ttype matches mtype.
int CheckTypeMatch(const TensorType& ttype, const TensorType& mtype,
                   int batch_size) {
  if (!mtype.matches(ttype)) {
    LOG(FATAL) << "Type mismatch on layer invocation: expected "
               << mtype << ", got " << ttype;
  }
  // Layers must be batch compatible, which means that all inputs must have
  // the same batch size.
  CHECK_GT(ttype.rank(), 0);
  if (batch_size == 0) {   // Infer batch size from first input if bs=0.
    batch_size = ttype.dimension(0);
  } else {
    CHECK_EQ(ttype.dimension(0), batch_size);
  }
  return batch_size;
}

// Check that the type of tensor matches mtype.
inline int CheckTensorType(const TensorBase& tensor, const TensorType& mtype,
                           int batch_size) {
  CHECK(tensor.is_tensor());
  return CheckTypeMatch(tensor.output_type(0), mtype, batch_size);
}

// Check that a type is a valid input or output type.
const TensorType& ValidateTensorType(const TensorType& ttype) {
  // Batch size is the first dimension.
  CHECK_GT(ttype.rank(), 0);
  return ttype;
}

// Check that a type is a valid input or output type.
inline const TensorType& ValidateTensor(const TensorBase& tensor) {
  // All inputs must be ordinary tensors.
  CHECK(tensor.is_tensor());
  return ValidateTensorType(tensor.output_type(0));
}

}  // namespace


int Layer::CheckInputs(InputList inputs) {
  CHECK_EQ(inputs.size(), input_types_.size()) <<
      "Layer called with wrong number of arguments.";
  int batch_size = 0;
  for (int i = 0, n = inputs.size(); i < n; ++i) {
    batch_size = CheckTensorType(inputs[i], input_types_[i], batch_size);
  }
  return batch_size;
}


void Layer::CheckOutputs(const TensorBase& output, int batch_size) {
  CHECK_EQ(has_multiple_outputs_, output.has_multiple_outputs());
  if (output.is_tuple()) {
    // Special case: the tensors in a tuple are stored in its inputs.
    CHECK_EQ(output.num_inputs(), output_types_.size()) <<
        "Layer returns wrong number of outputs.";
    for (int i = 0, n = output.num_inputs(); i < n; ++i) {
      batch_size = CheckTensorType(output.sub_expression(i), output_types_[i],
                                   batch_size);
    }
  } else if (output.has_multiple_outputs()) {
    // Some other multi-output tensor.
    CHECK_EQ(output.num_inputs(), output_types_.size()) <<
        "Layer returns wrong number of outputs.";
    for (int i = 0, n = output.num_inputs(); i < n; ++i) {
      batch_size = CheckTypeMatch(output.output_type(i), output_types_[i],
                                  batch_size);
    }
  } else {
    // Output is a single tensor, as expected.
    CheckTensorType(output, output_types_[0], batch_size);
  }
}


void Layer::Initialize(GraphEvaluator* evaluator, InputList inputs,
                       const TensorBase& output) {
  // Register this layer with the GraphEvaluator.
  layer_id_ = evaluator->get_next_layer_id();

  // Set input types.
  for (const auto& tensor : inputs) {
    input_types_.push_back(ValidateTensor(tensor));
  }
  // Set output types.
  if (output.has_multiple_outputs()) {
    has_multiple_outputs_ = true;
    if (output.is_tuple()) {
      for (int i = 0, n = output.num_inputs(); i < n; ++i) {
        output_types_.push_back(ValidateTensor(output.sub_expression(i)));
      }
    } else {
      for (int i = 0, n = output.num_outputs(); i < n; ++i) {
        output_types_.push_back(ValidateTensorType(output.output_type(i)));
      }
    }
  } else {
    output_types_.push_back(ValidateTensor(output));
  }
}


TensorBase FullyConnectedLayer::Invoke(Graph* g, InputList inputs,
                                       DeviceID /*device*/) {
  DCHECK_EQ(inputs.size(), 1);
  auto& x = inputs[0].as<float>();
  DCHECK_EQ(x.rank(), 2);

  if (!initialized()) {
    int input_size = x.dimension(1);

    // Scale weights according to the number of inputs to maintain constant
    // variance.  Also scale by a factor which is emperically derived from:
    // https://arxiv.org/pdf/1412.6558v3.pdf.
    float stddev = 1.0f/sqrtf(input_size);
    switch (activation_) {
      case kLinear:
        break;
      case kRelu:
        stddev *= sqrtf(2.0f);
        break;
      case kSigmoid:   // TODO(delesley): what should this be?
        break;
      case kTanh:
        stddev *= 1.15f;
        break;
    }

    Dimensions wdims = Dimensions(input_size, num_hidden());
    weights_ = name_space()->NewVariable<float>("weights", wdims,
        NormalRandomInitializer<float>(/*mean=*/ 0.0f, stddev));

    Dimensions bdims = Dimensions(1, num_hidden());
    bias_ = name_space()->NewVariable<float>("bias", bdims,
                                             ZerosInitializer<float>());
  }

  auto weights = g->Variable(weights_);
  auto bias = g->Variable(bias_);
  auto xm = g->Matmul(x, weights);
  auto xmb = g->Add(xm, g->Broadcast(bias, xm.dimensions()));

  switch (activation_) {
    case kLinear:  return xmb;
    case kRelu:    return g->Relu(xmb);
    case kSigmoid: return g->Sigmoid(xmb);
    case kTanh:    return g->Tanh(xmb);
  }
}

}  // namespace llgtm

