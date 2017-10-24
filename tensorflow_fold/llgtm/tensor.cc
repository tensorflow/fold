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

#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/gradients.h"
#include "tensorflow_fold/llgtm/graph.h"

namespace llgtm {
namespace nodes {

void TensorNodeBase::set_device(DeviceID device) {
  device_ = device;
}

// Tuples cannot be consumed directly.
void Tuple::InvokeKernel(Graph* graph) {}

// Tuples cannot be consumed directly.
void Tuple::ComputeGradients(TensorBase error, Gradients* gradients) {
  LOG(FATAL) << "Cannot pass gradients through a Tuple.";
}

void GetOutput::ComputeGradients(TensorBase error, Gradients* gradients) {
  TensorNodeBase* multi = sub_expression(0).get();
  if (multi->is_differentiable()) {
    gradients->PropagateMultiError(multi, error, output_index_);
  }
}


}  // namespace nodes
}  // namespace llgtm

