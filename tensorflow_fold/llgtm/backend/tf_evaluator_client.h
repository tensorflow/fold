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

#ifndef TENSORFLOW_FOLD_LLGTM_BACKEND_TF_EVALUATOR_CLIENT_H_
#define TENSORFLOW_FOLD_LLGTM_BACKEND_TF_EVALUATOR_CLIENT_H_

// Clients should include this header file if graphs are to be evaluated with
// TensorFlow. Only one backend may be selected per compilation unit.

// See eigen_evaluator.h (which is the reference backend implementation) for
// more documentation on implementation details.

#include "tensorflow_fold/llgtm/backend/tf_evaluator.h"
#include "tensorflow_fold/llgtm/tensor.h"

#ifdef LLGTM_BACKEND_SELECTED
#error "Multiple backends selected. Include only one backend header file."
#else

#define LLGTM_BACKEND_SELECTED Tf

namespace llgtm {

class Graph;

template<class Self>
void nodes::TensorNodeBaseSelf<Self>::InvokeKernel(Graph* graph) {
  CHECK_EQ(this->device(), kDeviceIDCPU);  // TODO(matthiasspringer): GPU support.
  auto* evaluator = reinterpret_cast<TfGraphEvaluator*>(graph->evaluator());
  evaluator->InvokeKernel(reinterpret_cast<Self*>(this), graph);
}

template<typename DT, class Self>
void nodes::TensorNodeSelf<DT, Self>::InvokeKernel(Graph* graph) {
  CHECK_EQ(this->device(), kDeviceIDCPU);  // TODO(matthiasspringer): GPU support.
  auto* evaluator = reinterpret_cast<TfGraphEvaluator*>(graph->evaluator());
  evaluator->InvokeKernel(reinterpret_cast<Self*>(this), graph);
}

#define LLGTM_NODE_DEFINITION(NODE) \
  template class NODE<float>; \
  template class NODE<int32_t>;
#include "tensorflow_fold/llgtm/backend/llgtm_nodes.inc"
#undef LLGTM_NODE_DEFINITION

// Handle classes that are not templatized by data type separately.
template class nodes::TensorNodeBaseSelf<nodes::GetOutput>;

}  // namespace llgtm

#endif  // LLGTM_BACKEND_SELECTED
#endif  // TENSORFLOW_FOLD_LLGTM_BACKEND_TF_EVALUATOR_CLIENT_H_
