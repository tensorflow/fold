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

#ifndef TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_EVALUATOR_CLIENT_H_
#define TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_EVALUATOR_CLIENT_H_

// Clients should include this header file if graphs are to be evaluated with
// Eigen. Only one backend may be selected per compilation unit.

// This evaluator is the "reference" implementation with implementations for
// all operations. Other evaluators can fall back onto this evaluator if they
// don't provide an implementation for an operation.

// The entry point into the evaluator during graph evaluation is InvokeKernel
// in TensorNodeSelf<DT, Self>, where Self is the type of the node, e.g.,
// Add<DT>. This method dispatches to the appropriate kernel implementation
// EigenKernel<Self>::Kernel, e.g., EigenKernel<Add<DT>>::Kernel. Kernels are
// declared in eigen_evaluator.h but defined in eigen_evaluator.cc, such that
// including (selecting) an evaluator header file does not transitively include
// any implementation-specific header such as the Eigen headers.

// Kernels are wrapped in EigenKernel<NodeType> such that other evaluators can
// include eigen_evaluator.h and fall back on its kernel implementations.
// (As opposed to putting kernel code in InvokeKernel directly.)

// Only one evaluator header file may be included per translation unit because
// every evaluator provides different implementations of InvokeKernel for
// TensorNode subclasses. Including multiple evaluators is in violation of the
// C++ One Definition Rule and leads to a compile error if detected.


#include "tensorflow_fold/llgtm/backend/eigen_evaluator.h"
#include "tensorflow_fold/llgtm/tensor.h"

#ifdef LLGTM_BACKEND_SELECTED
#error "Multiple backends selected. Include only one backend header file."
#else

#define LLGTM_BACKEND_SELECTED Eigen

namespace llgtm {

class Graph;

template<class Self>
void nodes::TensorNodeBaseSelf<Self>::InvokeKernel(Graph* graph) {
  LaunchEigenKernel<Self>(reinterpret_cast<Self*>(this), graph);
}

template<typename DT, class Self>
void nodes::TensorNodeSelf<DT, Self>::InvokeKernel(Graph* graph) {
  LaunchEigenKernel<Self>(reinterpret_cast<Self*>(this), graph);
}

#define LLGTM_NODE_DEFINITION(NODE) \
  template class NODE<float>; \
  template class NODE<int32_t>;
#define LLGTM_NODE_DEFINITION_FP(NODE) \
  template class NODE<float>;
#include "tensorflow_fold/llgtm/backend/llgtm_nodes.inc"
#undef LLGTM_NODE_DEFINITION

// Handle classes that are not templatized by data type separately.
template class nodes::TensorNodeBaseSelf<nodes::GetOutput>;

}  // namespace llgtm

#endif  // LLGTM_BACKEND_SELECTED
#endif  // TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_EVALUATOR_CLIENT_H_
