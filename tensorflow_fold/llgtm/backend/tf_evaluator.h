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

#ifndef TENSORFLOW_FOLD_LLGTM_BACKEND_TF_EVALUATOR_H_
#define TENSORFLOW_FOLD_LLGTM_BACKEND_TF_EVALUATOR_H_

#include "tensorflow_fold/llgtm/backend/eigen_evaluator.h"
#include "tensorflow_fold/llgtm/graph_evaluator.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/tensor_ops.h"

namespace tensorflow {
class Device;
class DeviceMgr;
};

namespace llgtm {

class TfKernelAdapter;

class TfGraphEvaluator : public EigenEvaluator {
 public:
  using EigenEvaluator::NewGraph;

  TfGraphEvaluator(DeviceID device, uint64 seed);
  explicit TfGraphEvaluator(DeviceID device);
  explicit TfGraphEvaluator();
  ~TfGraphEvaluator() override;

  void Init();

  GraphImplementation* NewGraphImpl() override;

 private:
  // TfGraphImplementation is defined in tf_evaluator.cc to avoid poluting this
  // header file with TF dependencies.
  friend class TfGraphImplementation;
  template<class Self> friend class nodes::TensorNodeBaseSelf;
  template<class Self, typename DT> friend class nodes::TensorNodeSelf;

  void register_tf_tensor(nodes::TensorNodeBase* node, Graph* graph);

  // Default implementation: If no other specialization is available, fall back
  // to the corresponding Eigen kernel.
  template<typename NodeType>
  void InvokeKernel(NodeType* node, Graph* graph) {
    // Print a log message if a TF kernel was requested for a certain node type
    // but is not currently implemented. Exceptions are ConstantFromFunction,
    // GetOutput and Value: These operations will always use the Eigen kernel.
    if (node->opcode() != kOpConstantFromFunction &&
        node->opcode() != kOpGetOutput &&
        node->opcode() != kOpValue) {
      LOG(INFO) << "No TF kernel available for " << node->type_str() << ".";
    }

    // Just a sanity check: Ensure that we do not fall back to an Eigen kernel
    // if a node type is in tf_nodes.inc, i.e., it should have a TF kernel.
    #define LLGTM_TF_KERNEL_DEFINITION(NODETYPE) \
        static_assert(!std::is_same<NODETYPE<float>, NodeType>::value, \
                      "Node in tf_nodes.inc, but no specialization applies.");
    #include "tensorflow_fold/llgtm/backend/tf_nodes.inc"
    #undef LLGTM_TF_KERNEL_DEFINITION

    LaunchEigenKernel<NodeType>(node, graph);
    register_tf_tensor(node, graph);
  }

  // TF kernel declarations.
  #define LLGTM_TF_KERNEL_DEFINITION(NODETYPE) \
    void InvokeKernel(NODETYPE<float>* node, Graph* graph);
  #include "tensorflow_fold/llgtm/backend/tf_nodes.inc"
  #undef LLGTM_TF_KERNEL_DEFINITION

  // Helper method that can invoke either UniformRandom or NormalRandom.
  template<typename NodeType>
  void InvokeRandomKernel(NodeType* node, Graph* graph,
                          TfKernelAdapter* kernel);

  std::unique_ptr<tensorflow::DeviceMgr> device_mgr_;

  // Note: these devices are owned by the above device_mgr and cached here.
  std::vector<tensorflow::Device*> devices_;

  // Map from opcode to tensorflow kernel
  std::vector<std::unique_ptr<TfKernelAdapter>> kernels_;
};

}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_BACKEND_TF_EVALUATOR_H_
