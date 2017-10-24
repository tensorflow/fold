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

#ifndef TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_EVALUATOR_H_
#define TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_EVALUATOR_H_

#include "tensorflow_fold/llgtm/backend/eigen_graph_implementation.h"
#include "tensorflow_fold/llgtm/graph_evaluator.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/tensor_ops.h"

namespace Eigen {
class CudaStreamDevice;
class DefaultDevice;
class GpuDevice;
}  // namespace Eigen

namespace llgtm {

// Eigen kernel for a given node.
template<class NodeType>
class EigenKernelDeviceSelector {
 public:
  static void Kernel(NodeType* node, Graph* graph);
};


// Evaluator for the Eigen backend.
class EigenEvaluator : public GraphEvaluator {
 public:
  // Maximum block size for non-Eigen CUDA kernels.
  static constexpr uint32_t kCUDAMaxBlockSize = 1024;

  // Maximum CUDA grid size for non-Eigen CUDA kernels.
  static constexpr uint32_t kCUDAMaxGridSize = 1024;

  EigenEvaluator(DeviceID device, uint64 seed) : GraphEvaluator(device, seed) {
    InitializeDevices();
  }

  explicit EigenEvaluator(DeviceID device) : GraphEvaluator(device) {
    InitializeDevices();
  }

  EigenEvaluator() : GraphEvaluator() { InitializeDevices(); }

  ~EigenEvaluator() override;

  GraphImplementation* NewGraphImpl() override;

  // Import shadowed overloads.
  using GraphEvaluator::NewGraph;

  Eigen::DefaultDevice* default_device() { return default_device_; }
  Eigen::GpuDevice* gpu_device() { return gpu_device_; }

 protected:
  void* AllocateDeviceMemory(size_t bytes, DeviceID device) override;

  void FreeDeviceMemory(void* ptr, DeviceID device) override;

  void MemcpyHostToDevice(void* destination, void* source,
                          size_t bytes, DeviceID device) override;

 private:
  template<class NodeType>
  friend class EigenKernelDeviceSelector;

  template<class Device, class NodeType>
  friend class EigenKernel;

  void InitializeDevices();

  // Cannot use std::unique_ptr, because it requires a full definition of its
  // template argument.
  Eigen::CudaStreamDevice* cuda_stream_device_;
  Eigen::DefaultDevice* default_device_;
  Eigen::GpuDevice* gpu_device_;
};


// Entry point to Eigen kernels. Allocate result memory and run kernel.
template<class NodeType>
void LaunchEigenKernel(NodeType* node, Graph* graph) {
  EigenKernelDeviceSelector<NodeType>::Kernel(node, graph);
}


// We want to overload the following functions by rank, but C++ doesn't
// allow partial template specialization of functions.  We thus introduce
// a dummy argument to specify the rank.
template<int R> class DummyRank_ {};

// All functions and classes that are templatized by a function type (F) must
// remain in a header file. Since the exact function type is only known upon
// template instantiation, we cannot generate Eigen kernels up front, so
// putting them in eigen_evaluator.cc does not work.
template<typename DT, typename F>
void InitializeTensorData(DT* data, const Dimensions& dims, F f,
                          DummyRank_<0>) {
  *data = f();
}

template<typename DT, typename F>
void InitializeTensorData(DT* data, const Dimensions& dims, F f,
                          DummyRank_<1>) {
  for (int i = 0, m = dims[0]; i < m; ++i) {
    data[dims.offset(i)] = f(i);
  }
}

template<typename DT, typename F>
void InitializeTensorData(DT* data, const Dimensions& dims, F f,
                          DummyRank_<2>) {
  for (int i = 0, m = dims[0]; i < m; ++i) {
    for (int j = 0, n = dims[1]; j < n; ++j) {
      data[dims.offset(i, j)] = f(i, j);
    }
  }
}

template<typename DT, typename F>
void InitializeTensorData(DT* data, const Dimensions& dims, F f,
                          DummyRank_<3>) {
  for (int i = 0, m = dims[0]; i < m; ++i) {
    for (int j = 0, n = dims[1]; j < n; ++j) {
      for (int k = 0, o = dims[2]; k < o; ++k) {
        data[dims.offset(i, j, k)] = f(i, j, k);
      }
    }
  }
}


// Eigen kernel definition for ConstantFromFunction. Initializes a (constant)
// tensor using a function F that takes R arguments. This kernel is defined in
// this header file instead of the .cc file (like all other nodes), because
// we cannot force explicit template instantiation for all functions F.
template<typename DT, int Rank, typename F>
class EigenKernelDeviceSelector<nodes::ConstantFromFunction<DT, Rank, F>> {
 public:
  static void Kernel(nodes::ConstantFromFunction<DT, Rank, F>* node,
                     Graph* graph) {
    auto* evaluator = reinterpret_cast<EigenEvaluator*>(graph->evaluator());
    auto* graph_impl = reinterpret_cast<EigenGraphImplementation*>(
        graph->graph_implementation());
    // Avoid virtual method call since we know the exact type.
    graph_impl->EigenGraphImplementation::AllocateResultData(node);
    DCHECK(node->has_result());

    if (node->device() == kDeviceIDCPU) {
      DT* r_data = node->result_data();
      // TODO(delesley): make this into an efficient implementation.
      InitializeTensorData(r_data, node->dimensions(),
                           node->init_function_, DummyRank_<Rank>());
    } else {
      // TODO(matthiasspringer): GPU support.
      LOG(INFO) << "ConstantFromFunction only supported on CPU. Computing on "
                << "CPU and copying to device.";
      size_t bytes = sizeof(DT)*node->dimensions().num_elements();
      DT* temp_data = reinterpret_cast<DT*>(malloc(bytes));
      // TODO(delesley): make this into an efficient implementation.
      InitializeTensorData(temp_data, node->dimensions(),
                           node->init_function_, DummyRank_<Rank>());
      evaluator->MemcpyHostToDevice(node->result_data(), temp_data,
                                    bytes, node->device());
      free(temp_data);
    }
  }
};


}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_EVALUATOR_H_
