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

#include "tensorflow_fold/llgtm/backend/eigen_graph_implementation.h"

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace llgtm {

size_t EigenGraphImplementation::AllocateResultData(
    nodes::TensorNodeBase* node) {
  if (!node->allocates_result_data()) {
    return 0;
  }

  size_t size = node->result_data_size();
  if (size > 0) {
    void* rdata = nullptr;
    switch (node->device()) {
      case kDeviceIDCPU:
        rdata = default_device_->allocate(size);
        default_allocations_.push_back(rdata);
        break;
      case kDeviceIDGPU:
  #ifdef GOOGLE_CUDA
        rdata = gpu_device_->allocate(size);
        gpu_allocations_.push_back(rdata);
  #else
        LOG(FATAL) << "LLGTM built without CUDA support. Use --config=cuda.";
  #endif  // GOOGLE_CUDA
        break;
      default:
        LOG(FATAL) << "Device not supported: " << device_name(node->device())
                   << ".";
    }
    DCHECK(rdata);
    node->set_result_data(rdata);
  }
  return size;
}


EigenGraphImplementation::~EigenGraphImplementation() {
  for (void* d : default_allocations_) {
    default_device_->deallocate(d);
  }
#ifdef GOOGLE_CUDA
  for (void* d : gpu_allocations_) {
    gpu_device_->deallocate(d);
  }
#endif  // GOOGLE_CUDA
}

}  // namespace llgtm
