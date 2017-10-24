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

#ifndef TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_GRAPH_IMPLEMENTATION_H_
#define TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_GRAPH_IMPLEMENTATION_H_

#include "tensorflow_fold/llgtm/graph_implementation.h"

namespace Eigen {
class DefaultDevice;
class GpuDevice;
}  // namespace Eigen

namespace llgtm {

class EigenGraphImplementation : public GraphImplementation {
 public:
  EigenGraphImplementation(Eigen::DefaultDevice* default_device,
                           Eigen::GpuDevice* gpu_device)
      : GraphImplementation(),
        default_device_(default_device), gpu_device_(gpu_device) {}

  size_t AllocateResultData(nodes::TensorNodeBase* node) override;

  ~EigenGraphImplementation() override;

 private:
  // Devices belong to EigenEvaluator.
  Eigen::DefaultDevice* default_device_;
  Eigen::GpuDevice* gpu_device_;

  std::vector<void*> default_allocations_;
  std::vector<void*> gpu_allocations_;
};

}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_BACKEND_EIGEN_GRAPH_IMPLEMENTATION_H_
