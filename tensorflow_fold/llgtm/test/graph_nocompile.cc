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

#include "tensorflow_fold/llgtm/graph.h"
#include "tensorflow_fold/llgtm/tensor.h"

void failed_type_checks() {
#ifdef TEST_DIMENSIONS
  // These are compile-time errors.
  std::array<llgtm::Dimensions::IndexType, 5> dim_list = {{1, 2, 3, 4, 5}};
  llgtm::Dimensions dims(dim_list);
#endif
}

#ifdef TEST_MULTIPLE_BACKENDS
// Including more than one backend is forbidden. In that case, the linker has
// multiple different TensorNodeSelf::InvokeKernel implementations to choose
// from, violating ODR (One Definition Rule). This test ensures that we show
// a useful compile error message.
#include "tensorflow_fold/llgtm/backend/eigen_evaluator_client.h"
#include "tensorflow_fold/llgtm/backend/tf_evaluator_client.h"
#endif
