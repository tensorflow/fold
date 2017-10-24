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

// Base class for the underlying implementation of a Graph.
// Each Evaluator may extend GraphImplementation with additional information
// necessary to evaluate a graph.

#ifndef TENSORFLOW_FOLD_LLGTM_GRAPH_IMPLEMENTATION_H_
#define TENSORFLOW_FOLD_LLGTM_GRAPH_IMPLEMENTATION_H_

#include <iostream>
#include <memory>
#include <vector>

#include "tensorflow_fold/llgtm/platform/platform.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "absl/types/span.h"


namespace llgtm  {

class Graph;
class GraphEvaluator;
class Layer;


// Underlying implementation class for a Graph. A Graph contains a pointer
// to an implementation, so that classes derived from GraphEvaluator may
// instantiate derived versions of GraphImplementation. The implementation
// is responsible for allocating memory and evaluating nodes.
// TODO(delesley): Make this thread-safe.
class GraphImplementation {
 public:
  // InputList is used to pass a list of inputs to Layer() and Layer::Invoke().
  // It is guaranteed to support operator[], and be constructable from
  // std::initializer_list.  Clients should not rely on other operations.
  using InputList = absl::Span<const TensorBase>;
  template<class DT> using InputListT = absl::Span<const Tensor<DT>>;

  static const int kArenaBlockSize = 1 << 16;  // 64 k blocks
  static const int kNodeAlignment = 8;         // Align to 64-bit boundaries
  static const int kResultAlignment = TensorType::kResultAlignment;

  GraphImplementation(const GraphImplementation& g) = delete;
  virtual ~GraphImplementation() {}

  GraphImplementation& operator=(const GraphImplementation& g) = delete;

  // Allocates memory from the arena.
  void* AllocateInArena(size_t size) {
    return arena_.AllocAligned(size, kNodeAlignment);
  }

  // Allocates result data for node, and returns size of allocation.
  virtual size_t AllocateResultData(nodes::TensorNodeBase* node) = 0;

  // Invokes a layer.
  virtual TensorBase Layer(Graph* g, class Layer* layer, InputList inputs,
                           DeviceID device);

  // Evaluate all nodes in the given graph.
  virtual int Eval(Graph* graph);

 protected:
  friend class Graph;
  friend class GraphEvaluator;

  GraphImplementation() : arena_(kArenaBlockSize) {}

  // Exposes set_current_node to subclasses of GraphImplementation.
  void set_current_node(Graph* g, int i);

  // Arena for allocating nodes, inputs, and types.
  platform::Arena arena_;
};


}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_GRAPH_IMPLEMENTATION_H_
