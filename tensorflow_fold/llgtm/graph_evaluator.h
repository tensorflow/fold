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

#ifndef TENSORFLOW_FOLD_LLGTM_GRAPH_EVALUATOR_H_
#define TENSORFLOW_FOLD_LLGTM_GRAPH_EVALUATOR_H_

#include "tensorflow_fold/llgtm/graph.h"

namespace llgtm {

// GraphEvaluator is responsible for creating and evaluating graphs.
// A program may have several graphs, but only one GraphEvaluator.
// This class is abstract. Every backend provides its own evaluator.
class GraphEvaluator {
 public:
  GraphEvaluator(const GraphEvaluator& g) = delete;
  virtual ~GraphEvaluator() {}

  GraphEvaluator& operator=(const GraphEvaluator& g) = delete;

  // Return a new Graph, which is managed by this evaluator.
  Graph NewGraph() {
    return Graph(this, this->NewGraphImpl(), this->next_seed(),
                 /*is_custom_seed=*/ false, default_device_);
  }

  // Return a new Graph with a default device.
  Graph NewGraph(DeviceID device) {
    return Graph(this, this->NewGraphImpl(), this->next_seed(),
                 /*is_custom_seed=*/ false, device);
  }

  // Return a new Graph with a seed for random numbers and a default device.
  Graph NewGraph(DeviceID device, uint64 seed) {
    return Graph(this, this->NewGraphImpl(), seed,
                 /*is_custom_seed=*/ true, device);
  }

  // Request a new seed value for graphs. Seed values are deterministic.
  // Initial seed value can be specified upon graph creation.
  uint64 next_seed() {
    // Use (different) Lehmer random number generator to generate a new seed.
    seed_ = (seed_ * 48271UL) % 2147483647UL;
    return seed_;
  }

 protected:
  friend class Graph;
  friend class Layer;
  friend class VariableSet;

  // Creates a new GraphEvaluator.
  GraphEvaluator() {
    // Seed value cannot be zero.
    CHECK_NE(seed_, 0);
  }

  // Creates a new graph evaluator with a default device.
  explicit GraphEvaluator(DeviceID device) : default_device_(device) {
    // Seed value cannot be zero.
    CHECK_NE(seed_, 0);
  }

  // Creates a new graph evaluator with a non-zero seed and a default device.
  GraphEvaluator(DeviceID device, uint64 seed)
      : seed_(seed), is_custom_seed_(true), default_device_(device) {
    // Seed value cannot be zero.
    CHECK_NE(seed_, 0);
  }

  virtual GraphImplementation* NewGraphImpl() = 0;

  // Allocates memory on a given device.
  virtual void* AllocateDeviceMemory(size_t bytes, DeviceID device) = 0;

  // Frees device memory which was previously allocated with using this graph
  // evaluator.
  virtual void FreeDeviceMemory(void* ptr, DeviceID device) = 0;

  // Copies data from the host to a given device.
  // TODO(matthiasspringer): Provide other memcpy flavors.
  virtual void MemcpyHostToDevice(void* destination, void* source,
                                  size_t bytes, DeviceID device) = 0;

  // Returns the number of registered layers.
  int num_registered_layers() const { return current_layer_id_; }

  // Returns the next free layer id.
  int get_next_layer_id() { return current_layer_id_++; }

 private:
  static constexpr uint64 kDefaultSeed = 1;

  // Seed values for graphs constructed from this evalutor will be based of
  // this seed value.
  uint64 seed_ = kDefaultSeed;

  // Indicates whether this seed was provided by the programmer.
  const bool is_custom_seed_ = false;

  // The default device for this evaluator. Used if no device specified during
  // graph creation.
  const DeviceID default_device_ = kDeviceIDCPU;

  // The GraphEvaluator keeps track of the number of registered layers.
  int current_layer_id_ = 0;
};



// Defined here because it depends on the definition of GraphEvaluator.
template<class DT, typename F>
Variable<DT>* VariableSet::NewVariable(string name,  // NOLINT
                                       const Dimensions& dimensions,
                                       VarNameSpace* parent,
                                       F init_function) {
  if (parent == nullptr) parent = root_;
  VariableBase* vbase = parent->find(name);
  if (vbase != nullptr) {
    LOG(ERROR) << "Variable " << name << " already exists.";
    return nullptr;
  }

  // Allocate memory on all devices.
  size_t mem_size = dimensions.num_elements() * sizeof(DT);
  std::vector<void*> data(kDeviceMaximumID);
  for (DeviceID device = kDeviceIDCPU; device < kDeviceMaximumID; ++device) {
    data[device] = evaluator_->AllocateDeviceMemory(mem_size, device);
  }

  {
    // Initialize variable by creating a graph and running it.
    Graph g = evaluator_->NewGraph();
    Tensor<DT> init_value = init_function(&g, dimensions, kDeviceIDCPU);

    if (!(init_value.dimensions() == dimensions)) {
      LOG(FATAL) << "Variable initializer returned a value with dimensions "
                 << init_value.dimensions().str() << "; expecting value with "
                 << "dimensions " << dimensions.str();
    }
    g.Eval();

    // Copy the data over, because the graph will be destroyed.
    for (DeviceID device = kDeviceIDCPU; device < kDeviceMaximumID; ++device) {
      evaluator_->MemcpyHostToDevice(
          data[device], init_value.result_data(), mem_size, device);
    }
  }

  auto* var = new Variable<DT>(std::move(name), parent, variables_.size(),
     TensorType(CppTypeToDType<DT>::dtype, dimensions), std::move(data));
  variables_.emplace_back(var);
  parent->variable_map_.insert({var->name(), var});

  return var;
}


// Defined here because it depends on the definition of GraphEvaluator.
inline VariableSet::~VariableSet() {
  for (auto& v : variables_) {
    for (DeviceID device = 0; device < kDeviceMaximumID; ++device) {
      evaluator_->FreeDeviceMemory(v->data(device), device);
    }
  }
}

}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_GRAPH_EVALUATOR_H_
