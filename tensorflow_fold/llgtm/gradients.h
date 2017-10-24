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

// Interface for defining variables and computing derivatives with respect
// to variables.

#ifndef TENSORFLOW_FOLD_LLGTM_GRADIENTS_H_
#define TENSORFLOW_FOLD_LLGTM_GRADIENTS_H_

#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow_fold/llgtm/device.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/tensor_ops.h"
#include "tensorflow_fold/llgtm/util.h"
#include "tensorflow_fold/llgtm/platform/platform.h"

namespace llgtm  {

typedef int NameSpaceID;
typedef int VariableID;

class Graph;
class GraphEvaluator;
class VariableSet;
class VarNameSpace;

class VariableBase;
template<class DT> class Variable;


// Base class for Variable and VarNameSpace.
class NamedEntity {
 public:
  NamedEntity() = delete;
  NamedEntity(const NamedEntity& v) = delete;
  NamedEntity(NamedEntity&& v) = delete;
  virtual ~NamedEntity() {}

  NamedEntity& operator=(const NamedEntity& v) = delete;
  NamedEntity& operator=(NamedEntity&& v) = delete;

  const string& name() const { return name_; }
  const VarNameSpace* parent() const { return parent_; }
  VariableSet* variable_set() const { return variable_set_; }

 protected:
  friend class VariableSet;

  inline NamedEntity(string name, VarNameSpace* parent);

  string name_;  // The name of this namespace.  Need not be unique.
  VarNameSpace* parent_;       // The parent name space, if any.
  VariableSet* variable_set_;  // The VariableSet for this entity.
};


// A namespace serves to distinguish variables that might otherwise have
// similar names, like "weights", or "bias".  Namespaces are hierarchical.
// Namespace names need not be unique.
class VarNameSpace : public NamedEntity {
 public:
  NameSpaceID id() const { return id_; }

  // Finds the variable with the given name, or return nullptr if not found.
  VariableBase* find(const string& name);

  // Returns the number of variables in the namespace.
  int size() const { return variable_map_.size(); }

  // Creates a new variable with the given name, dimensions, and namespace.
  // If a variable with the same name already exists, then NewVariable will
  // fail.
  template<class DT, typename F>
  Variable<DT>* NewVariable(string name,
                            const Dimensions& dimensions,
                            F init_function);

  // Creates a new VarNameSpace with the given name.
  inline VarNameSpace* NewNameSpace(string name);

 private:
  friend class VariableSet;

  VarNameSpace(string name, VarNameSpace* parent, NameSpaceID id)
      : NamedEntity(std::move(name), parent), id_(id) {}

  NameSpaceID id_;
  std::unordered_map<string, VariableBase*> variable_map_;
};


// A tensor which is treated as a variable for the purposes of differentiation.
// Variables are not part of a graph; they persist until the model is
// destroyed, and can be used.
class VariableBase : public NamedEntity {
 public:
  VariableID id() const { return id_; }

  const TensorType& tensor_type() const { return tensor_type_; }

  int rank() const { return tensor_type_.rank(); }

  const Dimensions& dimensions() const { return tensor_type_.dimensions(); }

  TensorDataType dtype() const { return tensor_type_.dtype(); }

  void* data(DeviceID device) { return data_[device]; }

  template<class DT>
  Variable<DT>* as() {
    DCHECK(dtype() == CppTypeToDType<DT>::dtype);
    return static_cast<Variable<DT>*>(this);
  }

 protected:
  VariableBase(string name, VarNameSpace* parent, VariableID id,
               const TensorType& ttype, std::vector<void*>&& data)
      : NamedEntity(std::move(name), parent), id_(id), tensor_type_(ttype),
        data_(data) {
    DCHECK_EQ(data_.size(), kDeviceMaximumID);
  }

  VariableID id_;  // A unique id for this variable.
  TensorType tensor_type_;

 private:
  // Pointer to device memory for the variable.
  // TODO(delesley, matthiasspringer): Synchronize data between devices.
  std::vector<void*> data_;
};


template<class DT>
class Variable : public VariableBase {
 public:
  DT* data(DeviceID device) {
    return static_cast<DT*>(VariableBase::data(device));
  }

 private:
  friend class VariableSet;

  Variable(string name, VarNameSpace* parent, VariableID id,
           const TensorType& ttype, std::vector<void*>&& data)
      : VariableBase(std::move(name), parent, id, ttype, std::move(data)) {
    DCHECK(ttype.dtype() == CppTypeToDType<DT>::dtype);
  }
};


// A VariableSet is a container for the Variables (trainable parameters)
// and VarNameSpaces of a model.  It owns the Variables that it contains.
class VariableSet {
 public:
  explicit VariableSet(GraphEvaluator* evaluator, string model_name = "")
      : evaluator_(evaluator), root_(nullptr) {
    root_ = NewNameSpace(std::move(model_name), nullptr);
    root_->variable_set_ = this;
  }
  virtual ~VariableSet();

  // Creates a new variable with the given name, dimensions, and namespace.
  // The default namespace will be used if parent == nullptr.
  // If a variable with the same name already exists, then NewVariable will
  // fail. The variable will be initialized with init_function, which must be
  // of type  (Graph*, Dimensions) -> Tensor<DT>.
  // Returns a pointer to the VarNameSpace, which is owned by this VariableSet.
  // Defined in graph_evaluator.h.
  template<class DT, typename F>
  Variable<DT>* NewVariable(string name,
                            const Dimensions& dimensions,
                            VarNameSpace* parent,
                            F init_function);

  // Creates a new VarNameSpace with the given name and parent, and adds it
  // it to the set.  The default namespace will be used if parent==nullptr.
  // Returns a pointer to the VarNameSpace, which is owned by this VariableSet.
  VarNameSpace* NewNameSpace(string name,
                             VarNameSpace* parent = nullptr);

  // Copies the namespace structure of the given VariableSet.
  void CopyNameSpaces(const VariableSet& vset);

  // Returns the GraphEvaluator for this VariableSet.
  GraphEvaluator* evaluator() { return evaluator_; }

  // Returns the root VarNameSpace.
  VarNameSpace* root() { return root_; }

  // Returns the number of variables in this set.
  int size() const { return variables_.size(); }

  // Returns the i^th variable in this set.
  const VariableBase* operator[](int i) const {
    CHECK_GE(i, 0);
    CHECK_LT(i, variables_.size());
    return variables_[i].get();
  }

  // Returns the number of namespaces in this set.
  int num_namespaces() const { return namespaces_.size(); }

  // Returns the i^th namespace in this set.
  VarNameSpace* get_namespace(int i) {
    CHECK_GE(i, 0);
    CHECK_LT(i, namespaces_.size());
    return namespaces_[i].get();
  }

 private:
  using VariableList = std::vector<std::unique_ptr<VariableBase>>;
  using NameSpaceList = std::vector<std::unique_ptr<VarNameSpace>>;

 public:
  // The type of iterators, with value_type VariableBase.
  using iterator = IteratorWrapper<VariableBase,
                                   VariableList::iterator,
                                   UniquePtrGetter<VariableBase>>;
  // The type of const interators, with value_type const VariableBase.
  using const_iterator = IteratorWrapper<const VariableBase,
                                         VariableList::const_iterator,
                                         UniquePtrGetter<VariableBase>>;

  // Returns an iterator over the variables in this set.
  iterator begin() { return iterator(variables_.begin()); }
  iterator end() { return iterator(variables_.end()); }

  // Returns a const_iterator over the variables in this set.
  const_iterator begin() const { return const_iterator(variables_.begin()); }
  const_iterator end() const { return const_iterator(variables_.end()); }

 private:
  friend class Gradients;

  GraphEvaluator* const evaluator_;
  VarNameSpace* root_;
  NameSpaceList namespaces_;  // The namespaces in this VariableSet.
  VariableList variables_;    // The variables in this VariableSet.
};


// Gradients contains a list of gradients for each Variable in a VariableSet.
// The gradients are typically summed before applying them.
// Gradients are calculated symbolically with respect to a Graph, and are
// Tensor nodes in that graph.
class Gradients {
 public:
  Gradients() = delete;
  Gradients(const Gradients& g) = delete;
  Gradients& operator=(const Gradients& g) = delete;

  // Construct a new set of gradients for the given VariablesSet.
  explicit Gradients(VariableSet* varset)
    : graph_(nullptr), variable_set_(varset) { CHECK_NOTNULL(varset); }

  // Returns the number of variables in the set.
  int size() { return gradients_.size(); }

  // Returns true if there is a gradient for v, false otherwise.
  bool has_gradient(VariableBase* v) const {
    DCHECK_EQ((*variable_set_)[v->id()], v);
    return gradients_[v->id()].get() != nullptr;
  }

  // Returns a tensor in g that holds the gradient of v.
  Tensor<float> Gradient(VariableBase* v) const;

  // Returns the graph that these gradients are allocated in.
  Graph* graph() const { return graph_; }

  // Returns the VariableSet associated with these gradients.
  VariableSet* variable_set() const { return variable_set_; }

  // Clears all gradients, allowing this object to be used in a new Graph.
  // For safety, clearing should be always done before the Graph is destroyed;
  // otherwise Gradients will have dangling pointers to the deleted Graph.
  void Clear();

  // Computes gradients for the given loss and graph.
  void ComputeGradients(Graph* g, Tensor<float> loss);

  // Adds a gradient for the given variable.
  void AddGradient(VariableBase* v, TensorBase grad);

  // Propagates the error for the given tensor node.
  // Called by implementations of TensorNodeBase::ComputeGradients.
  void PropagateError(nodes::TensorNodeBase* node, TensorBase error);

  // Propagates the error to a node with multiple outputs, where error
  // is the error for the output_index^th output of the node.
  void PropagateMultiError(nodes::TensorNodeBase* node,
                           TensorBase error, int output_index);

 private:
  Graph* graph_;
  VariableSet* const variable_set_;

  std::vector<TensorBase> gradients_;   // Indexed by variable id.
  std::vector<int> num_uses_;           // Indexed by node id.
  std::vector<TensorBase> sum_nodes_;   // Indexed by node id.
};


NamedEntity::NamedEntity(string name, VarNameSpace* parent)
    : name_(std::move(name)), parent_(parent), variable_set_(nullptr) {
  if (parent != nullptr) variable_set_ = parent->variable_set_;
}


template<class DT, typename F>
Variable<DT>* VarNameSpace::NewVariable(string name,     // NOLINT
                                        const Dimensions& dimensions,
                                        F init_function) {
  return variable_set()->NewVariable<DT>(
      std::move(name), dimensions, this, init_function);
}


VarNameSpace* VarNameSpace::NewNameSpace(string name) {  // NOLINT
  return variable_set()->NewNameSpace(std::move(name), this);
}


}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_GRADIENTS_H_
