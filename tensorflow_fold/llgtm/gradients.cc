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

#include "tensorflow_fold/llgtm/gradients.h"
#include "tensorflow_fold/llgtm/graph.h"
#include "tensorflow_fold/llgtm/tensor_ops_impl.h"

namespace llgtm {

// Find the variable with the given name, or return nullptr if not found.
VariableBase* VarNameSpace::find(const string& name) {
  auto it = variable_map_.find(name);
  if (it != variable_map_.end()) {
    return it->second;
  }
  return nullptr;
}


VarNameSpace* VariableSet::NewNameSpace(string name,  // NOLINT
                                        VarNameSpace* parent) {
  if (parent == nullptr) {
    parent = root_;
  } else {
    CHECK_EQ(parent->variable_set(), this);
  }
  auto* nspace = new VarNameSpace(std::move(name), parent, namespaces_.size());
  namespaces_.emplace_back(nspace);
  return nspace;
}


void VariableSet::CopyNameSpaces(const VariableSet& vset) {
  CHECK_EQ(namespaces_.size(), 1);  // We should only have a root.

  for (const auto& ns : vset.namespaces_) {
    // The parent must have been created before the child, and thus
    // has a lower id.
    if (ns->parent() == nullptr) {
      DCHECK_EQ(ns->id(), 0);  // Only the root should have no parent.
      continue;
    }
    VarNameSpace* parent = namespaces_[ns->parent()->id()].get();
    VarNameSpace* ns_copy = NewNameSpace(ns->name(), parent);
    DCHECK_EQ(ns->id(), ns_copy->id());
  }
}


void Gradients::Clear() {
  gradients_.clear();
  num_uses_.clear();
  sum_nodes_.clear();
  graph_ = nullptr;
}


// Compute gradients for the given loss and graph.
void Gradients::ComputeGradients(Graph* g, Tensor<float> loss) {
  CHECK_EQ(loss.rank(), 0);

  Clear();
  graph_ = g;

  gradients_.resize(variable_set_->size());
  num_uses_.resize(g->size(), 0);
  sum_nodes_.resize(g->size());

  // Traverse the graph backwards, and calculate num_uses for each node.
  std::vector<nodes::TensorNodeBase*> node_stack;
  node_stack.push_back(loss.get());
  ++num_uses_[loss.get()->id()];   // Give loss a use_count of 1.

  while (!node_stack.empty()) {
    nodes::TensorNodeBase* node = node_stack.back();
    node_stack.pop_back();

    TensorBase* subexprs = node->sub_expressions();
    for (int i = 0, n = node->num_inputs(); i < n; ++i) {
      nodes::TensorNodeBase* child = subexprs[i].get();
      if (child->is_differentiable() && num_uses_[child->id()] == 0) {
        node_stack.push_back(child);
      }
      ++num_uses_[child->id()];
    }
  }

  // Now calculate the gradient for each node.
  auto one = g->Ones<float>(Dimensions());

  PropagateError(loss.get(), one);
}


Tensor<float> Gradients::Gradient(VariableBase* v) const {
  if (!has_gradient(v)) {
    return graph_->Zeros<float>(v->dimensions());
  }
  return gradients_[v->id()].as<float>();
}


void Gradients::AddGradient(VariableBase* v, TensorBase grad) {
  DCHECK_EQ(grad.num_outputs(), 1);
  DCHECK_EQ(v->tensor_type(), grad.output_type(0));
  if (!has_gradient(v)) {
    gradients_[v->id()] = grad;
  } else {
    Tensor<float> old_grad = gradients_[v->id()].as<float>();
    gradients_[v->id()] = graph_->Add(old_grad, grad.as<float>());
  }
}


void Gradients::PropagateError(nodes::TensorNodeBase* node,
                               TensorBase error) {
  DCHECK(!node->has_multiple_outputs());

  // TODO(delesley): support gradients that aren't float32.
  if (node->output_type(0).dtype() != kDTfloat32)
    return;
  auto& err = error.as<float>();

  int nid = node->id();
  DCHECK_GT(num_uses_[nid], 0);
  --num_uses_[nid];

  if (num_uses_[nid] == 0) {
    // Recursively compute gradients for children.
    if (sum_nodes_[nid].get() != nullptr) {
      error = graph_->Add(std::move(sum_nodes_[nid].as<float>()), err);
    }
    node->ComputeGradients(error, this);
  } else if (sum_nodes_[nid].get() == nullptr) {
    // Save error in sum_nodes_.
    sum_nodes_[nid] = error;
  } else {
    // Add error to sum_nodes_.
    sum_nodes_[nid] = graph_->Add(sum_nodes_[nid].as<float>(), err);
  }
}


void Gradients::PropagateMultiError(nodes::TensorNodeBase* node,
                                    TensorBase error, int output_index) {
  DCHECK(node->is_valid_output_index(output_index));

  int nid = node->id();
  DCHECK_GT(num_uses_[nid], 0);
  --num_uses_[nid];

  TensorBase error_tuple = sum_nodes_[nid];
  if (error_tuple.get() == nullptr) {
    // Create a tuple to hold incoming gradients from different outputs.
    error_tuple = graph_->Tuple(node->num_outputs(), node->device());
    sum_nodes_[nid] = error_tuple;
    // The inputs of the tuple are set later, and thus may refer to subsequent
    // nodes in the graph.
    graph_->register_out_of_order_node(error_tuple.get());
  } else {
    DCHECK(error_tuple.get()->is_tuple());
  }

  // We only create one GetOutput for each output of a node.
  // See Graph::GetOutput.
  DCHECK_EQ(error_tuple.sub_expression(output_index).get(), nullptr);

  // Store the error in the tuple.
  error_tuple.sub_expression(output_index) = std::move(error);

  if (num_uses_[nid] == 0) {
    node->ComputeGradients(std::move(error_tuple), this);
  }
}


}  // namespace llgtm
