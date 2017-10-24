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

#include "tensorflow_fold/llgtm/layers.h"
#include "tensorflow_fold/llgtm/tensor_ops_impl.h"

namespace llgtm {

TensorBase GraphImplementation::Layer(
    Graph* g, class Layer* layer, InputList inputs, DeviceID device) {
  // Default implementation merely adds the nodes to the graph.
  if (!layer->initialized()) {
    // On first invocation, initialize the layer and figure out the types.
    CHECK_EQ(inputs.size(), layer->num_inputs());
    TensorBase result = layer->Invoke(g, inputs, device);
    layer->Initialize(g->evaluator(), inputs, result);
    return result;
  } else {
    // On subsequent invocations, check the types.
    int batch_size = layer->CheckInputs(inputs);
    TensorBase result = layer->Invoke(g, inputs, device);
    layer->CheckOutputs(result, batch_size);
    return result;
  }
}


// Evaluate all nodes up to the current node.
// Eventually we'll block on the GraphExecutor to do this, but the initial
// prototype version is single-threaded.
int GraphImplementation::Eval(Graph* graph) {
  int current = graph->current_node();
  int gsize = graph->size();
  for (int i = current; i < gsize; ++i) {
    nodes::TensorNodeBase* node = graph->node(i);

    // If device is still unspecified, use default graph device.
    if (node->device() == kDeviceIDUnspecified) {
      graph->promote_node_device_shallow(node, graph->default_device());
    }

    DCHECK_EQ(node->id(), i);
    node->InvokeKernel(graph);
  }
  int num_evaluated = gsize - current;
  set_current_node(graph, gsize);
  return num_evaluated;
}


void GraphImplementation::set_current_node(Graph* g, int i) {
  g->set_current_node(i);
}


Graph::~Graph() {
#ifndef NDEBUG
  // Check that there are no dangling references.
  // Release refcounts from out of order nodes.
  for (nodes::TensorNodeBase* node : out_of_order_nodes_) {
    for (int j = 0, ni = node->num_inputs(); j < ni; ++j) {
      node->sub_expression(j).release();
    }
  }
  // Destroy nodes in inverse order so the ref counts work out.
  for (int i = nodes_.size() - 1; i >= 0; --i) {
    nodes::TensorNodeBase* node = nodes_[i];
    for (int j = 0, ni = node->num_inputs(); j < ni; ++j) {
      node->sub_expression(j).release();
    }
    node->~TensorNodeBase();  // Destructor checks refcount.
  }
#endif  // NDEBUG
}


void Graph::Dump(std::ostream& out) {
  for (nodes::TensorNodeBase* node : nodes_) {
    out << "n_" << node->id() << " = " << opcode_name(node->opcode());
    out << "(";

    auto* s_iter = node->sub_expressions();
    auto* s_end = s_iter + node->num_inputs();
    const char* sep = "";
    for (; s_iter != s_end; ++s_iter) {
      out << sep << "n_" << s_iter->get()->id();
      sep = ", ";
    }
    out << ")";
    out << " [" << node->num_uses_ << "]";
    out << "\n";
  }
}

}  // namespace llgtm
