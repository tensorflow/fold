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

// Interface for building and executing graphs of tensor operations.
//
// LLGTM uses a define-by-run semantics.  A model is a C++ program that
// performs a sequence of tensor operations.  The C++ code may interleave
// tensor operations with control-flow, loops, functions, recursion, I/O etc.
// Each time a tensor operation is performed, a node is added to the Graph,
// which allows gradients of the tensor operations to be computed later.
//
// See examples folder for API usage examples.
//
// See design doc at https://go/llgtm-doc for more information.

#ifndef TENSORFLOW_FOLD_LLGTM_GRAPH_H_
#define TENSORFLOW_FOLD_LLGTM_GRAPH_H_

#include <memory>
#include <iostream>
#include <vector>

#include "tensorflow_fold/llgtm/device.h"
#include "tensorflow_fold/llgtm/gradients.h"
#include "tensorflow_fold/llgtm/graph_implementation.h"
#include "tensorflow_fold/llgtm/platform/platform.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/tensor_opcodes.h"
#include "tensorflow_fold/llgtm/tensor_ops.h"
#include "tensorflow_fold/llgtm/util.h"
#include "absl/types/span.h"

namespace llgtm  {

class GraphEvaluator;

// A Graph serves as a factory for building a Tensor AST. The AST is a
// directed acyclic graph. Tensor AST nodes must always be created as part of
// a Graph, which encapsulates the AST, and manages memory for both the AST
// nodes, and for intermediate results.
class Graph {
 public:
  // InputList is used to pass a list of inputs to some operations.
  // It is guaranteed to support operator[], and be constructable from
  // std::initializer_list.  Clients should not rely on other operations.
  using InputList = GraphImplementation::InputList;
  template<class DT> using InputListT = GraphImplementation::InputListT<DT>;

  Graph() = delete;
  Graph(const Graph& g) = delete;
  Graph(Graph&& g) = default;

  explicit Graph(GraphEvaluator* evaluator, GraphImplementation* impl,
                 uint64 seed, bool is_custom_seed, DeviceID device)
          : evaluator_(evaluator), implementation_(impl), seed_(seed),
            is_custom_seed_(is_custom_seed), device_(device) {}

  ~Graph();

  Graph& operator=(const Graph& g) = delete;
  Graph& operator=(Graph&& g) = default;

  // Create a tuple of the given size, with null tensors.
  // Intended mainly for internal use, as the gradient of multi-output tensors.
  TensorBase Tuple(int num_tensors, DeviceID device) {
    TensorBase* inputs = allocate_inputs(num_tensors);
    return emplace_node_base<nodes::Tuple>(num_tensors, inputs, device);
  }

  // Create a tuple of tensors.
  TensorBase Tuple(InputList tensors) {
    TensorBase* inputs = make_inputs(tensors);
    DeviceID device = tensors.empty() ? kDeviceIDUnspecified
                                      : tensors[0].device();
    return emplace_node_base<nodes::Tuple>(tensors.size(), inputs, device);
  }

  // Get the i^th output of a multi-output tensor.
  TensorBase GetOutput(TensorBase a, int i) {
    if (a.is_tuple()) {
      // For tuples, just return the original tensor.
      return a.sub_expression(i);
    } else {
      CHECK(a.get()->is_valid_output_index(i));
      // We pre-allocated GetOutput nodes for multi-output nodes.
      nodes::TensorNodeBase* node = nodes_[a.get()->output_id(i)];
      DCHECK(node->opcode() == kOpGetOutput);
      DCHECK_EQ(node->sub_expression(0).get(), a.get());
      return TensorBase(node);
    }
  }

  // Create a new constant value, from existing data.
  // The value does not assume ownership of data, and may or may not copy it.
  // Callers should not attempt to change data until the graph has executed.
  template<typename DT>
  Tensor<DT> Value(const Dimensions& dims, DT* data,
                   DeviceID device = kDeviceIDUnspecified) {
    auto* otype = make_output_type<DT>(dims);
    return emplace_node<DT, nodes::TensorValue<DT>>(otype, data, device);
  }

  // Return a tensor that represents the value of the given variable.
  template<typename DT>
  Tensor<DT> Variable(Variable<DT>* variable,
                      DeviceID device = kDeviceIDUnspecified) {
    return emplace_node<DT, nodes::TensorVariable<DT>>(variable, device);
  }

  // Adds the given tensor to the given variable.
  // Used by Trainers to apply gradients.
  template<typename DT>
  inline void AssignAdd(llgtm::Variable<DT>* variable, Tensor<DT> a) {
    CHECK_EQ(a.dimensions(), variable->dimensions());
    auto* inputs = make_inputs(std::move(a));
    emplace_node_base<nodes::AssignAdd<DT>>(inputs, variable);
  }

  // Create a new constant value, initialized to zero.
  template<typename DT>
  Tensor<DT> Zeros(const Dimensions& dims,
                   DeviceID device = kDeviceIDUnspecified) {
    auto* otype = make_output_type<DT>(dims);
    return emplace_node<DT, nodes::Zeros<DT>>(otype, device);
  }

  // Create a random constant value with uniform distribution.
  template<typename DT>
  Tensor<DT> UniformRandom(const Dimensions& dims, uint64 seed = 0,
                           DeviceID device = kDeviceIDUnspecified) {
    bool is_custom_seed = seed != 0 || is_custom_seed_;
    uint64 real_seed = seed == 0 ? this->next_seed() : seed;

    auto* otype = make_output_type<DT>(dims);
    return emplace_node<DT, nodes::UniformRandom<DT>>(
        otype, real_seed, is_custom_seed, device);
  }

  // Create a random constant value with normal distribution.
  template<typename DT>
  Tensor<DT> NormalRandom(const Dimensions& dims, uint64 seed = 0,
                          DeviceID device = kDeviceIDUnspecified) {
    bool is_custom_seed = seed != 0 || is_custom_seed_;
    uint64 real_seed = seed == 0 ? this->next_seed() : seed;

    auto* otype = make_output_type<DT>(dims);
    return emplace_node<DT, nodes::NormalRandom<DT>>(
        otype, real_seed, is_custom_seed, device);
  }

  // Create a new constant value, initialized to one.
  template<typename DT>
  Tensor<DT> Ones(const Dimensions& dims,
                  DeviceID device = kDeviceIDUnspecified) {
    return this->ConstantFromScalar<DT>(dims, 1, device);
  }

  // Create a new constant value, intialized with a lambda.
  // f(i,j) will be invoked for each element i,j in the matrix.
  template<typename DT, int R, typename F>
  Tensor<DT> ConstantFromFunction(const Dimensions& dims, F f,
                                  DeviceID device = kDeviceIDUnspecified) {
    auto* otype = make_output_type<DT>(dims);
    return emplace_node<DT, nodes::ConstantFromFunction<DT, R, F>>(
        otype, f, device);
  }

  // Initialize a tensor with a constant value.
  template<typename DT>
  Tensor<DT> ConstantFromScalar(const Dimensions& dims, DT value,
                                DeviceID device = kDeviceIDUnspecified) {
    auto* otype = make_output_type<DT>(dims);
    return emplace_node<DT, nodes::ConstantFromScalar<DT>>(
        otype, value, device);
  }

  // Negate a tensor element-wise.
  template<typename DT>
  Tensor<DT> Negative(Tensor<DT> a) {
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Negative<DT>>(inputs, otype);
  }

  // Add two tensors element-wise.
  template<typename DT>
  Tensor<DT> Add(Tensor<DT> a, Tensor<DT> b) {
    CHECK_EQ(a.dimensions(), b.dimensions());
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a), std::move(b));
    return emplace_node<DT, nodes::Add<DT>>(inputs, otype);
  }

  // Subtract two tensors element-wise.
  template<typename DT>
  Tensor<DT> Subtract(Tensor<DT> a, Tensor<DT> b) {
    return this->Add(std::move(a), this->Negative<DT>(std::move(b)));
  }

  // Invert a tensor element-wise.
  template<typename DT>
  Tensor<DT> Reciprocal(Tensor<DT> a) {
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Reciprocal<DT>>(inputs, otype);
  }

  // Multiply two tensors element-wise.
  template<typename DT>
  Tensor<DT> Multiply(Tensor<DT> a, Tensor<DT> b) {
    CHECK_EQ(a.dimensions(), b.dimensions());
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a), std::move(b));
    return emplace_node<DT, nodes::Multiply<DT>>(inputs, otype);
  }

  // Divide two tensors element-wise.
  template<typename DT>
  Tensor<DT> Divide(Tensor<DT> a, Tensor<DT> b) {
    return this->Multiply(std::move(a), this->Reciprocal<DT>(std::move(b)));
  }

  // Compute element-wise sigmoid of a tensor.
  template<typename DT>
  Tensor<DT> Sigmoid(Tensor<DT> a) {
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Sigmoid<DT>>(inputs, otype);
  }

  // Compute element-wise hyperbolic tangent of a tensor.
  template<typename DT>
  Tensor<DT> Tanh(Tensor<DT> a) {
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Tanh<DT>>(inputs, otype);
  }

  // Compute element-wise ReLU of a tensor.
  template<typename DT>
  Tensor<DT> Relu(Tensor<DT> a) {
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Relu<DT>>(inputs, otype);
  }

  // Compute element-wise first derivative of ReLU of a tensor and multiply
  // by error.
  template<typename DT>
  Tensor<DT> ReluGrad(Tensor<DT> error, Tensor<DT> a) {
    DCHECK_EQ(a.dimensions(), error.dimensions());
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(error), std::move(a));
    return emplace_node<DT, nodes::ReluGrad<DT>>(inputs, otype);
  }

  // Return a tensor with the same number of elements, but different dimensions
  // and rank.
  template<typename DT>
  Tensor<DT> Reshape(Tensor<DT> a, Dimensions dims) {
    auto* otype = make_output_type<DT>(dims);
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Reshape<DT>>(inputs, otype);
  }

  // Broadcast a Tensor over the given dimensions. The contents of Tensor 'a'
  // will be copied/tiled to fill in the larger result tensor.
  // Broadcast does not change rank; Reshape must be used to change rank.
  // a.dimension(i) must either be 1, or equal to dims[i]
  template<typename DT>
  Tensor<DT> Broadcast(Tensor<DT> a, const Dimensions& dims) {
    nodes::Broadcast<DT>::CheckBroadcastDimensions(a.dimensions(), dims);
    auto* otype = make_output_type<DT>(dims);
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Broadcast<DT>>(inputs, otype);
  }

  // Take the sum of rows or columns. The rank of the tensor stays the same.
  template<typename DT>
  Tensor<DT> ReduceSum(Tensor<DT> a, const Dimensions& dims) {
    nodes::ReduceSum<DT>::CheckReduceDimensions(a.dimensions(), dims);
    auto* otype = make_output_type<DT>(dims);
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::ReduceSum<DT>>(inputs, otype);
  }

  // Take the sum of rows or columns. The rank of the tensor stays the same.
  // Indices is a list of dimensions to reduce. For each i in indices,
  // dimension(i) is changed to 1; this is the same as keep_dims=true in TF.
  // Note: Dimensions can be initialized with an std::initializer_list, but
  // the compiler will choose this overload when an initializer list is passed,
  // because this overload is a "perfect fit".
  // See also: https://goo.gl/t6ZMMr
  template<typename DT>
  Tensor<DT> ReduceSum(Tensor<DT> a,
                       const std::initializer_list<int>& indices) {
    auto dims = nodes::ReduceSum<DT>::ReducedDimensions(
        a.dimensions(), indices);
    return ReduceSum(a, dims);
  }

  // Gathers a subset of columns or rows from Tensor 'a', with the indices
  // given by indices.  The dimensions of the output are the same as a, except
  // that the 'axis' dimension is the same size as 'indices'.
  // Indices may contain duplicates.
  template<typename DT>
  Tensor<DT> Gather(Tensor<int32_t> indices, Tensor<DT> a,
                    int axis = 0) {
    CHECK_EQ(indices.rank(), 1);
    CHECK_LE(axis, a.rank());
    auto* otype = make_output_type<DT>(
        nodes::Gather<DT>::OutputDimensions(a.dimensions(), axis,
                                            indices.dimension(0)));
    auto* inputs = make_inputs(std::move(indices), std::move(a));
    return emplace_node<DT, nodes::Gather<DT>>(inputs, otype, axis);
  }

  // Scatters the values in Tensor 'a' over a larger output tensor.  The columns
  // or rows in 'a' will be copied to the positions given by indices, and all
  // other values will be initialized to zero.  The dimensions of the output
  // tensor are the same as a, except the 'axis' dimension is num_rows.
  // Indices may contain duplicates.
  template<typename DT>
  Tensor<DT> Scatter(Dimensions::IndexType num_rows, Tensor<int32_t> indices,
                     Tensor<DT> a, int axis = 0) {
    CHECK_EQ(indices.rank(), 1);
    CHECK_EQ(indices.dimension(0), a.dimension(axis));
    CHECK_LE(axis, a.rank());
    auto* otype = make_output_type<DT>(
        nodes::Gather<DT>::OutputDimensions(a.dimensions(), axis, num_rows));
    auto* inputs = make_inputs(std::move(indices), std::move(a));
    return emplace_node<DT, nodes::Scatter<DT>>(inputs, otype, axis);
  }

  // Transpose (reorder) a tensor according to dim_indices.
  // dim_indices stores the indices of the transposition and not the resulting
  // dimensions.
  template<typename DT>
  Tensor<DT> Transpose(Tensor<DT> a, const DimensionIndices& dim_indices) {
    nodes::Transpose<DT>::CheckTransposeIndices(a.dimensions(), dim_indices);
    auto* otype = make_output_type<DT>(
        nodes::Transpose<DT>::TransposedDimensions(a.dimensions(),
                                                   dim_indices));
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Transpose<DT>>(inputs, otype, dim_indices);
  }

  template<typename DT>
  Tensor<DT> Transpose(Tensor<DT> a) {
    return this->Transpose<DT>(a, { 1, 0 });
  }

  // Multiply two matrices.
  template<typename DT>
  Tensor<DT> Matmul(Tensor<DT> a, Tensor<DT> b) {
    auto* otype = make_output_type<DT>(
        nodes::Matmul<DT>::ResultDimensions(a.dimensions(), b.dimensions()));
    auto* inputs = make_inputs(std::move(a), std::move(b));
    return emplace_node<DT, nodes::Matmul<DT>>(inputs, otype);
  }

  // A concatenation of two or more tensors. The `tensors` Span will be copied
  // and its memory can be released after calling this method.
  template<typename DT>
  Tensor<DT> Concat(InputListT<DT> tensors, int axis) {
    // Compute new dimensions, and do safety checks (only one output per input
    // tensor, at least two inputs, and dimensions match).
    auto* otypes = make_output_type<DT>(
        nodes::Concat<DT>::ConcatenatedDimensions(tensors, axis));
    auto* inputs = make_inputs(tensors);
    return emplace_node<DT, nodes::Concat<DT>>(tensors.size(),
                                               inputs, otypes, axis);
  }

  // A concatenation of two tensors.
  template<typename DT>
  Tensor<DT> Concat(const Tensor<DT>& a, const Tensor<DT>& b, int axis) {
    return this->Concat<DT>({ a, b }, axis);
  }

  // Split a tensor into multiple parts of given sizes along a given axis.
  template<typename DT>
  TensorBase Split(Tensor<DT> a,
                   const absl::Span<const Dimensions::IndexType> in_sizes,
                   int axis,
                   bool copy_sizes = true) {
    // Copy sizes to arena if requested (default: yes).
    auto sizes = copy_sizes ? copy_span_to_arena(in_sizes) : in_sizes;

    auto* output_types = allocate_output_types(sizes.size());
    nodes::Split<DT>::InitializeOutputTypes(a.dimensions(),
                                            output_types, sizes, axis);
    auto* inputs = make_inputs(std::move(a));
    return emplace_node_base<nodes::Split<DT>>(
        inputs, output_types, sizes, axis);
  }

  // Split a tensor into equally-sized parts along a given axis.
  template<typename DT>
  TensorBase Split(Tensor<DT> a, int num_parts, int axis) {
    CHECK_GE(num_parts, 2);

    auto* memory = reinterpret_cast<Dimensions::IndexType*>(
        AllocateInArena(sizeof(Dimensions::IndexType) * num_parts));
    absl::Span<Dimensions::IndexType> sizes =
        absl::MakeSpan(memory, num_parts);

    Dimensions::IndexType part_size = a.dimension(axis) / num_parts;
    for (int i = 0; i < num_parts; ++i) {
      sizes[i] = part_size;
    }

    // Last part might be a bit bigger.
    sizes[num_parts - 1] += a.dimension(axis) % num_parts;

    return Split(a, sizes, axis);
  }

  // Apply softmax to a tensor of rank 1 or rank 2. If the tensor is of rank 2,
  // then the first dimension is the batch dimension and the second dimension
  // is being summed over.
  template<typename DT>
  Tensor<DT> Softmax(Tensor<DT> a) {
    static_assert(std::is_floating_point<DT>::value,
                  "Only floating point tensors allowed for Softmax.");
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::Softmax<DT>>(inputs, otype);
  }

  // Calculate the cross-entropy between two probability distributions `labels`
  // and `probabilities`. This operation currently supports softmax only.
  template<typename DT>
  Tensor<DT> CrossEntropyLoss(Tensor<DT> labels, Tensor<DT> probabilities) {
    DCHECK_EQ(labels.dimensions(), probabilities.dimensions());
    DCHECK_LE(probabilities.dimensions().rank(), 2);

    TensorOpcode opcode = probabilities.get()->opcode();
    auto reduced_dims = nodes::SoftmaxCrossEntropy<DT>::ReducedDimensions(
        probabilities.dimensions());
    auto* otype = make_output_type<DT>(reduced_dims);
    auto* inputs = make_inputs(std::move(labels), std::move(probabilities));

    switch (opcode) {
      case kOpSoftmax:
        return emplace_node<DT, nodes::SoftmaxCrossEntropy<DT>>(inputs, otype);
      default:
        LOG(FATAL) << "Not implemented: Cross entropy for "
                   << opcode_name(opcode);
    }
  }

  // Calculate the sparse cross-entropy between two probability distributions
  // `labels` and `probabilities`. I.e., the first distribution assigns `1.0`
  // to asingle category per batch element. The indices of those categories are
  // specified by `labels`. This operation currently supports softmax only.
  template<typename DT>
  Tensor<DT> SparseCrossEntropyLoss(Tensor<int32_t> labels,
                                    Tensor<DT> probabilities) {
    DCHECK_EQ(labels.dimensions().rank(),
              probabilities.dimensions().rank() - 1);
    DCHECK_LE(probabilities.dimensions().rank(), 2);

    if (probabilities.dimensions().rank() == 2) {
      // `labels` is a scalar for Rank 1 operations.
      DCHECK_EQ(labels.dimension(0), probabilities.dimension(0));
    }

    TensorOpcode opcode = probabilities.get()->opcode();
    auto* otype = make_output_type<DT>(
        nodes::SoftmaxCrossEntropy<DT>::ReducedDimensions(
            probabilities.dimensions()));
    auto* inputs = make_inputs(std::move(labels), std::move(probabilities));

    switch (opcode) {
      case kOpSoftmax:
        return emplace_node<DT, nodes::SoftmaxSparseCrossEntropy<DT>>(inputs,
                                                                      otype);
      default:
        LOG(FATAL) << "Not implemented: Sparse cross entropy for "
                   << opcode_name(opcode);
    }
  }

  // Calculates the gradient of `SparseCrossEntropy(labels, Softmax(logits))`
  // in terms of `logits`.
  template<typename DT>
  Tensor<DT> SoftmaxSparseCrossEntropyGrad(Tensor<int32_t> labels,
                                           Tensor<DT> softmax) {
    DCHECK_EQ(labels.dimensions().rank(), softmax.dimensions().rank() - 1);
    DCHECK_LE(softmax.dimensions().rank(), 2);

    auto* otype = &softmax.tensor_type();
    auto* inputs = make_inputs(std::move(labels), std::move(softmax));
    return emplace_node<DT, nodes::SoftmaxSparseCrossEntropyGrad<DT>>(inputs,
                                                                      otype);
  }

  // Copies a tensor to another device. If the source device equals the
  // destination device, no copy is performed.
  template<typename DT>
  Tensor<DT> CopyToDevice(Tensor<DT> a, DeviceID device) {
    // TODO(matthiasspringer): Implement block copy of memory.
    auto* otype = &a.tensor_type();
    auto* inputs = make_inputs(std::move(a));
    return emplace_node<DT, nodes::CopyToDevice<DT>>(inputs, otype, device);
  }

  // Invokes a layer.
  // An optimizing evaluation engine may JIT-compile code for the layer.
  TensorBase Layer(class Layer* layer, InputList inputs,
                   DeviceID device = kDeviceIDUnspecified) {
    return implementation_->Layer(this, layer, inputs, device);
  }

  // Evaluate all nodes which have been created up until this point.
  // Returns the number of nodes which were evaluated.
  inline int Eval() {
    return implementation_->Eval(this);
  }

  // Compute gradients of the graph with respect to loss.
  void ComputeGradients(Gradients* gradients, Tensor<float> loss) {
    gradients->ComputeGradients(this, std::move(loss));
  }

  // Dump a pretty-printed version of the Graph to out.
  void Dump(std::ostream& out);

  // Return a Tensor that contains the sum of the gradients for variable v.
  template<class DT>
  Tensor<DT> Gradient(const Gradients& gradients, llgtm::Variable<DT>* v) {
    DCHECK_EQ(gradients.graph(), this);
    return gradients.Gradient(v);
  }

  GraphEvaluator* evaluator() { return evaluator_; }

  // Request a new seed value for random number generators. Seed values are
  // deterministic. Initial seed value can be specified upon graph creation.
  uint64 next_seed() {
    // Use Lehmer random number generator to generate a new seed.
    // See also: https://en.wikipedia.org/wiki/Lehmer_random_number_generator.
    seed_ = (seed_ * 279470273UL) % 4294967291UL;
    return seed_;
  }

  int size() const {  return nodes_.size(); }
  int current_node() const { return current_node_; }

  nodes::TensorNodeBase* node(int i) { return nodes_[i]; }
  const nodes::TensorNodeBase* node(int i) const { return nodes_[i]; }

  // Register a node that may refer to subsequent nodes in the graph.
  void register_out_of_order_node(nodes::TensorNodeBase* node) {
    out_of_order_nodes_.push_back(node);
  }

  GraphImplementation* graph_implementation() { return implementation_.get(); }

  DeviceID default_device() const { return device_; }

 private:
  friend class GraphImplementation;

  // Concat::ComputeGradients should be able to access the graph implementation
  // to allocate memory for `sizes` in the arena.
  template<typename DT> friend class nodes::Concat;

  // Split::ComputeGradients should be able to call ConcatTuple.
  template<typename DT> friend class nodes::Split;

  TensorBase CopyToDevice(const TensorBase& a, DeviceID device) {
    // If we hit this line, it is probably a programming error. Users of LLGTM
    // do not usually work with multi-output nodes, but use GetOutput on the
    // result of a multi-output operation.
    LOG(FATAL) << "CopyToDevice not implemented for multi-output nodes.";
  }

  void set_current_node(int i) {
    DCHECK_LE(i, nodes_.size());
    current_node_ = i;
  }

  void* AllocateInArena(size_t size) {
    return implementation_->AllocateInArena(size);
  }

  // Allocate memory for a node of type T.  Does not construct it.
  template<class NodeType>
  void* allocate_node_memory() {
    return AllocateInArena(sizeof(NodeType));
  }

  // Allocate the input expressions for a node.
  TensorBase* allocate_inputs(int num_inputs) {
    void* addr = AllocateInArena(sizeof(TensorBase) * num_inputs);
    return new (addr) TensorBase[num_inputs];
  }

  TensorType* allocate_output_types(int num_outputs) {
    return reinterpret_cast<TensorType*>(
        AllocateInArena(sizeof(TensorType) * num_outputs));
  }

  template<typename DT>
  absl::Span<const DT> copy_span_to_arena(
      const absl::Span<const DT> input) {
    size_t bytes = sizeof(Dimensions::IndexType)*input.size();
    auto arena_memory = reinterpret_cast<Dimensions::IndexType*>(
        AllocateInArena(bytes));
    memcpy(arena_memory, input.data(), bytes);
    return absl::MakeConstSpan(arena_memory, input.size());
  }

  template<typename TensorT>
  TensorT ensure_device_type(DeviceID device, TensorT&& t) {
    if (t.device() != device) {
      // Transfer result to another device. CopyToDevice has overloadings for
      // Tensor<DT> and TensorBase.
      LOG(INFO) << "Inserting CopyToDevice to " << static_cast<int>(device)
                << " for Tensor " << t.get()->type_str();
      return this->CopyToDevice(t, device);
    } else {
      return t;
    }
  }

  // Promotes a node to a given device. Does not promote children.
  // We usually want to update an entire subtree (including children).
  // However, this is not the case during node evaluation, because nodes
  // are evaluated from the leaf upward (increasing node IDs).
  void promote_node_device_shallow(nodes::TensorNodeBase* t, DeviceID device) {
    t->set_device(device);
  }

  // Promotes a node to a given device, including all children.
  void promote_node_device_deep(nodes::TensorNodeBase* t, DeviceID device) {
    for (int i = 0; i < t->num_inputs(); ++i) {
      auto* subnode = t->sub_expression(i).get();

      // Invariant: The device of a child is equal to the device of its
      // parent. The only exception to this are CopyToDevice nodes. Since we
      // are promoting an unspecified node, its children's devices must also
      // be unspecified. (CopyToDevice nodes cannot appear down the path.)
      DCHECK(subnode->device() == kDeviceIDUnspecified ||
             subnode->device() == device);

      promote_node_device_deep(subnode, device);
    }

    promote_node_device_shallow(t, device);
  }

  void promote_input_devices(TensorBase* inputs, int num_inputs) {
    // First pass: Determine to which device unspecified inputs should be
    // promoted.
    DeviceID device_specified = kDeviceIDUnspecified;
    for (int i = 0; i < num_inputs; ++i) {
      if (inputs[i].device() != kDeviceIDUnspecified) {
        if (device_specified == kDeviceIDUnspecified) {
          // Set all unspecified nodes to device of inputs[i].
          device_specified = inputs[i].device();
        } else if (inputs[i].device() != device_specified) {
          LOG(FATAL) << "Input nodes have different devices: "
                     << device_name(inputs[i].device()) << " and "
                     << device_name(device_specified) << ".";
        }
      }
    }

    // Second pass: Promote unspecified inputs.
    for (int i = 0; i < num_inputs; ++i) {
      if (inputs[i].device() != device_specified) {
        promote_node_device_deep(inputs[i].get(), device_specified);
      }
    }
  }

  // Return a pointer to allocated inputs, containing t0.
  TensorBase* make_inputs(TensorBase&& t0) {
    auto* inputs = allocate_inputs(1);
    inputs[0] = std::move(t0);
    return inputs;
  }

  // Return a pointer to allocated inputs, containing t0, t1.
  TensorBase* make_inputs(TensorBase&& t0, TensorBase&& t1) {
    TensorBase* inputs = allocate_inputs(2);
    inputs[0] = std::move(t0);
    inputs[1] = std::move(t1);
    promote_input_devices(inputs, /*num_inputs=*/ 2);
    return inputs;
  }

  // Copy tensors to the arena, and return a pointer to the allocated array.
  TensorBase* make_inputs(InputList tensors) {
    auto* inputs = allocate_inputs(tensors.size());
    for (int i = 0; i < tensors.size(); ++i) {
      inputs[i] = tensors[i];
    }
    promote_input_devices(inputs, tensors.size());
    return inputs;
  }

  // Copy tensors to the arena, and return a pointer to the allocated array.
  template<typename DT>
  TensorBase* make_inputs(InputListT<DT> tensors) {
    auto* inputs = allocate_inputs(tensors.size());
    for (int i = 0; i < tensors.size(); ++i) {
      inputs[i] = tensors[i];
    }
    promote_input_devices(inputs, tensors.size());
    return inputs;
  }

  // Allocate the output types for a node.
  template<typename DT>
  TensorType* make_output_type(const Dimensions& dimensions) {
    void* addr = AllocateInArena(sizeof(TensorType));
    return new (addr) TensorType(CppTypeToDType<DT>::dtype, dimensions);
  }

  void insert_output_nodes(nodes::TensorNodeBase* node) {
    size_t offset = 0;
    auto* node_in = make_inputs(TensorBase(node));
    for (int i = 0; i < node->num_outputs(); ++i) {
      auto* mem_ptr = allocate_node_memory<nodes::GetOutput>();
      auto* output_node = new (mem_ptr) nodes::GetOutput(node_in, i, offset);
      output_node->id_ = nodes_.size();
      nodes_.push_back(output_node);
      offset += node->output_type(i).aligned_memory_size();
    }
  }

  TensorBase insert(nodes::TensorNodeBase* node) {
    node->id_ = nodes_.size();
    nodes_.push_back(node);
    if (node->has_multiple_outputs() && node->num_outputs() > 0) {
      insert_output_nodes(node);
    }
    return TensorBase(node);
  }

  template<class DT>
  Tensor<DT> insert(nodes::TensorNode<DT>* node) {
    node->id_ = nodes_.size();
    nodes_.push_back(node);
    return Tensor<DT>(node);
  }

  template<class NodeType, typename ...Args>
  TensorBase emplace_node_base(Args... args) {
    void* mem_ptr = allocate_node_memory<NodeType>();
    auto* node = new (mem_ptr) NodeType(args...);
    return insert(node);
  }

  template<typename DT, class NodeType, typename ...Args>
  Tensor<DT> emplace_node(Args... args) {
    void* mem_ptr = allocate_node_memory<NodeType>();
    auto* node = new (mem_ptr) NodeType(args...);
    return insert(node);
  }

  // Build a single tensor (concatenation) from a tuple. This concat variant is
  // used during gradient computation of Split.
  template<typename DT>
  Tensor<DT> ConcatTuple(TensorBase* tuple, int axis,
                         const TensorType& output_type) {
    // Not passing `tuple` as const reference because we can't to invoke the
    // non-const version of `get` here.
    DCHECK(tuple->is_tuple());
    return emplace_node<DT, nodes::Concat<DT>>(tuple->num_inputs(),
                                               tuple->get()->sub_expressions(),
                                               &output_type,
                                               axis);
  }

  GraphEvaluator* evaluator_ = nullptr;  // GraphEvaluator for this graph.
  std::unique_ptr<GraphImplementation> implementation_;
  std::vector<nodes::TensorNodeBase*> nodes_;     // Nodes in the graph.
  std::vector<nodes::TensorNodeBase*> out_of_order_nodes_;
  int current_node_ = 0;                // Index of first unevaluated node.

  // Seed value used for the next UniformRandom and NormalRandom (unless
  // different seed is specified upon node creation).
  uint64 seed_;

  // Indicates whether this graph was given a custom seed by the programmer.
  bool is_custom_seed_;

  // Device used for new operations, unless device explicitly specified.
  DeviceID device_;
};


}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_GRAPH_H_
