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

// Interface for Tensors.
//
// LLGTM is a library for performing operations and doing symbolic
// differentiation on tensors, where a tensor is a multi-dimensional array.
// Whenever an operation is performed, LLGTM will add a TensorNode to a Graph
// (see Graph.h), to represent the operation. It then returns a Tensor, which
// is a reference to the node. The Graph thus keeps a record of a sequence of
// operations which can be differentiated later.
//
// Subclasses of TensorNode define the various operations that are supported
// by the library. Each subclass implements methods for performing the tensor
// computation associated with that node, and also implements a gradient()
// method for performing the automatic differentiation.
//
// User code should not manipulate pointers to TensorNodes directly. A Tensor
// is a handle to a TensorNode that can be manipulated in user code.
//
// See examples folder for API usage examples.
//
// See design doc at https://go/llgtm-doc for more information.


#ifndef TENSORFLOW_FOLD_LLGTM_TENSOR_H_
#define TENSORFLOW_FOLD_LLGTM_TENSOR_H_

#include <sstream>
#include "tensorflow_fold/llgtm/device.h"
#include "tensorflow_fold/llgtm/dimensions.h"
#include "tensorflow_fold/llgtm/platform/platform.h"
#include "tensorflow_fold/llgtm/tensor_opcodes.h"

namespace llgtm  {

class Gradients;
class Graph;
class GraphImplementation;
class GraphEvaluator;

class TensorBase;
template<typename DT> class Tensor;
template<typename DT> class Variable;


namespace nodes {  // Internal namespace for Node definitions.

template<typename DT> class TensorNode;


// Non-templatized base class for tensor AST nodes.
// Client code should never refer to AST nodes directly; use the TensorBase or
// Tensor classes, defined below.
//
// TODO(delesley): Make reference counting thread-safe.
class TensorNodeBase {
 public:
  static const uint16_t kAllocatesResultData = 0x01;
  static const uint16_t kDifferentiable = 0x02;
  static const uint16_t kHasMultipleOutputs = 0x04;

  // Nodes are allocated in graphs, and are non-copyable.
  TensorNodeBase() = delete;
  TensorNodeBase(const TensorNodeBase&) = delete;
  TensorNodeBase(TensorNodeBase&&) = delete;
  TensorNodeBase& operator=(const TensorNodeBase&) = delete;
  TensorNodeBase& operator=(TensorNodeBase&&) = delete;

  // The opcode for this tensor node.
  TensorOpcode opcode() const { return opcode_; }

  int num_inputs() const { return num_inputs_; }
  int num_outputs() const { return num_outputs_; }

  // Return an integer that uniquely identifies this node in the graph.
  int id() const { return id_; }

  // Return an integer that uniquely identifies an individual output.
  int output_id(int out_i) const {
    // Output nodes are allocated immediately after this node in the graph.
    DCHECK_LT(out_i, num_outputs_);
    if (!has_multiple_outputs()) {
      return id_;
    } else {
      return id_ + out_i + 1;
    }
  }

  // Returns a pointer to an array of nodes that hold the inputs to this node.
  TensorBase* sub_expressions() { return input_nodes_; }
  const TensorBase* sub_expressions() const { return input_nodes_; }

  inline TensorBase& sub_expression(int i);
  inline const TensorBase& sub_expression(int i) const;

  // Returns a pointer to an array of types for the outputs of this node.
  const TensorType* output_types() const { return output_types_; }

  // Returns the type of the i^th output.
  const TensorType& output_type(int i) const {
    DCHECK_LT(i, num_outputs_);
    return output_types_[i];
  }

  DeviceID device() const { return device_; }

  // Downcast this to the appropriate typed TensorNode.
  template<typename DT>
  const TensorNode<DT>* as() const {
    DCHECK_EQ(num_outputs_, 1);
    DCHECK(output_types_->dtype() == CppTypeToDType<DT>::dtype);
    return reinterpret_cast<const TensorNode<DT>*>(this);
  }

  template<typename DT>
  TensorNode<DT>* as() {
    DCHECK_EQ(num_outputs_, 1);
    DCHECK(output_types_->dtype() == CppTypeToDType<DT>::dtype);
    return reinterpret_cast<TensorNode<DT>*>(this);
  }

  // Invoke the kernel to evaluate this operation.
  virtual void InvokeKernel(Graph* graph) = 0;

  // Compute the symbolic gradient of this operation.
  virtual void ComputeGradients(TensorBase error,
                                Gradients* gradients) = 0;

  // Return a device pointer to the underlying tensor data.
  void* result_data() { return result_data_; }
  const void* result_data() const { return result_data_; }

  // Return true if the result for this node has been computed.
  bool has_result() const { return result_data_ != nullptr; }

  // Return the offset into result_data for the i^th output.
  // Results are allocated sequentially in memory. Only for testing.
  size_t result_data_offset(int i) const {
    DCHECK_LT(i, num_outputs_);
    return TensorType::memory_offset(i, output_types_);
  }

  // Return the size of the result data in memory.
  size_t result_data_size() {
    return TensorType::total_memory_size(num_outputs_, output_types_);
  }

  // Return true if this node is a tensor -- i.e. has a single tensor output.
  bool is_tensor() const { return !has_multiple_outputs(); }

  // Return true if this node is a tuple.
  bool is_tuple() const { return opcode_ == kOpTuple; }

  // Return true (the default) if this node allocates data.
  bool allocates_result_data() const {
    return (flags_ & kAllocatesResultData) != 0;
  }

  // Return false if this node is constant, otherwise return true.
  bool is_differentiable() const {
    return (flags_ & kDifferentiable) != 0;
  }

  // Return true if this node has multiple outputs.  Such nodes take tuples
  // of tensors as gradients, and can be used with GetOutput.
  bool has_multiple_outputs() const {
    // We use a flag to distinguish the case where a node conceptually has
    // multiple outputs, but a particular instance only has one output.
    // I.e. the node expects a tuple of length one as a gradient.
    return (flags_ & kHasMultipleOutputs) != 0;
  }

  // Return true if idx is a valid output index into a multi-output tensor.
  bool is_valid_output_index(int idx) {
    if (!has_multiple_outputs()) return false;
    return idx < num_outputs_;
  }

  // Set the set the underlying data pointer for this node.
  void set_result_data(void* data) {
    DCHECK_EQ(result_data_, nullptr);  // The result should only be set once.
    result_data_ = data;
  }

  // Return a debug string describing the type of this node.
  virtual string type_str() const {
    std::ostringstream outstr;
    outstr << opcode_name(opcode_) << "@" << static_cast<int>(device());
    return outstr.str();
  }

 protected:
  void set_flag(int flag_name, bool value) {
    if (value) {
      flags_ |= flag_name;
    } else {
      flags_ &= ~flag_name;
    }
  }

  // Mark whether result_data_ is allocated by this node. (Default is true.)
  void set_allocates_result_data(bool value) {
    set_flag(kAllocatesResultData, value);
  }

  // Mark whether result_data_ is allocated by this node. (Default is true.)
  void set_differentiable(bool value) {
    set_flag(kDifferentiable, value);
  }

  // Mark whether this node has multiple outputs.
  void set_multiple_outputs(bool value) {
    set_flag(kHasMultipleOutputs, value);
  }

  void set_device(DeviceID device);

 protected:
  friend class ::llgtm::TensorBase;
  friend class ::llgtm::GraphImplementation;
  friend class ::llgtm::Graph;

  // Constructor.
  inline TensorNodeBase(TensorOpcode op, int num_inputs, int num_outputs,
                        TensorBase* inputs, const TensorType* output_types,
                        DeviceID device);

  // Only a Graph can create or delete a node.
  virtual ~TensorNodeBase() {
    // Sanity check that the reference counting is right.
    DCHECK_EQ(num_uses_, 0);
  }

  // Reference count is int32_t.
  static const int kMaxNumberOfUses = (1 << 30) - 1;

  inline void attach() {
    DCHECK_LT(num_uses_, kMaxNumberOfUses);
    ++num_uses_;
  }

  inline void detach() {
    DCHECK_GT(num_uses_, 0);
    --num_uses_;
  }

  // Member variables.
  // Note that alignment is important here, the structure is packed so that
  // there is no wasted space.

  // 32 bits.
  const TensorOpcode opcode_;  // Code for the operation performed.  8 bits.
  const int8_t num_inputs_;    // Number of inputs to this node.     8 bits.
  const int8_t num_outputs_;   // Number of outputs from this node.  8 bits.
  DeviceID device_;            // Device for evaluation.             8 bits.

  // 32 bits.
  uint32_t flags_;     // Boolean options.

  // 64 bits.
  int32_t num_uses_;   // Reference count number of uses of the node.
  int32_t id_;         // Index of this node in the graph.

  // Pointer to an array of input nodes.  Owned by Graph.
  TensorBase* input_nodes_;

  // Pointer to an array of output types.  Owned by Graph.
  const TensorType* output_types_;

  // Device pointer to the output data for this tensor.  Owned by Graph.
  void* result_data_;
};


// The purpose of this class is to provide a single override of InvokeKernel
// for all nodes in backend client headers. See also eigen_evaluator_client.h.
// InvokeKernel will statically have access to the runtime type of "this".
// This pattern is known as "Curiously recurring template pattern (CRTP)" or
// "F-bound polymorphism".
template<class Self>
class TensorNodeBaseSelf : public TensorNodeBase {
 public:
  // This method is defined by the backend.
  void InvokeKernel(Graph* graph) override;

 protected:
  TensorNodeBaseSelf(TensorOpcode op, int num_inputs, int num_outputs,
                     TensorBase* inputs, const TensorType* output_types,
                     DeviceID device)
      : TensorNodeBase(op, num_inputs, num_outputs, inputs,
                       output_types, device) {}

  // If no device is specified: Use same device as inputs.
  inline TensorNodeBaseSelf(TensorOpcode op, int num_inputs,
                            int num_outputs, TensorBase* inputs,
                            const TensorType* output_types);
};


// Template base class for AST nodes that return a single tensor.
// DT is the type of data in the tensor.
template<typename DT>
class TensorNode : public TensorNodeBase {
 public:
  typedef DT DataType;

  TensorNode() = delete;
  TensorNode(TensorOpcode op, int num_inputs, TensorBase* inputs,
             const TensorType* output_type, DeviceID device)
      : TensorNodeBase(op, num_inputs, /*num_outputs=*/1, inputs,
        output_type, device) {
    DCHECK(output_type->dtype() == CppTypeToDType<DT>::dtype);
  }

  // Output type of the Tensor.
  const TensorType& tensor_type() const { return *output_types_; }

  // Rank of the tensor. Override with static constant.
  // This returns the same value as Dimensions::rank(), but allows better
  // compiler optimization.
  int rank() const { return dimensions().rank(); }

  // Data type of elements in this Tensor. (Override with static constant.)
  TensorDataType dtype() const { return CppTypeToDType<DT>::dtype; }

  // Return the dimensions of this Tensor.
  const Dimensions& dimensions() const { return tensor_type().dimensions(); }

  // Return the i^th dimension.
  int64 dimension(int i) const { return dimensions()[i]; }

  // Get a device pointer to the underlying tensor data.
  const DT* result_data() const {
    DCHECK(has_result());
    return reinterpret_cast<const DT*>(TensorNodeBase::result_data());
  }
  DT* result_data() {
    DCHECK(has_result());
    return reinterpret_cast<DT*>(TensorNodeBase::result_data());
  }

  void set_result_data(DT* data) { TensorNodeBase::set_result_data(data); }

  // Wrap this node in a Tensor. May only be called if the node is already part
  // of the graph.
  Tensor<DT> as_tensor() { return Tensor<DT>(this); }

  // Return a debug string describing the type of this node.
  string type_str() const override {
    std::ostringstream outstr;
    outstr << TensorNodeBase::type_str() << "<";
    outstr << dtype_name(CppTypeToDType<DT>::dtype) << ">";
    return outstr.str();
  }
};


// The purpose of this class is to provide a single override of InvokeKernel
// for all nodes in backend client headers. See also eigen_evaluator_client.h.
template<typename DT, class Self>
class TensorNodeSelf : public TensorNode<DT> {
 public:
  // This method is defined by the backend.
  void InvokeKernel(Graph* graph) override;

 protected:
  TensorNodeSelf(TensorOpcode op, int num_inputs, TensorBase* inputs,
                 const TensorType* output_type, DeviceID device)
      : TensorNode<DT>(op, num_inputs, inputs, output_type, device) {}

  // If no device is specified: Use same device as input.
  inline TensorNodeSelf(TensorOpcode op, int num_inputs, TensorBase* inputs,
                        const TensorType* output_type);
};


// Collects a set of inputs together into a single node with multiple outputs.
// Used internally to hold gradients of nodes with multiple outputs on the
// backpropagation pass.
class Tuple : public TensorNodeBase {
 public:
  // Technically, the output_types of a tuple should be a concatenation of
  // its input types.  However, we avoid allocating space for the output types,
  // and all the associated GetOutput nodes, by pretending that a tuple has
  // zero outputs.  This is possible because tuples are not really first class
  // citizens.  The only valid operation on a tuple is GetOutput, and that
  // short-circuits to the tuple's input.
  Tuple(int num_inputs, TensorBase* inputs, DeviceID device)
      : TensorNodeBase(kOpTuple, num_inputs, /*num_outputs=*/ 0, inputs,
                       /*output_types=*/nullptr, device) {
    set_allocates_result_data(false);
    set_multiple_outputs(true);  // Zero counts as "multiple".
  }

  void InvokeKernel(Graph* graph) override;
  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Takes a node with multiple outputs, and selects one of them.
// GetOutput nodes are automatically created whenever a node with multiple
// outputs is added to the Graph.
class GetOutput : public TensorNodeBaseSelf<GetOutput> {
 public:
  // Get an output from a node that's not a tuple.
  inline GetOutput(TensorBase* input, int idx, size_t offset);

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  int output_index_;
  size_t offset_;
};


// A tensor value: a tensor allocated in memory.
// TensorValues do not take ownership of the memory.
template<typename DT>
class TensorValue : public TensorNodeSelf<DT, TensorValue<DT>> {
 public:
  TensorValue(TensorType* output_type, DT* data, DeviceID device)
      : TensorNodeSelf<DT, TensorValue<DT>>(kOpValue,
                                            /*num_inputs=*/0,
                                            /*inputs=*/nullptr,
                                            output_type, device),
        data_ptr_(data) {
    // Decide if TensorValue should allocate memory for this tensor.
    if (device != kDeviceIDCPU) {
      // Must copy input data to device.
      this->set_allocates_result_data(/*value=*/ true);
    } else if (!TensorType::is_aligned(reinterpret_cast<uintptr_t>(data))) {
      // Data not aligned, make copy.
      LOG(INFO) << "Tensor data should be 16-byte aligned for efficiency.";
      this->set_allocates_result_data(/*value=*/ true);
    } else {
      // TensorValue will reuse the original memory (data).
      this->set_allocates_result_data(/*value=*/ false);
    }
  }

  // Constant value has a gradient of zero.
  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  DT* data_ptr_;
};


// A tensor that refers to a variable.
template<typename DT>
class TensorVariable : public TensorNodeSelf<DT, TensorVariable<DT>> {
 public:
  // TODO(matthiasspringer): Implement device support.
  TensorVariable(Variable<DT>* variable, DeviceID device)
      : TensorNodeSelf<DT, TensorVariable<DT>>(kOpVariable, /*num_inputs=*/0,
                                               /*inputs=*/nullptr,
                                               &variable->tensor_type(),
                                               device),
        variable_(variable) {
    this->set_allocates_result_data(false);
    this->set_differentiable(true);
  }

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  Variable<DT>* variable() const { return variable_; }

  Variable<DT>* variable_;
};


template<typename DT>
class CopyToDevice
    : public TensorNodeSelf<DT, CopyToDevice<DT>> {
 public:
  explicit CopyToDevice(TensorBase* inputs, const TensorType* output_type,
                        DeviceID device)
      : TensorNodeSelf<DT, CopyToDevice<DT>>(
          kOpCopyToDevice, /*num_inputs=*/ 1, inputs, output_type, device) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


}  // end namespace nodes


// Base class for references to tensors that is not templatized by
// dtype. Performs reference counting to track uses.
class TensorBase {
 public:
  TensorBase() : tensor_node_(nullptr) {}
  TensorBase(const TensorBase& b) : tensor_node_(b.tensor_node_) {
    if (tensor_node_) tensor_node_->attach();
  }
  TensorBase(TensorBase&& b) : tensor_node_(b.tensor_node_) {
    b.tensor_node_ = nullptr;
  }
  ~TensorBase() {
    if (tensor_node_) tensor_node_->detach();
  }

  TensorBase& operator=(const TensorBase& b) {
    reset(b.tensor_node_);
    return *this;
  }
  TensorBase& operator=(TensorBase&& b) {
    if (tensor_node_) tensor_node_->detach();
    tensor_node_ = b.tensor_node_;
    b.tensor_node_ = nullptr;
    return *this;
  }

  bool is_null() const { return tensor_node_ == nullptr; }

  int num_inputs() const { return tensor_node_->num_inputs(); }
  int num_outputs() const { return tensor_node_->num_outputs(); }

  const TensorType& output_type(int i) const {
    return tensor_node_->output_type(i);
  }

  bool has_multiple_outputs() const {
    return tensor_node_->has_multiple_outputs();
  }

  bool is_tensor() const { return tensor_node_->is_tensor(); }

  bool is_tuple() const { return tensor_node_->is_tuple(); }

  TensorBase& sub_expression(int i) {
    return tensor_node_->sub_expression(i);
  }
  const TensorBase& sub_expression(int i) const {
    return tensor_node_->sub_expression(i);
  }

  // Release this reference early, before the destructor is called.
  void release() {
    if (tensor_node_) tensor_node_->detach();
    tensor_node_ = nullptr;
  }

  template<class DT>
  Tensor<DT>& as() {
    DCHECK_EQ(tensor_node_->num_outputs_, 1);
    DCHECK(tensor_node_->output_types_->dtype() == CppTypeToDType<DT>::dtype);
    return *reinterpret_cast<Tensor<DT>*>(this);
  }

  template<class DT>
  const Tensor<DT>& as() const {
    DCHECK_EQ(tensor_node_->num_outputs_, 1);
    DCHECK(tensor_node_->output_types_->dtype() == CppTypeToDType<DT>::dtype);
    return *reinterpret_cast<const Tensor<DT>*>(this);
  }

  template<class DT>
  DT* result_data_as() {
    DCHECK_EQ(tensor_node_->num_outputs_, 1);
    DCHECK(tensor_node_->output_types_->dtype() == CppTypeToDType<DT>::dtype);
    return reinterpret_cast<DT*>(tensor_node_->result_data());
  }

  template<class DT>
  const DT* result_data_as() const {
    DCHECK_EQ(tensor_node_->num_outputs_, 1);
    DCHECK(tensor_node_->output_types_->dtype() == CppTypeToDType<DT>::dtype);
    return reinterpret_cast<const DT*>(tensor_node_->result_data());
  }

  // For internal use only.  Made public for use by gradient and kernel code.
  nodes::TensorNodeBase* get() { return tensor_node_; }
  const nodes::TensorNodeBase* get() const { return tensor_node_; }

  DeviceID device() const { return tensor_node_->device(); }

 protected:
  friend class Graph;

  // Only a Graph is allowed to get or set the underlying AST nodes.
  explicit TensorBase(nodes::TensorNodeBase* t) : tensor_node_(t) {
    if (t) t->attach();
  }

  void reset(nodes::TensorNodeBase* node) {
    if (node) node->attach();
    if (tensor_node_) tensor_node_->detach();
    tensor_node_ = node;
  }

  nodes::TensorNodeBase* tensor_node_;
};


// External reference to a tensor of the given dtype.
// Client code should always use Tensor, rather than TensorNode, so that
// references are counted properly.
template<typename DT>
class Tensor : public TensorBase {
 public:
  typedef DT DataType;

  Tensor() = default;
  Tensor(const Tensor<DT>& t) = default;
  Tensor(Tensor<DT>&& t) = default;
  ~Tensor() {}

  Tensor<DT>& operator=(const Tensor<DT>& t) = default;
  Tensor<DT>& operator=(Tensor<DT>&& t) = default;

  const TensorType& tensor_type() const { return get()->tensor_type(); }

  int rank() const { return get()->rank(); }

  const Dimensions& dimensions() const { return get()->dimensions(); }

  TensorDataType dtype() const { return get()->dtype(); }

  // Return the i^th dimension.
  int64 dimension(int i) const { return dimensions()[i]; }

  DT* result_data() { return get()->result_data(); }
  const DT* result_data() const { return get()->result_data(); }

    // For internal use only.  Made public for use by gradient and kernel code.
  nodes::TensorNode<DT>* get() {
    return reinterpret_cast<nodes::TensorNode<DT>*>(tensor_node_);
  }

  const nodes::TensorNode<DT>* get() const {
    return reinterpret_cast<const nodes::TensorNode<DT>*>(tensor_node_);
  }

 private:
  friend class Graph;
  friend class GraphImplementation;

  // Make sure that TensorNode<DT> can call the private constructor.
  friend class nodes::TensorNode<DT>;

  // Only a Graph can construct Tensors.
  explicit Tensor(nodes::TensorNode<DT>* t) : TensorBase(t) {}
};


namespace nodes {

// Definition here requires complete type for TensorBase.
TensorBase& TensorNodeBase::sub_expression(int i) {
  DCHECK_LT(i, num_inputs_);
  return input_nodes_[i];
}

const TensorBase& TensorNodeBase::sub_expression(int i) const {
  DCHECK_LT(i, num_inputs_);
  return input_nodes_[i];
}

template<typename DT, typename Self>
TensorNodeSelf<DT, Self>::TensorNodeSelf(
    TensorOpcode op, int num_inputs, TensorBase* inputs,
    const TensorType* output_type)
        : TensorNodeSelf<DT, Self>(op, num_inputs, inputs, output_type,
                                   inputs[0].device()) {}

template<typename Self>
TensorNodeBaseSelf<Self>::TensorNodeBaseSelf(
    TensorOpcode op, int num_inputs, int num_outputs, TensorBase* inputs,
    const TensorType* output_types)
        : TensorNodeBaseSelf<Self>(op, num_inputs, num_outputs, inputs,
                                   output_types, inputs[0].device()) {}

// Safe for use with partially constructed nodes.
inline bool maybe_node_is_differentiable(const TensorNodeBase* b) {
  if (b == nullptr) {
    return true;
  } else {
    return b->is_differentiable();
  }
}


// Constructor.
TensorNodeBase::TensorNodeBase(TensorOpcode op, int num_inputs,
                               int num_outputs, TensorBase* inputs,
                               const TensorType* output_types,
                               DeviceID device)
    : opcode_(op), num_inputs_(num_inputs), num_outputs_(num_outputs),
      device_(device), flags_(kAllocatesResultData), num_uses_(0), id_(-1),
      input_nodes_(inputs), output_types_(output_types), result_data_(nullptr)
{
  // A node is differentiable if any of its subexpressions are.
  bool is_diff = std::any_of(input_nodes_, input_nodes_ + num_inputs_,
      [](const TensorBase& subexpr) {
        return maybe_node_is_differentiable(subexpr.get());
      });
  // TODO(delesley): support non-float32 gradients.
  // A node is differentiable only if it has outputs of type float32.
  bool has_float_out = std::any_of(output_types_, output_types + num_outputs_,
      [](const TensorType& ttype) {
        return ttype.dtype() == kDTfloat32;
      });
  set_differentiable(is_diff && has_float_out);
}


inline GetOutput::GetOutput(TensorBase* input, int idx, size_t offset)
    : TensorNodeBaseSelf<GetOutput>(kOpGetOutput, /*num_inputs=*/ 1,
                                    /*num_outputs=*/ 1, input,
                                    &input->output_type(idx)),
      output_index_(idx), offset_(offset) {
  DCHECK(input->get()->is_valid_output_index(idx));
  set_allocates_result_data(false);
}

}  // namespace nodes
}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_TENSOR_H_
