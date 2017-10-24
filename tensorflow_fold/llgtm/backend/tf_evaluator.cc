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

#include "tensorflow_fold/llgtm/backend/tf_evaluator.h"
#include "tensorflow_fold/llgtm/backend/eigen_evaluator.h"
#include "tensorflow_fold/llgtm/backend/eigen_graph_implementation.h"

#include "absl/memory/memory.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/runtime.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/version.h"

namespace llgtm {

using tensorflow::Status;
using tensorflow::gtl::ArraySlice;

// Class used to allocate TensorFlow tensors that reference LLGTM memory.
class NonOwningTensorBuffer : public tensorflow::TensorBuffer {
 public:
  NonOwningTensorBuffer(void* data, size_t size)
    : data_(data), size_(size)  {}
  ~NonOwningTensorBuffer() override {}

  void* data() const override { return data_; }
  size_t size() const override { return size_; }

  TensorBuffer* root_buffer() override { return this; }

  // Whether this TensorBuffer owns the underlying memory.
  bool OwnsMemory() const override { return false; }

  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override {
    proto->set_requested_bytes(size_);
    proto->set_allocator_name("llgtm");
    proto->set_allocated_bytes(0);
    proto->set_ptr(0);
  }

 private:
  void* data_;
  size_t size_;
};


// TODO(delesley): Consider using the tensorflow enum everywhere.
tensorflow::DataType GetTfDataType(TensorDataType dtype) {
  switch (dtype) {
    case kDTvoid: return tensorflow::DT_INVALID;
    case kDTbool: return tensorflow::DT_BOOL;
    case kDTint8: return tensorflow::DT_INT8;
    case kDTint16: return tensorflow::DT_INT16;
    case kDTint32: return tensorflow::DT_INT32;
    case kDTint64: return tensorflow::DT_INT64;
    case kDTfloat32: return tensorflow::DT_FLOAT;
    case kDTfloat64: return tensorflow::DT_DOUBLE;
    case kMaximumTensorDataType: return tensorflow::DT_INVALID;
  }
}


// Map existing data to a TF tensor. The resulting tensor does not own the
// memory.
tensorflow::Tensor make_tf_tensor(TensorDataType dtype, void* data,
                                  const Dimensions& dims) {
  tensorflow::Tensor result;

  // Create a new tensorflow tensor from the result.
  size_t size = sizeof_dtype(dtype)*dims.num_elements();
  auto* buf = new NonOwningTensorBuffer(data, size);
  tensorflow::TensorShape shape(ArraySlice<int64>(dims.as_ptr(),
                                                  dims.rank()));
  result = tensorflow::TensorCApi::MakeTensor(
      static_cast<TF_DataType>(GetTfDataType(dtype)), shape, buf);

  // The buffer is out of scope now and belongs to the TF tensor. Decrease
  // reference count. The memory referenced by the buffer is still managed
  // by LLGTM.
  buf->Unref();

  return result;
}

template<typename DT>
tensorflow::Tensor make_tf_tensor(DT* data, const Dimensions& dims) {
  return make_tf_tensor(CppTypeToDType<DT>::dtype, data, dims);
}

// Map existing data to a TF tensor of rank 1.
template<typename DT>
tensorflow::Tensor make_tf_tensor(DT* data, size_t num_elements) {
  return make_tf_tensor(
      data, Dimensions(static_cast<Dimensions::IndexType>(num_elements)));
}


// Extends GraphImplementation to manage the TensorFlow state associated with
// a Graph.  Constructed by TfGraphEvaluator.
class TfGraphImplementation : public EigenGraphImplementation {
 public:
  TfGraphImplementation(Eigen::DefaultDevice* default_device,
                        Eigen::GpuDevice* gpu_device)
      : EigenGraphImplementation(default_device, gpu_device) {}

  // Evaluate all nodes in the given graph.
  int Eval(Graph* graph) override {
    result_tensors_.resize(graph->size());
    return GraphImplementation::Eval(graph);
  }

 private:
  friend class TfGraphEvaluator;
  friend class TfKernelAdapter;

  std::vector<tensorflow::Tensor> result_tensors_;
};

// Entry point to TensorFlow kernels. Instances of this class represent
// specific TF kernels. They are created with builder notation and can be
// invoked via `InvokeKernel`.
class TfKernelAdapter {
 public:
  // Create a kernel for the operation named op_name on the given device.
  TfKernelAdapter& Create(const char* op_name) {
    node_def_.set_name(op_name);
    node_def_.set_op(op_name);
    return *this;
  }

  // Set an attribute of the kernel.
  // Copied from tensorflow::NodeDefBuilder::Attr.
  template <typename T>
  TfKernelAdapter& Attr(const char* attr_name, T&& value) {
    const tensorflow::AttrValue* found =
      tensorflow::AttrSlice(node_def_).Find(attr_name);
    if (found == nullptr) {
      tensorflow::AddNodeAttr(attr_name, std::forward<T>(value), &node_def_);
    } else {
      // TODO(matthiasspringer): Implement consistency checking as in NodeDefBuilder.
      LOG(INFO) << "TF kernel attribute " << attr_name << " already set.";
    }
    return *this;
  }

  // Set the data type of the kernel.
  template <typename DT>
  TfKernelAdapter& TypeAttr(const char* attr_name) {
    // Tensorflow stores the data type as a property named "T".
    return Attr(attr_name, tensorflow::DataTypeToEnum<DT>::v());
  }

  // Add num_inputs dummy inputs to the kernel.
  TfKernelAdapter& NumInputs(int num_inputs) {
    for (int i = 0; i < num_inputs; ++i) {
      node_def_.add_input("dummy_input");
    }
    return *this;
  }

  void Init(tensorflow::Device* device) {
    Status s = tensorflow::KernelAndDevice::InitOp(device,
                                                   node_def_, &kernel_);
    if (!s.ok()) {
      LOG(ERROR) << "Error initializing kernel. " << s;
    }
  }

  // Invokes a TF kernel for a given node.
  void InvokeKernel(nodes::TensorNodeBase* node, Graph* graph) {
    InvokeKernelWithExtraInputs(node, graph, {});
  }

  // Invokes a TF kernel for a given node with extra inputs.
  void InvokeKernelWithExtraInputs(nodes::TensorNodeBase* node, Graph* graph,
      const std::initializer_list<tensorflow::Tensor>& extra_input) {
    auto* graph_impl = reinterpret_cast<TfGraphImplementation*>(
        graph->graph_implementation());
    auto& result_tensors = graph_impl->result_tensors_;

    int arity = node->num_inputs();
    int extra_input_size = extra_input.size();
    TensorBase* input_nodes = node->sub_expressions();

    std::vector<tensorflow::Tensor> inputs(arity + extra_input_size);
    for (int i = 0; i < arity; ++i) {
      inputs[i] = result_tensors[input_nodes[i].get()->id()];
    }
    for (const auto& input : extra_input) {
      inputs[arity++] = input;
    }

    std::vector<tensorflow::Tensor> outputs;
    // Run the tensorflow kernel, which will allocate its own tensors.
    Status status = kernel_.Run(&inputs, &outputs);
    if (!status.ok()) {
      LOG(ERROR) << "Error invoking kernel. " << status;
    }

    // TODO(delesley): support operations with more than one output.
    tensorflow::Tensor* output_tensor = &result_tensors[node->id()];
    *output_tensor = outputs[0];

    // Get the data from tensorflow.
    auto* buf = tensorflow::TensorCApi::Buffer(result_tensors[node->id()]);

    for (int out_i = 0; out_i < node->num_outputs(); ++out_i) {
      nodes::TensorNodeBase* out_node = graph->node(node->output_id(out_i));
      out_node->set_result_data(buf->data());
    }
  }

 private:
  friend class TfGraphEvaluator;

  tensorflow::NodeDef node_def_;

  tensorflow::KernelAndDevice kernel_ {nullptr};
};

// Constructor and destructor are defined here where we have complete types
// for the unique_ptrs.
TfGraphEvaluator::~TfGraphEvaluator() {}

TfGraphEvaluator::TfGraphEvaluator(DeviceID device, uint64 seed)
    : EigenEvaluator(device, seed) {
  Init();
}

TfGraphEvaluator::TfGraphEvaluator(DeviceID device) : EigenEvaluator(device) {
  Init();
}


TfGraphEvaluator::TfGraphEvaluator() : EigenEvaluator() {
  Init();
}

// Wraps a result buffer in a TF tensor and stores it with the TF graph
// implementation. This function is used to map the result of an Eigen kernel
// to a TF tensor.
void TfGraphEvaluator::register_tf_tensor(nodes::TensorNodeBase* node,
                                          Graph* graph) {
  auto* graph_impl = reinterpret_cast<TfGraphImplementation*>(
      graph->graph_implementation());
  auto& result_tensors = graph_impl->result_tensors_;

  for (int out_i = 0; out_i < node->num_outputs(); ++out_i) {
    const TensorType& output_type = node->output_type(out_i);
    int output_id = node->output_id(out_i);
    auto* out_node = graph->node(output_id);

    // Create a new tensorflow tensor from the result.
    result_tensors[output_id] = make_tf_tensor(output_type.dtype(),
                                               out_node->result_data(),
                                               output_type.dimensions());
  }
}

GraphImplementation* TfGraphEvaluator::NewGraphImpl() {
  // TODO(matthiasspringer): Remove this once TfGraphEvaluator is standalone. At the
  // moment, TfGraphEvaluator is a subclass of EigenEvaluator and
  // TfGraphImplementation is a subclass of EigenGraphImplementation, such that
  // we can call into Eigen kernels from the TF backend.
  return new TfGraphImplementation(this->default_device(), this->gpu_device());
}

void TfGraphEvaluator::Init() {
  // Code adapted from learning/brain/contrib/eager
  Status status;

  tensorflow::SessionOptions options;
  std::vector<tensorflow::Device*> devices;
  status = tensorflow::DeviceFactory::AddDevices(
      options, /*name_prefix=*/ "/job:llgtm/replica:0/task:0", &devices);
  CHECK(status.ok());

  device_mgr_ = absl::make_unique<tensorflow::DeviceMgr>(devices);
  devices_ = device_mgr_->ListDevices();
  // TODO(delesley): Add support for multiple devices other than cpu.
  tensorflow::Device* cpu_device = devices_[0];

  kernels_.resize(kMaximumTensorOpcode);

  kernels_[kOpAdd] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpAdd]->Create("Add")
                  .TypeAttr<float>("T")
                  .NumInputs(2)
                  .Init(cpu_device);

  kernels_[kOpConstantFromScalar] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpConstantFromScalar]->Create("Fill")
                                 .TypeAttr<float>("T")
                                 .NumInputs(2)
                                 .Init(cpu_device);

  kernels_[kOpMultiply] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpMultiply]->Create("Mul")
                       .TypeAttr<float>("T")
                       .NumInputs(2)
                       .Init(cpu_device);

  kernels_[kOpReciprocal] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpReciprocal]->Create("Reciprocal")
                         .TypeAttr<float>("T")
                         .NumInputs(1)
                         .Init(cpu_device);

  // TODO(matthiasspringer): Utilize transpose_a/b attributes.
  kernels_[kOpMatmul] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpMatmul]->Create("MatMul")
                     .Attr("transpose_a", false)
                     .Attr("transpose_b", false)
                     .TypeAttr<float>("T")
                     .NumInputs(2)
                     .Init(cpu_device);

  kernels_[kOpNegative] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpNegative]->Create("Neg")
                       .TypeAttr<float>("T")
                       .NumInputs(1)
                       .Init(cpu_device);

  kernels_[kOpSigmoid] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpSigmoid]->Create("Sigmoid")
                      .TypeAttr<float>("T")
                      .NumInputs(1)
                      .Init(cpu_device);

  kernels_[kOpTanh] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpTanh]->Create("Tanh")
                   .TypeAttr<float>("T")
                   .NumInputs(1)
                   .Init(cpu_device);

  kernels_[kOpTranspose] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpTranspose]->Create("Transpose")
                        .TypeAttr<float>("T")
                        .TypeAttr<int32_t>("Tperm")
                        .NumInputs(2)
                        .Init(cpu_device);

  kernels_[kOpReduceSum] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpReduceSum]->Create("Sum")
                        .Attr("keep_dims", true)
                        .TypeAttr<float>("T")
                        .TypeAttr<int32_t>("Tidx")
                        .NumInputs(2)
                        .Init(cpu_device);

  kernels_[kOpReshape] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpReshape]->Create("Reshape")
                      .TypeAttr<float>("T")
                      .TypeAttr<int32_t>("Tshape")
                      .NumInputs(2)
                      .Init(cpu_device);

  kernels_[kOpRelu] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpRelu]->Create("Relu")
                   .TypeAttr<float>("T")
                   .NumInputs(1)
                   .Init(cpu_device);

  kernels_[kOpReluGrad] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpReluGrad]->Create("ReluGrad")
                       .TypeAttr<float>("T")
                       .NumInputs(2)
                       .Init(cpu_device);

  kernels_[kOpNormalRandom] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpNormalRandom]->Create("RandomStandardNormal")
                           .Attr("seed", 0)
                           .Attr("seed2", 0)
                           .TypeAttr<int32_t>("T")
                           .TypeAttr<float>("dtype")
                           .NumInputs(1)
                           .Init(cpu_device);

  kernels_[kOpUniformRandom] = absl::make_unique<TfKernelAdapter>();
  kernels_[kOpUniformRandom]->Create("RandomUniform")
                            .Attr("seed", 0)
                            .Attr("seed2", 0)
                            .TypeAttr<int32_t>("T")
                            .TypeAttr<float>("dtype")
                            .NumInputs(1)
                            .Init(cpu_device);

  // TODO(delesley): Handle broadcast.
  // TODO(matthiasspringer): Handle softmax and softmax cross-entropy (+gradient).
  // TODO(matthiasspringer): Handle concat and split.
}

void TfGraphEvaluator::InvokeKernel(nodes::Add<float>* node,
                                    Graph* graph) {
  kernels_[kOpAdd].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::ConstantFromScalar<float>* node,
                                    Graph* graph) {
  // Prepare dimensions (int32_t tensor) and constant value.
  auto dimensions alignas(TensorType::kResultAlignment) =
      node->dimensions().as_array<int32_t>();
  float value alignas(TensorType::kResultAlignment) = node->value_;

  kernels_[kOpConstantFromScalar].get()->InvokeKernelWithExtraInputs(
      node, graph, { make_tf_tensor(dimensions.data(), node->rank()),
                     make_tf_tensor(&value, Dimensions()) });
}

void TfGraphEvaluator::InvokeKernel(nodes::Matmul<float>* node,
                                    Graph* graph) {
  kernels_[kOpMatmul].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::Multiply<float>* node,
                                    Graph* graph) {
  kernels_[kOpMultiply].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::Negative<float>* node,
                                    Graph* graph) {
  kernels_[kOpNegative].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::Reciprocal<float>* node,
                                    Graph* graph) {
  kernels_[kOpReciprocal].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::ReduceSum<float>* node,
                                    Graph* graph) {
  auto sa = node->sub_expression(0).template as<float>();

  // Calculate reduce dimensions.
  std::array<int32_t, Dimensions::kMaxRank> reduce_indices
      alignas(TensorType::kResultAlignment);
  int index = 0;
  for (int i = 0; i < node->rank(); ++i) {
    if (node->dimension(i) < sa.dimension(i)) {
      reduce_indices[index++] = i;
    }
  }
  DCHECK_EQ(index, node->num_reductions());

  kernels_[kOpReduceSum].get()->InvokeKernelWithExtraInputs(
      node, graph, { make_tf_tensor(reduce_indices.data(), index) });
}

void TfGraphEvaluator::InvokeKernel(nodes::Reshape<float>* node,
                                    Graph* graph) {
  // Prepare indices (int32_t tensor).
  auto dims alignas(TensorType::kResultAlignment) =
      node->dimensions().as_array<int32_t>();

  kernels_[kOpReshape].get()->InvokeKernelWithExtraInputs(
      node, graph, { make_tf_tensor(dims.data(), node->rank()) });
}

void TfGraphEvaluator::InvokeKernel(nodes::Relu<float>* node,
                                    Graph* graph) {
  kernels_[kOpRelu].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::ReluGrad<float>* node,
                                    Graph* graph) {
  kernels_[kOpReluGrad].get()->InvokeKernel(node, graph);
}

// Helper method that can invoke either UniformRandom or NormalRandom.
template<typename NodeType>
void TfGraphEvaluator::InvokeRandomKernel(NodeType* node, Graph* graph,
                                          TfKernelAdapter* kernel) {
  if (node->is_custom_seed_) {
    // TF kernel cannot handle custom seed. Use Eigen kernel.
    LaunchEigenKernel<NodeType>(node, graph);
    register_tf_tensor(node, graph);
  } else {
    auto dimensions alignas(TensorType::kResultAlignment) =
        node->dimensions().template as_array<int32_t>();
    kernel->InvokeKernelWithExtraInputs(node, graph,
        { make_tf_tensor(dimensions.data(), node->rank()) });
  }
}

void TfGraphEvaluator::InvokeKernel(nodes::NormalRandom<float>* node,
                                    Graph* graph) {
  InvokeRandomKernel<nodes::NormalRandom<float>>(
      node, graph, kernels_[kOpNormalRandom].get());
}

void TfGraphEvaluator::InvokeKernel(nodes::UniformRandom<float>* node,
                                    Graph* graph) {
  InvokeRandomKernel<nodes::UniformRandom<float>>(
      node, graph, kernels_[kOpUniformRandom].get());
}

void TfGraphEvaluator::InvokeKernel(nodes::Tanh<float>* node,
                                    Graph* graph) {
  kernels_[kOpTanh].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::Sigmoid<float>* node,
                                    Graph* graph) {
  kernels_[kOpSigmoid].get()->InvokeKernel(node, graph);
}

void TfGraphEvaluator::InvokeKernel(nodes::Transpose<float>* node,
                                    Graph* graph) {
  // Prepare transpose indices (int32_t tensor).
  auto transpose_indices alignas(TensorType::kResultAlignment) =
      node->indices_.as_array<int32_t>();

  kernels_[kOpTranspose].get()->InvokeKernelWithExtraInputs(
      node, graph, { make_tf_tensor(transpose_indices.data(),
                                    node->indices_.rank()) });
}

void TfGraphEvaluator::InvokeKernel(nodes::Zeros<float>* node,
                                    Graph* graph) {
  // Prepare dimensions (int32_t tensor) and constant value.
  auto dimensions alignas(TensorType::kResultAlignment) =
      node->dimensions().as_array<int32_t>();
  float value alignas(TensorType::kResultAlignment) = 0.0f;

  kernels_[kOpConstantFromScalar].get()->InvokeKernelWithExtraInputs(
      node, graph, { make_tf_tensor(dimensions.data(), node->rank()),
                     make_tf_tensor(&value, Dimensions()) });
}

}  // namespace llgtm
