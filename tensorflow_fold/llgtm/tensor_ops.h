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

// This file defines subclasses of TensorNode that implement various tensor
// operations.

#ifndef TENSORFLOW_FOLD_LLGTM_TENSOR_OPS_H_
#define TENSORFLOW_FOLD_LLGTM_TENSOR_OPS_H_

#include "tensorflow_fold/llgtm/gradients.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/util.h"
#include "absl/types/span.h"

namespace llgtm {
namespace nodes {  // Internal namespace for node definitions.


// Initialize a constant tensor to all zeros.
template<typename DT>
class Zeros : public TensorNodeSelf<DT, Zeros<DT>> {
 public:
  explicit Zeros(const TensorType* output_type, DeviceID device)
      : TensorNodeSelf<DT, Zeros<DT>>(kOpZeros, /*num_inputs=*/ 0,
                                      /*inputs=*/ nullptr,
                                      output_type, device) {}

  void ComputeGradients(TensorBase error,
                        Gradients* gradients) override {}
};


// Initialize a constant tensor randomly (uniform).
template<typename DT>
class UniformRandom : public TensorNodeSelf<DT, UniformRandom<DT>> {
 public:
  explicit UniformRandom(const TensorType* output_type, uint64 seed,
                         bool is_custom_seed, DeviceID device)
      : TensorNodeSelf<DT, UniformRandom<DT>>(kOpUniformRandom,
                                              /*num_inputs=*/ 0,
                                              /*inputs=*/ nullptr,
                                              output_type, device),
        seed_(seed), is_custom_seed_(is_custom_seed) {}

  void ComputeGradients(TensorBase /*error*/,
                        Gradients* /*gradients*/) override {}

  uint64 seed_;

  bool is_custom_seed_;
};


// Initialize a constant tensor randomly (normal).
template<typename DT>
class NormalRandom : public TensorNodeSelf<DT, NormalRandom<DT>> {
 public:
  explicit NormalRandom(const TensorType* output_type, uint64 seed,
                        bool is_custom_seed, DeviceID device)
      : TensorNodeSelf<DT, NormalRandom<DT>>(kOpNormalRandom, /*num_inputs=*/ 0,
                                             /*inputs=*/ nullptr, output_type,
                                             device),
        seed_(seed), is_custom_seed_(is_custom_seed) {}

  void ComputeGradients(TensorBase /*error*/,
                        Gradients* /*gradients*/) override {}

  uint64 seed_;

  bool is_custom_seed_;
};


// Initialize a constant tensor using the given function.
template<typename DT, int Rank, typename F>
class ConstantFromFunction
    : public TensorNodeSelf<DT, ConstantFromFunction<DT, Rank, F>> {
 public:
  ConstantFromFunction(const TensorType* output_type, F f, DeviceID device)
      : TensorNodeSelf<DT, ConstantFromFunction<DT, Rank, F>>(
          kOpConstantFromFunction, 0, /*inputs=*/ nullptr,
          output_type, device),
        init_function_(f) {
    CHECK_EQ(output_type->dimensions().rank(), Rank);
  }

  void ComputeGradients(TensorBase error,
                        Gradients* gradients) override {}

  F init_function_;
};


// Initialize a constant tensor using a scalar.
template<typename DT>
class ConstantFromScalar : public TensorNodeSelf<DT, ConstantFromScalar<DT>> {
 public:
  ConstantFromScalar(const TensorType* output_type, DT value, DeviceID device)
      : TensorNodeSelf<DT, ConstantFromScalar<DT>>(kOpConstantFromScalar,
                                                   /*num_inputs=*/ 0,
                                                   /*inputs=*/ nullptr,
                                                   output_type, device),
        value_(value) {}

  void ComputeGradients(TensorBase error,
                        Gradients* gradients) override {}

  DT value_;
};


// Element-wise negation.
template<typename DT>
class Negative : public TensorNodeSelf<DT, Negative<DT>> {
 public:
  explicit Negative(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Negative<DT>>(kOpNegative, /*num_inputs=*/ 1,
                                         inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Element-wise addition of two tensors.
template<typename DT>
class Add : public TensorNodeSelf<DT, Add<DT>> {
 public:
  Add(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Add<DT>>(kOpAdd, /*num_inputs=*/ 2, inputs,
                                    output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Adds a gradient to a variable.  (Implements operator+=() on variables.)
template<typename DT>
class AssignAdd : public TensorNodeBaseSelf<AssignAdd<DT>> {
 public:
  // TODO(matthiasspringer): Implement device support.
  AssignAdd(TensorBase* input, Variable<DT>* var)
      : TensorNodeBaseSelf<AssignAdd<DT>>(kOpAssignAdd, 1, /*num_outputs=*/0,
                                          input, /*output_types=*/nullptr),
        variable_(var) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override {}

  Variable<DT>* variable() { return variable_; }

 private:
  Variable<DT>* variable_;
};


// Element-wise inversion of a tensor.
template<typename DT>
class Reciprocal : public TensorNodeSelf<DT, Reciprocal<DT>> {
 public:
  explicit Reciprocal(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Reciprocal<DT>>(kOpReciprocal, /*num_inputs=*/ 1,
                                           inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Element-wise multiplication of two tensors.
template<typename DT>
class Multiply : public TensorNodeSelf<DT, Multiply<DT>> {
 public:
  Multiply(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Multiply<DT>>(kOpMultiply, /*num_inputs=*/ 2,
                                         inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Element-wise sigmoid of a tensor.
template<typename DT>
class Sigmoid : public TensorNodeSelf<DT, Sigmoid<DT>> {
 public:
  explicit Sigmoid(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Sigmoid<DT>>(kOpSigmoid, /*num_inputs=*/ 1,
                                        inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Element-wise hyperbolic tangent of a tensor.
template<typename DT>
class Tanh : public TensorNodeSelf<DT, Tanh<DT>> {
 public:
  explicit Tanh(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Tanh<DT>>(kOpTanh, /*num_inputs=*/ 1,
                                     inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Element-wise Relu of a tensor.
template<typename DT>
class Relu : public TensorNodeSelf<DT, Relu<DT>> {
 public:
  explicit Relu(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Relu<DT>>(kOpRelu, /*num_inputs=*/ 1,
                                     inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Create a tensor with the same number of elements but different dimensions.
template<typename DT>
class Reshape : public TensorNodeSelf<DT, Reshape<DT>> {
 public:
  Reshape(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Reshape<DT>>(kOpReshape, /*num_inputs=*/ 1,
                                        inputs, output_type) {
    this->set_allocates_result_data(false);
  }

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Element-wise first derivative of Relu of a tensor.
template<typename DT>
class ReluGrad : public TensorNodeSelf<DT, ReluGrad<DT>> {
 public:
  ReluGrad(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, ReluGrad<DT>>(kOpReluGrad, /*num_inputs=*/ 2,
                                         inputs, output_type) {}

  void ComputeGradients(TensorBase /*error*/,
                        Gradients* /*gradients*/) override {
    LOG(FATAL) << "Gradient of ReluGrad is not implemented.";
  }
};


// Create a larger tensor by copying a smaller tensor.
template<typename DT>
class Broadcast : public TensorNodeSelf<DT, Broadcast<DT>> {
 public:
  Broadcast(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Broadcast<DT>>(kOpBroadcast, /*num_inputs=*/ 1,
                                          inputs, output_type) {}

  static void CheckBroadcastDimensions(const Dimensions& a_dims,
                                       const Dimensions& b_dims);

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Reduce a tensor by summing rows and columns.
// This version keeps the rank of the tensor the same; the reduced dimensions
// are replaced by ones.
template<typename DT>
class ReduceSum : public TensorNodeSelf<DT, ReduceSum<DT>> {
 public:
  // The indices parameter specifies which dimensions to reduce.
  ReduceSum(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, ReduceSum<DT>>(kOpReduceSum, /*num_inputs=*/ 1,
                                          inputs, output_type) {
    // Count number of reductions.
    num_reductions_ = 0;
    const auto& input = this->sub_expression(0).template as<DT>();
    for (int r = 0; r < input.dimensions().rank(); ++r) {
      if (output_type->dimension(r) < input.dimension(r)) {
        ++num_reductions_;
      }
    }
    // ReduceSum is an identity in this case.
    if (num_reductions_ == 0) this->set_allocates_result_data(false);
  }

  // Get the dimensions of a tensor that has been reduced (e.g ReduceAdd)
  static Dimensions ReducedDimensions(const Dimensions& dims,
      const std::initializer_list<int>& indices);

  static void CheckReduceDimensions(const Dimensions& before,
                                    const Dimensions& after);

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  int num_reductions() { return num_reductions_; }

 private:
  int num_reductions_;
};


// Gathers a set of rows from its input.
// The first input is a Tensor<int32_t> which contains the indices, and the
// second input is the tensor to gather from.
// TODO(delesley): extend to multi-gather.
template<typename DT>
class Gather : public TensorNodeSelf<DT, Gather<DT>> {
 public:
  Gather(TensorBase* inputs, const TensorType* output_type, int axis)
      : TensorNodeSelf<DT, Gather<DT>>(kOpGather, /*num_inputs=*/ 2,
                                       inputs, output_type),
        axis_(axis) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  static Dimensions OutputDimensions(const Dimensions& input_dims, int axis,
                                     Dimensions::IndexType num_indices) {
    Dimensions dims = input_dims;
    dims.set_dimension(axis, num_indices);
    return dims;
  }

  int axis_;
};


// Takes a dense tensor as input, and scatters the rows to specified locations
// of the output.  It essentially implements a sparse tensor.
// The first input is a Tensor<int32_t> which contains the indices to scatter,
// and the second contains the data to scatter.
template<typename DT>
class Scatter : public TensorNodeSelf<DT, Scatter<DT>> {
 public:
  Scatter(TensorBase* inputs, const TensorType* output_type, int axis)
      : TensorNodeSelf<DT, Scatter<DT>>(kOpScatter, /*num_inputs=*/ 2,
                                        inputs, output_type),
        axis_(axis) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  int axis_;
};


// Transposition of a tensor.
template<typename DT>
class Transpose : public TensorNodeSelf<DT, Transpose<DT>> {
 public:
  Transpose(TensorBase* inputs, const TensorType* output_type,
            const DimensionIndices& indices)
      : TensorNodeSelf<DT, Transpose<DT>>(kOpTranspose, /*num_inputs=*/ 1,
                                          inputs, output_type),
        indices_(indices) {}

  // Assert that `indices` contains all integers in [0; Rank).
  static void CheckTransposeIndices(const Dimensions& dims,
                                    const DimensionIndices& indices);

  // Returns new Dimensions which are reordered according to idx.
  static Dimensions TransposedDimensions(const Dimensions& dims,
                                         const DimensionIndices& indices);

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  const DimensionIndices indices_;
};


// Softmax of a tensor. The last dimension is being summed over.
template<typename DT>
class Softmax : public TensorNodeSelf<DT, Softmax<DT>> {
 public:
  explicit Softmax(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Softmax<DT>>(kOpSoftmax, /*num_inputs=*/ 1,
                                        inputs, output_type) {}

  void ComputeGradients(TensorBase error,
                        Gradients* gradients) override;
};


// Cross-entropy between a probability distribution and a softmax operation.
// This operation does not compute softmax itself but requires a softmax node
// as input.
template<typename DT>
class SoftmaxCrossEntropy
    : public TensorNodeSelf<DT, SoftmaxCrossEntropy<DT>> {
 public:
  SoftmaxCrossEntropy(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, SoftmaxCrossEntropy<DT>>(kOpSoftmaxCrossEntropy,
                                                    /*num_inputs=*/ 2,
                                                    inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  // Returns new the dimensions of the result of cross-entropy.
  static Dimensions ReducedDimensions(const Dimensions& dims);
};


// Cross-entropy between a probability distribution (represented as indices
// where probability is 1.0) and a softmax operation.
// This operation does not compute softmax itself but requires a softmax node
// as input.
template<typename DT>
class SoftmaxSparseCrossEntropy
    : public TensorNodeSelf<DT, SoftmaxSparseCrossEntropy<DT>> {
 public:
  SoftmaxSparseCrossEntropy(TensorBase* inputs,
                            const TensorType* output_type)
      : TensorNodeSelf<DT, SoftmaxSparseCrossEntropy<DT>>(
          kOpSoftmaxSparseCrossEntropy, /*num_inputs=*/ 2,
          inputs, output_type) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;
};


// Gradient of `SparseCrossEntropy(labels, Softmax(logits))` in terms of
// `logits`. This is *not* the gradient of `SoftmaxSparseCrossEntropy(
// labels, s)` with respect to `s`.
template<typename DT>
class SoftmaxSparseCrossEntropyGrad
    : public TensorNodeSelf<DT, SoftmaxSparseCrossEntropyGrad<DT>> {
 public:
  SoftmaxSparseCrossEntropyGrad(TensorBase* inputs,
                                const TensorType* output_type)
      : TensorNodeSelf<DT, SoftmaxSparseCrossEntropyGrad<DT>>(
          kOpSoftmaxSparseCrossEntropyGrad, /*num_inputs=*/ 2,
          inputs, output_type) {}

  void ComputeGradients(TensorBase /*error*/,
                        Gradients* /*gradients*/) override {
    LOG(FATAL) << "SoftmaxSparseCrossEntropyGrad gradient is not implemented.";
  }
};


// Multiplication of two matrices. Restricted to rank 2 matrices/tensors at
// the moment, but may be extended in the future.
template<typename DT>
class Matmul : public TensorNodeSelf<DT, Matmul<DT>> {
 public:
  Matmul(TensorBase* inputs, const TensorType* output_type)
      : TensorNodeSelf<DT, Matmul<DT>>(kOpMatmul, /*num_inputs=*/ 2,
                                       inputs, output_type) {
    // Dimensions are checked in Dimensions::matmul.
  }

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  static Dimensions ResultDimensions(Dimensions a, Dimensions b);
};


// Concatenation of at least two tensors along a given axis.
template<typename DT>
class Concat : public TensorNodeSelf<DT, Concat<DT>> {
 public:
  Concat(int num_inputs, TensorBase* inputs,
         const TensorType* output_type, int axis)
      : TensorNodeSelf<DT, Concat<DT>>(kOpConcat, num_inputs,
                                       inputs, output_type),
        axis_(axis) {}

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  // Returns new the dimensions of the result of Concat.
  static Dimensions ConcatenatedDimensions(
      const absl::Span<const Tensor<DT>> tensors, int axis);

  const int axis_;
};


// Given a single tensor, outputs multiple tensors split according to given
// sizes along a given axis.
template<typename DT>
class Split : public TensorNodeBaseSelf<Split<DT>> {
 public:
  Split(TensorBase* inputs, TensorType* output_types,
        const absl::Span<const Dimensions::IndexType> sizes, int axis)
      : TensorNodeBaseSelf<Split<DT>>(kOpSplit, /*num_inputs=*/ 1,
                                      sizes.size(), inputs, output_types),
        sizes_(sizes), axis_(axis) {
    this->set_multiple_outputs(true);
  }

  void ComputeGradients(TensorBase error, Gradients* gradients) override;

  static void InitializeOutputTypes(
      Dimensions in_dims, TensorType* output_types,
      const absl::Span<const Dimensions::IndexType> sizes, int axis);

  const absl::Span<const Dimensions::IndexType> sizes_;

  const int axis_;
};


}  // end namespace nodes
}  // end namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_TENSOR_OPS_H_
