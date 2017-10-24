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

// This file provides an implementation for all operations in tensor_ops.
// The initial implementation simply calls the appropriate Eigen function for
// each kernel. The Eigen dependency is localized to this file, so future
// versions may use a different implementation, e.g. Halide or Techila.

#ifndef TENSORFLOW_FOLD_LLGTM_TENSOR_OPS_IMPL_H_
#define TENSORFLOW_FOLD_LLGTM_TENSOR_OPS_IMPL_H_

#include <cmath>
#include "tensorflow_fold/llgtm/graph.h"
#include "tensorflow_fold/llgtm/tensor_ops.h"
#include "tensorflow_fold/llgtm/util.h"
#include "absl/types/span.h"

namespace llgtm {
namespace nodes {


// Variable naming conventions:
// Input nodes ("sub expressions") of operations are named s*, e.g., sa for the
// first input and sb for the second input.


template<typename DT>
void TensorValue<DT>::ComputeGradients(TensorBase error,
                                       Gradients* gradients) {}

template<typename DT>
void TensorVariable<DT>::ComputeGradients(TensorBase error,
                                          Gradients* gradients) {
  gradients->AddGradient(variable_, error);
}

template<typename DT>
void Reshape<DT>::ComputeGradients(TensorBase error,
                                   Gradients* gradients) {
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    Graph* g = gradients->graph();
    auto a_err = g->Reshape<DT>(err, sa.dimensions());
    gradients->PropagateError(sa.get(), a_err);
  }
}

template<typename DT>
void Negative<DT>::ComputeGradients(TensorBase error,
                                    Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    auto a_err = g->Negative(err);
    gradients->PropagateError(sa.get(), a_err);
  }
}

template<typename DT>
void Add<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& sb = this->sub_expression(1).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    gradients->PropagateError(sa.get(), err);
  }
  if (sb.get()->is_differentiable()) {
    gradients->PropagateError(sb.get(), err);
  }
}

template<typename DT>
void Reciprocal<DT>::ComputeGradients(TensorBase error,
                                      Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    auto a_err =
        g->Multiply(err, g->Negative(g->Reciprocal(g->Multiply(sa, sa))));
    gradients->PropagateError(sa.get(), a_err);
  }
}

template<typename DT>
void Multiply<DT>::ComputeGradients(TensorBase error,
                                    Gradients* gradients) {
  Graph* g = gradients->graph();

  auto& sa = this->sub_expression(0).template as<DT>();
  auto& sb = this->sub_expression(1).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    auto a_err = g->Multiply(err, sb);
    gradients->PropagateError(sa.get(), a_err);
  }

  if (sb.get()->is_differentiable()) {
    auto b_err = g->Multiply(err, sa);
    gradients->PropagateError(sb.get(), b_err);
  }
}

template<typename DT>
void Sigmoid<DT>::ComputeGradients(TensorBase error,
                                   Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    // TODO(delesley): Add fused SigmoidError node.
    auto sig_a = this->as_tensor();
    auto a_err = g->Multiply(err,
        g->Multiply(sig_a, g->Subtract(g->Ones<DT>(sa.dimensions()), sig_a)));
    gradients->PropagateError(sa.get(), a_err);
  }
}

template<typename DT>
void Tanh<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    // d tanh(x) / dx = 1 - tanh(x) * tanh(x)
    // TODO(delesley,matthiasspringer): Add fused TanhError node.
    auto tanh_a = this->as_tensor();
    auto a_err = g->Multiply(err, g->Subtract(
        g->ConstantFromScalar<DT>(sa.dimensions(), 1.0f),
        g->Multiply(tanh_a, tanh_a)));
    gradients->PropagateError(sa.get(), a_err);
  }
}

template<typename DT>
void Relu<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    gradients->PropagateError(sa.get(), g->ReluGrad(err, sa));
  }
}

// Check that b_dims is a valid broadcast of a_dims.
template<typename DT>
void Broadcast<DT>::CheckBroadcastDimensions(const Dimensions& a_dims,
                                             const Dimensions& b_dims) {
  CHECK_EQ(a_dims.rank(), b_dims.rank());

  for (int i = 0; i < a_dims.rank(); ++i) {
    DCHECK(a_dims[i] == 1 || a_dims[i] == b_dims[i]);
  }
}

template<typename DT>
void Broadcast<DT>::ComputeGradients(TensorBase error,
                                     Gradients* gradients) {
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    Graph* g = gradients->graph();
    auto a_err = g->ReduceSum<DT>(err, sa.dimensions());
    gradients->PropagateError(sa.get(), a_err);
  }
}

// Get the dimensions of a tensor that has been reduced.
template<typename DT>
Dimensions ReduceSum<DT>::ReducedDimensions(const Dimensions& dims,
    const std::initializer_list<int>& indices) {
  // Reduced dimensions are set to 1 for each dimension that's reduced.
  Dimensions result(dims);

  for (const auto& el : indices) {
    CHECK_LT(el, dims.rank());
    result.set_dimension(el, 1);
  }
  return result;
}

template<typename DT>
void ReduceSum<DT>::CheckReduceDimensions(const Dimensions& before,
                                          const Dimensions& after) {
  CHECK_EQ(before.rank(), after.rank());

  for (int r = 0; r < before.rank(); ++r) {
    CHECK(after[r] == 1 || after[r] == before[r]);
  }
}

template<typename DT>
void ReduceSum<DT>::ComputeGradients(TensorBase error,
                                     Gradients* gradients) {
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    Graph* g = gradients->graph();
    auto a_err = g->Broadcast<DT>(err, sa.dimensions());
    gradients->PropagateError(sa.get(), a_err);
  }
}

// Returns new Dimensions which are reordered according to indices.
template<typename DT>
Dimensions Transpose<DT>::TransposedDimensions(
    const Dimensions& dims, const DimensionIndices& indices) {
  CHECK_EQ(dims.rank(), indices.rank());
  Dimensions result(dims);

  for (int i = 0; i < dims.rank(); ++i) {
    result.set_dimension(i, dims[indices[i]]);
  }
  return result;
}

template<typename DT>
void Gather<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  auto& sb = this->sub_expression(1).template as<DT>();

  if (sb.get()->is_differentiable()) {
    Graph* g = gradients->graph();
    auto& indices = this->sub_expression(0).template as<int32_t>();
    auto& err = error.template as<DT>();
    auto& b_dims = this->sub_expression(1).output_type(0).dimensions();
    auto b_err = g->Scatter(b_dims[this->axis_], indices, err);
    gradients->PropagateError(sb.get(), b_err);
  }
}

template<typename DT>
void Scatter<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  auto& sb = this->sub_expression(1).template as<DT>();

  if (sb.get()->is_differentiable()) {
    Graph* g = gradients->graph();
    auto& indices = this->sub_expression(0).template as<int32_t>();
    auto& err = error.template as<DT>();

    auto b_err = g->Gather(indices, err);
    gradients->PropagateError(sb.get(), b_err);
  }
}

// Check if indices is a valid array of indices for tranpose.
template<typename DT>
void Transpose<DT>::CheckTransposeIndices(const Dimensions& dims,
                                          const DimensionIndices& indices) {
  CHECK_EQ(indices.rank(), dims.rank());
  int size = indices.rank();

  // Check if indices contains all values from 0 to R.
  // Note: This is N^2 but N is small.
  for (int i = 0; i < size; ++i) {
    bool found = false;

    for (int j = 0; j < size; ++j) {
      if (indices[j] == i) {
        found = true;
        break;
      }
    }

    if (!found) {
      LOG(FATAL) << "Invalid transpose indices.";
    }
  }
}

template<typename DT>
void Transpose<DT>::ComputeGradients(TensorBase error,
                                     Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    // Apply inverse transposition
    DimensionIndices inverse_idx(indices_);

    for (int i = 0; i < sa.dimensions().rank(); ++i) {
      inverse_idx.set_dimension(indices_[i], i);
    }

    auto a_err = g->Transpose<DT>(err, inverse_idx);
    DCHECK_EQ(sa.dimensions(), a_err.dimensions());
    gradients->PropagateError(sa.get(), a_err);
  }
}

template<typename DT>
void Softmax<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    auto softmax_a = this->as_tensor();
    auto inner = g->Broadcast(g->ReduceSum(g->Multiply(softmax_a, err),
                                           { sa.dimensions().rank() - 1 }),
                              sa.dimensions());
    gradients->PropagateError(sa.get(), g->Multiply(softmax_a,
                                                   g->Subtract(err, inner)));
  }
}

template<typename DT>
void SoftmaxCrossEntropy<DT>::ComputeGradients(TensorBase error,
                                               Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& labels = this->sub_expression(0).template as<DT>();
  auto& softmax = this->sub_expression(1).template as<DT>();
  auto softmax_node = reinterpret_cast<Softmax<DT>*>(softmax.get());
  auto& logits = softmax_node->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (softmax.get()->is_differentiable()) {
    gradients->PropagateError(
        logits.get(),
        g->Multiply(g->Broadcast(err, softmax.dimensions()),
                    g->Subtract(softmax, labels)));
  }
}

// Return the dimensions of the result of cross-entropy: Set the last dimension
// to 1.
template<typename DT>
Dimensions SoftmaxCrossEntropy<DT>::ReducedDimensions(const Dimensions& dims) {
  Dimensions result(dims);
  result.set_dimension(dims.rank() - 1, 1);
  return result;
}

template<typename DT>
void SoftmaxSparseCrossEntropy<DT>::ComputeGradients(
    TensorBase error, Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& labels = this->sub_expression(0).template as<int>();
  auto& softmax = this->sub_expression(1).template as<DT>();
  auto softmax_node = reinterpret_cast<Softmax<DT>*>(softmax.get());
  auto& logits = softmax_node->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (softmax.get()->is_differentiable()) {
    gradients->PropagateError(logits.get(),
        g->Multiply(g->Broadcast(err, softmax.dimensions()),
                    g->SoftmaxSparseCrossEntropyGrad(labels, softmax)));
  }
}

template<typename DT>
void Matmul<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  Graph* g = gradients->graph();

  auto& sa = this->sub_expression(0).template as<DT>();
  auto& sb = this->sub_expression(1).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    gradients->PropagateError(sa.get(), g->Matmul(err, g->Transpose(sb)));
  }

  if (sb.get()->is_differentiable()) {
    gradients->PropagateError(sb.get(), g->Matmul(g->Transpose(sa), err));
  }
}

// Return the Dimensions of a matrix multiplication. Currently supports only
// matrices (rank 2). May be extended in the future.
template<typename DT>
Dimensions Matmul<DT>::ResultDimensions(Dimensions a, Dimensions b) {
    CHECK_EQ(a.rank(), 2);
    CHECK_EQ(b.rank(), 2);
    CHECK_EQ(a[1], b[0]);
    return Dimensions(a[0], b[1]);
}

template<typename DT>
Dimensions Concat<DT>::ConcatenatedDimensions(
      const absl::Span<const Tensor<DT>> tensors, int axis) {
  // Must concatenate at least two tensors.
  CHECK_GE(tensors.size(), 2);

  Dimensions result(tensors[0].template as<DT>().dimensions());

  for (int i = 1; i < tensors.size(); ++i) {
    auto next_dims = tensors[i].template as<DT>().dimensions();

    // Check if Dimensions match.
    CHECK_EQ(result.rank(), next_dims.rank());

    for (int j = 0; j < result.rank(); ++j) {
      if (j == axis) {
        // Concatenation along this axis.
        result.set_dimension(j, result[j] + next_dims[j]);
      } else {
        CHECK_EQ(result[j], next_dims[j]);
      }
    }
  }
  return result;
}

template<typename DT>
void Concat<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& err = error.template as<DT>();

  // Determine sizes of every part.
  Dimensions::IndexType* sizes_memory =
      reinterpret_cast<Dimensions::IndexType*>(g->AllocateInArena(
          sizeof(Dimensions::IndexType) * this->num_inputs_));

  for (int i = 0; i < this->num_inputs_; ++i) {
    auto& input = this->sub_expression(i).template as<DT>();
    sizes_memory[i] = input.dimension(this->axis_);
  }

  auto sizes = absl::MakeConstSpan(sizes_memory, this->num_inputs_);

  // Split (inverse operation).
  auto split_error = g->Split(err, sizes, this->axis_, /*copy_sizes=*/ false);

  for (int i = 0; i < this->num_inputs_; ++i) {
    auto& input = this->sub_expression(i).template as<DT>();

    if (input.get()->is_differentiable()) {
      gradients->PropagateError(input.get(), g->GetOutput(split_error, i));
    }
  }
}

template<typename DT>
void Split<DT>::InitializeOutputTypes(
    Dimensions in_dims, TensorType* output_types,
    const absl::Span<const Dimensions::IndexType> sizes, int axis) {
  CHECK_GE(sizes.size(), 2);

  for (int i = 0; i < sizes.size(); ++i) {
    Dimensions out_dims(in_dims);
    out_dims.set_dimension(axis, sizes[i]);

    TensorType* addr = &output_types[i];
    new (addr) TensorType(CppTypeToDType<DT>::dtype, out_dims);
  }
}

template<typename DT>
void Split<DT>::ComputeGradients(TensorBase error, Gradients* gradients) {
  Graph* g = gradients->graph();

  auto& sa = this->sub_expression(0).template as<DT>();

  if (sa.get()->is_differentiable()) {
    auto errors_concat = g->ConcatTuple<DT>(&error, this->axis_,
                                            sa.tensor_type());
    gradients->PropagateError(sa.get(), errors_concat);
  }
}

template<typename DT>
void CopyToDevice<DT>::ComputeGradients(TensorBase error,
                                        Gradients* gradients) {
  Graph* g = gradients->graph();
  auto& sa = this->sub_expression(0).template as<DT>();
  auto& err = error.template as<DT>();

  if (sa.get()->is_differentiable()) {
    gradients->PropagateError(sa.get(), g->CopyToDevice(err, sa.device()));
  }
}


}  // namespace nodes
}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_TENSOR_OPS_IMPL_H_
