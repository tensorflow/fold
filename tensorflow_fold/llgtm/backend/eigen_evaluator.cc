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

#include <algorithm>
#include <unordered_map>

#include "tensorflow_fold/llgtm/backend/eigen_evaluator.h"
#include "tensorflow_fold/llgtm/backend/eigen_graph_implementation.h"
#include "tensorflow_fold/llgtm/tensor.h"
#include "tensorflow_fold/llgtm/tensor_ops.h"
#include "absl/memory/memory.h"

#ifdef GOOGLE_CUDA
#include "cuda/include/cuda.h"
#include "cuda/include/cuda_runtime.h"
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace llgtm {

namespace {

using Eigen::DefaultDevice;

#ifdef GOOGLE_CUDA
using Eigen::GpuDevice;
#endif  // GOOGLE_CUDA

// TODO(matthiasspringer): Implement GPU device support.

// Variable naming conventions:
// Input nodes ("sub expressions") of operations are named s*, e.g., sa for the
// first input and sb for the second input. Corresponding Eigen tensors
// consist of a single letter, starting from "a" for the first input. Eigen
// result variables are named "r".

// Helper macro for LLGTM_RANK_KERNEL_CASE. Generates a single case stmt.
// inside the switch statement.
#define LLGTM_RANK_KERNEL_CASE(CASE, KERNEL, NODE, GRAPH, DEVICE, ...) \
  case CASE: \
    KERNEL<__VA_ARGS__>(NODE, GRAPH, DEVICE); \
    break;

// Helper macro for operations whose Eigen backend implementation must be
// implemented with rank templatized variables/types. This macro generates a
// switch statement that calls the function KERNEL<COND>(GRAPH), where COND
// is an expression. The last parameter (var args) can be utilized to nest
// multiple kernel switches: Var args are used as additional template arguments
// to KERNEL. For example, ReduceSum requires two template parameters: rank and
// the number of reduced dimensions.
// Supports up to 4 dimensions. Should be kept in sync with Dimensions.
#define LLGTM_RANK_KERNEL_SWITCH(COND, KERNEL, NODE, GRAPH, DEVICE, ...) \
  switch (COND) { \
    LLGTM_RANK_KERNEL_CASE(1, KERNEL, NODE, GRAPH, DEVICE, 1, ##__VA_ARGS__) \
    LLGTM_RANK_KERNEL_CASE(2, KERNEL, NODE, GRAPH, DEVICE, 2, ##__VA_ARGS__) \
    LLGTM_RANK_KERNEL_CASE(3, KERNEL, NODE, GRAPH, DEVICE, 3, ##__VA_ARGS__) \
    LLGTM_RANK_KERNEL_CASE(4, KERNEL, NODE, GRAPH, DEVICE, 4, ##__VA_ARGS__) \
    default: LOG(FATAL) << "Unsupported dimensions: " << COND; \
  }

// Same as LLGTM_RANK_KERNEL_SWITCH but uses -1 as template argument in case
// rank (COND) is not within 1-4.
#define LLGTM_RANK_KERNEL_SWITCH_DEFAULT(COND, KERNEL, NODE, GRAPH, DEVICE, \
    ...) \
  switch (COND) { \
    LLGTM_RANK_KERNEL_CASE(1, KERNEL, NODE, GRAPH, DEVICE, 1, ##__VA_ARGS__) \
    LLGTM_RANK_KERNEL_CASE(2, KERNEL, NODE, GRAPH, DEVICE, 2, ##__VA_ARGS__) \
    LLGTM_RANK_KERNEL_CASE(3, KERNEL, NODE, GRAPH, DEVICE, 3, ##__VA_ARGS__) \
    LLGTM_RANK_KERNEL_CASE(4, KERNEL, NODE, GRAPH, DEVICE, 4, ##__VA_ARGS__) \
    default: \
      KERNEL<-1, ## __VA_ARGS__>(NODE, GRAPH, DEVICE); \
  }

#define LLGTM_RANK_KERNEL_SWITCH_2(COND, KERNEL, NODE, GRAPH, DEVICE, ...) \
  switch (COND) { \
    LLGTM_RANK_KERNEL_CASE(1, KERNEL, NODE, GRAPH, DEVICE, 1, ##__VA_ARGS__) \
    LLGTM_RANK_KERNEL_CASE(2, KERNEL, NODE, GRAPH, DEVICE, 2, ##__VA_ARGS__) \
  }

// Traits class for Eigen types.
template<typename DT, int Rank>
struct ETypes {
  typedef int Index;
  typedef Eigen::TensorMap<Eigen::Tensor<DT, Rank, Eigen::RowMajor, Index>,
                           Eigen::Aligned> Tensor;
  typedef typename Tensor::Dimensions Dimensions;
};

// Eigen rolls its own std::array, so we have to do this the hard way;
// there's no default constructor from std::array to eigen:array.
template<typename DT, int Rank>
void initialize_dims(typename ETypes<DT, Rank>::Dimensions* edims,
                     const Dimensions& dims) {
  DCHECK_EQ(dims.rank(), Rank);
  for (int i = 0; i < Rank; ++i)
    (*edims)[i] = dims[i];
}

// Helper method that creates an Eigen tensor of given dimensions from an LLGTM
// tensor. Assigning the return value of this method to an lvalue will not call
// the copy constructor due to return value optimization (returning a prvalue).
// See also: http://en.cppreference.com/w/cpp/language/copy_elision.
template<typename DT, int R, typename T>
typename ETypes<DT, R>::Tensor eigen_tensor(
    T* t, const Dimensions& d) {
  typename ETypes<DT, R>::Dimensions dims;
  initialize_dims<DT, R>(&dims, d);
  return typename ETypes<DT, R>::Tensor(t->result_data(), dims);
}

template<typename DT, int R, typename T>
typename ETypes<DT, R>::Tensor eigen_tensor(T* t) {
  return eigen_tensor<DT, R>(t, t->dimensions());
}

// Every CUDA kernel is launched with a 1D block size of min(result_size,
// kCUDAMaxBlockSize) and a 1D grid size of ceil(result_size / block_size).
// If the grid size would exceed kCUDAMaxGridSize, it is set to
// kCUDAMaxGridSize and every GPU thread handles multiple elements in a loop.

// Every CUDA kernel contains a for loop for this case. It runs over a variable
// "i" (iteration), starting from 0 and incremented by steps of "num_threads",
// which is the number of threads the CUDA kernel is launched with (i.e.,
// grid_size * block_size). The index of the element to be processed inside the
// loop is "i + tid" (tid ranging from 0 to num_threads among threads).

// Calculate block size for non-Eigen CUDA kernels.
template<typename DT>
uint32_t cuda_block_size(DT result_size) {
  return static_cast<uint32_t>(std::min<DT>(EigenEvaluator::kCUDAMaxBlockSize,
                                            result_size));
}

// Calculate grid size for non-Eigen CUDA kernels.
template<typename DT>
uint32_t cuda_grid_size(DT result_size) {
  auto block_size = cuda_block_size(result_size);
  return std::min(
      static_cast<uint32_t>((result_size + block_size - 1) / block_size),
      EigenEvaluator::kCUDAMaxGridSize);
}


template<class Device, typename NodeType>
class EigenKernel {
 public:
  static void Kernel(NodeType* node, Graph* /*graph*/, Device* device) {
    // TODO(matthiasspringer): GPU kernels should fall back to CPU kernels, but this
    // is difficult to implement in the current architecture.
    LOG(FATAL) << "No Eigen kernel found for "
               << opcode_name(node->opcode()) << ".";
  }
};


template<class Device>
class EigenKernel<Device, nodes::GetOutput> {
 public:
  static void Kernel(nodes::GetOutput* node, Graph* /*graph*/,
                     Device* /*device*/) {
    if (node->has_result())
      return;  // Some evaluators (e.g. TfEvaluator) may set outputs directly.

    nodes::TensorNodeBase* multi = node->sub_expression(0).get();
    // Sanity check.
    DCHECK_EQ(node->offset_, multi->result_data_offset(node->output_index_));
    // Grab result data from the appropriate offset.
    char* rdata = reinterpret_cast<char*>(multi->result_data());
    node->set_result_data(rdata + node->offset_);
  }
};

template<typename DT>
class EigenKernel<DefaultDevice, nodes::TensorValue<DT>> {
 public:
  static void Kernel(nodes::TensorValue<DT>* node, Graph* /*graph*/,
                     DefaultDevice* device) {
    if (node->allocates_result_data()) {
      // Make copy of data (properly aligned).
      DCHECK(node->has_result());
      size_t bytes = sizeof(DT) * node->dimensions().num_elements();
      device->memcpy(node->result_data(), node->data_ptr_, bytes);
    } else {
      // Data is already aligned, no copy necessary.
      node->set_result_data(node->data_ptr_);
    }
  }
};

#ifdef GOOGLE_CUDA
template<typename DT>
class EigenKernel<GpuDevice, nodes::TensorValue<DT>> {
 public:
  static void Kernel(nodes::TensorValue<DT>* node, Graph* /*graph*/,
                     GpuDevice* device) {
    if (node->allocates_result_data()) {
      DCHECK(node->has_result());
      size_t bytes = sizeof(DT) * node->dimensions().num_elements();
      device->memcpyHostToDevice(node->result_data(), node->data_ptr_, bytes);
      device->synchronize();
    } else {
      // TODO(matthiasspringer): Handle data that is already on the GPU.
      LOG(FATAL) << "TensorValue can handle only host input data.";
    }
  }
};
#endif

template<class Device, typename DT>
class EigenKernel<Device, nodes::TensorVariable<DT>> {
 public:
  static void Kernel(nodes::TensorVariable<DT>* node, Graph* /*graph*/,
                     Device* /*device*/) {
    node->set_result_data(node->variable_->data(node->device()));
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Zeros<DT>> {
 public:
  static void Kernel(nodes::Zeros<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor r(r_data, d);
    r.device(*device) = r.constant(static_cast<DT>(0));
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::ConstantFromScalar<DT>> {
 public:
  static void Kernel(nodes::ConstantFromScalar<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor r(r_data, d);
    r.device(*device) = r.constant(node->value_);
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::UniformRandom<DT>> {
 public:
  static void Kernel(nodes::UniformRandom<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor r(r_data, d);
    r.device(*device) =
        r.nullaryExpr(Eigen::internal::UniformRandomGenerator<DT>(node->seed_));
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::NormalRandom<DT>> {
 public:
  static void Kernel(nodes::NormalRandom<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor r(r_data, d);
    r.device(*device) =
        r.nullaryExpr(Eigen::internal::NormalRandomGenerator<DT>(node->seed_));
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Broadcast<DT>> {
 public:
  static void Kernel(nodes::Broadcast<DT>* node, Graph* graph,
                     Device* device) {
    const auto& sa = node->sub_expression(0).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH(sa.dimensions().rank(), KernelR,
                             node, graph, device);
  }

 private:
  template<int R>
  static void KernelR(nodes::Broadcast<DT>* node, Graph* /*graph*/,
                      Device* device) {
    DCHECK(node->has_result());

    const auto a = eigen_tensor<DT, R>(
        &node->sub_expression(0).template as<DT>());
    auto r = eigen_tensor<DT, R>(node);

    // Eigen broadcast_dims are the number of times to broadcast on each dim.
    Eigen::array<Dimensions::IndexType, R> eigen_broadcast_dims;
    for (int i = 0; i < R; ++i) {
      if (a.dimension(i) == r.dimension(i)) {
        eigen_broadcast_dims[i] = 1;
      } else if (a.dimension(i) == 1) {
        eigen_broadcast_dims[i] = r.dimension(i);
      } else {
        // Should already fail in CheckBroadcastDimensions.
        LOG(FATAL) << "Invalid dimensions for broadcast.";
      }
    }

    r.device(*device) = a.broadcast(eigen_broadcast_dims);
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::ReduceSum<DT>> {
 public:
  static void Kernel(nodes::ReduceSum<DT>* node, Graph* graph,
                     Device* device) {
    auto& sa = node->sub_expression(0).template as<DT>();
    if (node->num_reductions() == 0) {
      // Do nothing in this case; ReduceSum is the identity function.
      node->set_result_data(sa.result_data());
      return;
    }

    // Switch based on runtime rank.
    int rank = sa.dimensions().rank();
    LLGTM_RANK_KERNEL_SWITCH(rank, KernelR, node, graph, device);
  }

 private:
  // ReduceSum Eigen kernel. We can be sure that N <= R, i.e., the number of
  // reduce dimensions is less than or equal to rank. Also see comment of the
  // second ReduceSumKernelNR for N > R cases.
  template<int N, int R>
  static typename std::enable_if<(N <= R), void>::type
  KernelNR(nodes::ReduceSum<DT>* node, Graph* /*graph*/, Device* device) {
    static_assert(N <= R, "Reduce indices must be fewer than rank.");
    DCHECK(node->has_result());

    const auto a = eigen_tensor<DT, R>(
        &node->sub_expression(0).template as<DT>());
    auto r = eigen_tensor<DT, R>(node);

    // Calculate reduce dimensions.
    typename Eigen::array<int, N> eigen_reduce_indices;
    int index = 0;

    for (int i = 0; i < R; ++i) {
      if (node->dimension(i) < a.dimension(i)) {
        eigen_reduce_indices[index++] = i;
      }
    }

    DCHECK_EQ(index, N);
    r.device(*device) = a.sum(eigen_reduce_indices).reshape(r.dimensions());
  }

  // Due to the combination of two LLGTM_RANK_KERNEL_SWITCH, the preprocessor
  // generates certain combinations that do not compile. For example, the code
  // (template args combination) for R = 3 and N = 4 does not compile. (Eigen
  // reports a compile error. Number of reduce dimensions must be less/equal to
  // rank.). With std::enable_if, we can generate different code for that case.
  // This code will never be reached at runtime.
  template<int N, int R>
  static typename std::enable_if<(N > R), void>::type
  KernelNR(nodes::ReduceSum<DT>* /*node*/, Graph* /*graph*/,
           Device* /*device*/) {
    // This code should never be reached at runtime.
    LOG(FATAL) << "Number of reduce indices must be smaller than rank.";
  }

  template<int R>
  static void KernelR(nodes::ReduceSum<DT>* node, Graph* graph,
                      Device* device) {
    // Switch based on number of reduce dimensions.
    LLGTM_RANK_KERNEL_SWITCH(node->num_reductions(), KernelNR,
                             node, graph, device, R);
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Transpose<DT>> {
 public:
  static void Kernel(nodes::Transpose<DT>* node, Graph* graph,
                     Device* device) {
    const auto& sa = node->sub_expression(0).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH(sa.dimensions().rank(), KernelR,
                             node, graph, device);
  }

 private:
  template<int R>
  static void KernelR(nodes::Transpose<DT>* node, Graph* /*graph*/,
                      Device* device) {
    DCHECK(node->has_result());

    const auto a = eigen_tensor<DT, R>(
        &node->sub_expression(0).template as<DT>());
    auto r = eigen_tensor<DT, R>(node);

    // Copy indices to Eigen::array.  Ugh.
    typename Eigen::array<int, R> eigen_indices;
    for (int i = 0; i < R; ++i) {
      eigen_indices[i] = node->indices_[i];
    }

    r.device(*device) = a.shuffle(eigen_indices);
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Reshape<DT>> {
 public:
  static void Kernel(nodes::Reshape<DT>* node, Graph* /*graph*/,
                     Device* /*device*/) {
    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    node->set_result_data(a_data);
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Concat<DT>> {
 public:
  static void Kernel(nodes::Concat<DT>* node, Graph* graph, Device* device) {
    auto& sa = node->sub_expression(0).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH(sa.dimensions().rank(), KernelR,
                             node, graph, device);
  }

 private:
  template<int N, int R>
  static typename std::enable_if<N == 1, void>::type
  KernelNR(nodes::Concat<DT>* /*node*/, Graph* /*graph*/, Device* /*device*/) {
    LOG(FATAL) << "Concatenation requires at least two tensors.";
  }

  template<int N, int R>
  static typename std::enable_if<N == 2, void>::type
  KernelNR(nodes::Concat<DT>* node, Graph* /*graph*/, Device* device) {
    const auto t0 = eigen_tensor<DT, R>(
        &node->sub_expression(0).template as<DT>());
    const auto t1 = eigen_tensor<DT, R>(
        &node->sub_expression(1).template as<DT>());
    auto r = eigen_tensor<DT, R>(node);

    r.device(*device) = t0.concatenate(t1, node->axis_);
  }

  template<int N, int R>
  static typename std::enable_if<N == 3, void>::type
  KernelNR(nodes::Concat<DT>* node, Graph* /*graph*/, Device* device) {
    const auto t0 = eigen_tensor<DT, R>(
        &node->sub_expression(0).template as<DT>());
    const auto t1 = eigen_tensor<DT, R>(
        &node->sub_expression(1).template as<DT>());
    const auto t2 = eigen_tensor<DT, R>(
        &node->sub_expression(2).template as<DT>());
    auto r = eigen_tensor<DT, R>(node);

    r.device(*device) = t0.concatenate(t1, node->axis_)
                         .concatenate(t2, node->axis_);
  }

  template<int N, int R>
  static typename std::enable_if<N == 4, void>::type
  KernelNR(nodes::Concat<DT>* node, Graph* /*graph*/, Device* device) {
    const auto t0 = eigen_tensor<DT, R>(
        &node->sub_expression(0).template as<DT>());
    const auto t1 = eigen_tensor<DT, R>(
        &node->sub_expression(1).template as<DT>());
    const auto t2 = eigen_tensor<DT, R>(
        &node->sub_expression(2).template as<DT>());
    const auto t3 = eigen_tensor<DT, R>(
        &node->sub_expression(3).template as<DT>());
    auto r = eigen_tensor<DT, R>(node);

    r.device(*device) = t0.concatenate(t1, node->axis_)
                         .concatenate(t2, node->axis_)
                         .concatenate(t3, node->axis_);
  }

  // Perform concatenation. Eigen won't let us concatenate empty tensors,
  // so we have to handle all cases (1 - 4 tensors).
  template<int R>
  static void KernelConcatUpToFour(
      Device* device, int num_inputs, int iteration, int axis,
      const std::vector<DT*>& data_a, const std::vector<DT*>& data_r,
      const std::vector<Dimensions::IndexType>& size_a,
      const typename ETypes<DT, R>::Dimensions& dimensions_base,
      const typename ETypes<DT, R>::Dimensions& dimensions_r) {
    // Eigen tensors for input of concatenation.
    typename ETypes<DT, R>::Dimensions dims_0(dimensions_base);
    dims_0[axis] = size_a[iteration];
    const typename ETypes<DT, R>::Tensor t_0(data_a[iteration], dims_0);

    typename ETypes<DT, R>::Dimensions dims_1(dimensions_base);
    dims_1[axis] = size_a[iteration + 1];
    const typename ETypes<DT, R>::Tensor t_1(data_a[iteration + 1], dims_1);

    typename ETypes<DT, R>::Dimensions dims_2(dimensions_base);
    dims_2[axis] = size_a[iteration + 2];
    const typename ETypes<DT, R>::Tensor t_2(data_a[iteration + 2], dims_2);

    typename ETypes<DT, R>::Dimensions dims_3(dimensions_base);
    dims_3[axis] = size_a[iteration + 3];
    const typename ETypes<DT, R>::Tensor t_3(data_a[iteration + 3], dims_3);

    typename ETypes<DT, R>::Tensor t_r(data_r[iteration / 4], dimensions_r);

    if (num_inputs - iteration >= 4) {
      // No dummies.
      t_r.device(*device) = t_0.concatenate(t_1, axis)
                              .concatenate(t_2, axis)
                              .concatenate(t_3, axis);
    } else if (num_inputs - iteration == 3) {
      // Last one is dummy.
      t_r.device(*device) = t_0.concatenate(t_1, axis)
                              .concatenate(t_2, axis);
    } else if (num_inputs - iteration == 2) {
      // Last  two are dummies.
      t_r.device(*device) = t_0.concatenate(t_1, axis);
    } else {
      // Only one tensor (no concatenation necessary).
      DCHECK_EQ(num_inputs - iteration, 1);
      t_r.device(*device) = t_0;
    }
  }

  // Deallocate temporary tensor data buffers.
  static void KernelDeallocate(bool is_first_level, int iteration,
                               const std::vector<DT*>& data,
                               Graph* graph, Device* device) {
    if (!is_first_level) {
      // Free previously allocated memory. Do not free input tensors of this
      // operation. I.e., do not free data if this is the first iteration.
      device->deallocate(data[iteration]);
      device->deallocate(data[iteration + 1]);
      device->deallocate(data[iteration + 2]);
      device->deallocate(data[iteration + 3]);
    }
  }

  template<int R>
  static void KernelLevel(
      const std::vector<DT*> data_a,
      const std::vector<Dimensions::IndexType>& size_a,
      std::vector<DT*>* data_r,
      std::vector<Dimensions::IndexType>* size_r,
      int num_results,
      int num_inputs,
      int num_inputs_with_dummies,
      bool is_first_level,
      nodes::Concat<DT>* node,
      Graph* graph,
      Device* device) {
    // Result dimensions.
    typename ETypes<DT, R>::Dimensions dimensions_base;
    initialize_dims<DT, R>(&dimensions_base, node->dimensions());
    Dimensions::IndexType base_size =
        node->dimensions().num_elements() / node->dimension(node->axis_);

    for (int i = 0; i < num_inputs_with_dummies; i += 4) {
      // Determine dimensions of temporary result.
      typename ETypes<DT, R>::Dimensions dimensions_r(dimensions_base);
      dimensions_r[node->axis_] = size_a[i] + size_a[i + 1] +
                                  size_a[i + 2] + size_a[i + 3];
      (*size_r)[i / 4] = dimensions_r[node->axis_];

      if (num_results == 1) {
        // Store result of last contenation in result tensor.
        (*data_r)[i / 4] = node->result_data();
      } else {
        // Allocate new memory for temporary result.
        (*data_r)[i / 4] = reinterpret_cast<DT*>(device->allocate(
            base_size * dimensions_r[node->axis_] * sizeof(DT)));
      }

      KernelConcatUpToFour<R>(device, num_inputs, /*iteration=*/ i,
                              node->axis_, data_a, *data_r, size_a,
                              dimensions_base, dimensions_r);
      KernelDeallocate(is_first_level, /*iteration=*/ i, data_a,
                       graph, device);
    }
  }

  // This is the general case where more than 4 tensors are concatenated. To
  // minimize the amount of copying, we perform evenly distributed, nested
  // concatenations.
  // E.g., for 8 inputs: Concat(Conat(1, 2, 3, 4), Concat(5, 6, 7, 8)), instead
  // of: Concat(Concat(Concat(1, 2, 3, 4), 5, 6, 7), 8).
  // This can possibly be rewritten with C++14 templates to generate code that
  // generates a single Eigen expression with the correct number of concats.
  template<int N, int R>
  static typename std::enable_if<N == -1, void>::type
  KernelNR(nodes::Concat<DT>* node, Graph* graph, Device* device) {
    DCHECK_GE(node->num_inputs(), 5);

    int num_inputs = node->num_inputs();
    int num_results;

    // Input data.
    std::vector<DT*> data_a(num_inputs);
    std::vector<Dimensions::IndexType> size_a(num_inputs);

    for (int i = 0; i < num_inputs; ++i) {
      auto& s = node->sub_expression(i).template as<DT>();
      data_a[i] = s.result_data();
      size_a[i] = s.dimension(node->axis_);
    }

    // Result data.
    int first_result_size = (num_inputs + 4 - 1) / 4;
    std::vector<DT*> data_r(first_result_size);
    std::vector<Dimensions::IndexType> size_r(first_result_size);

    bool first_level = true;

    // Run another iteration as long there are more than one tensors to
    // concatenate.
    while (num_inputs > 1) {
      // Add dummy tensors. To remove some special cases, we make sure that the
      // number of tensors on every level is always divisible by 4. Dummy
      // tensors have a size of zero and do not hold any data (nullptr).
      // Freeing a nullptr is legal and a no-op.
      int num_inputs_with_dummies = TensorType::aligned_size(num_inputs, 4);
      DCHECK_LT(num_inputs_with_dummies - num_inputs, 4);
      DCHECK_EQ(num_inputs_with_dummies % 4, 0);

      data_a.resize(num_inputs_with_dummies, nullptr);
      size_a.resize(num_inputs_with_dummies, 0);

      num_results = num_inputs_with_dummies / 4;
      data_r.resize(num_results);
      size_r.resize(num_results);

      // Concatenate tensors in data_a/size_a and store results in
      // data_r/size_r.
      KernelLevel<R>(data_a, size_a, &data_r, &size_r, num_results, num_inputs,
                     num_inputs_with_dummies, first_level, node, graph,
                     device);

      // Swap input and output vectors.
      data_a.swap(data_r);
      size_a.swap(size_r);
      num_inputs = num_results;

      // From now on we can deallocate inputs (temporaries).
      first_level = false;
    }
  }

  template<int R>
  static void KernelR(nodes::Concat<DT>* node, Graph* graph, Device* device) {
    // Switch based on number of inputs.
    LLGTM_RANK_KERNEL_SWITCH_DEFAULT(node->num_inputs(), KernelNR,
                                     node, graph, device, R);
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Split<DT>> {
 public:
  static void Kernel(nodes::Split<DT>* node, Graph* graph, Device* device) {
    auto& sa = node->sub_expression(0).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH(sa.dimensions().rank(), KernelR,
                             node, graph, device);
  }

 private:
  template<int R>
  static void KernelR(nodes::Split<DT>* node, Graph* /*graph*/,
                      Device* device) {
    DCHECK(node->has_result());

    // Input tensor.
    const auto a = eigen_tensor<DT, R>(
        &node->sub_expression(0).template as<DT>());

    // Result dimensions. Will be updated for every output.
    typename ETypes<DT, R>::Dimensions dims_r(a.dimensions());

    typename Eigen::array<int, R> offsets;
    typename Eigen::array<int, R> extents;
    for (int i = 0; i < R; ++i) {
      offsets[i] = 0;
      extents[i] = a.dimension(i);
    }

    offsets[node->axis_] = 0;
    extents[node->axis_] = 0;

    int memory_offset = 0;

    for (int i = 0; i < node->num_outputs(); ++i) {
      offsets[node->axis_] += extents[node->axis_];
      dims_r[node->axis_] = extents[node->axis_] = node->sizes_[i];

      // Generate i^th output.
      DT *result_data = reinterpret_cast<DT*>(
          static_cast<char*>(node->result_data()) + memory_offset);
      typename ETypes<DT, R>::Tensor r(result_data, dims_r);
      r.device(*device) = a.slice(offsets, extents);

      memory_offset += node->output_type(i).aligned_memory_size();
    }
  }
};

template<typename DT>
class EigenKernel<DefaultDevice, nodes::Gather<DT>> {
 public:
  static void Kernel(nodes::Gather<DT>* node, Graph* /*graph*/,
                     DefaultDevice* device) {
    DCHECK(node->has_result());
    if (node->axis_ != 0) {
      LOG(FATAL) << "Gather only supported on axis 0 at this time.";
    }

    const auto& dims = node->dimensions();
    Dimensions::IndexType num_rows = dims[0];  // axis_ is 0
    size_t slice_size = 1;
    for (int i = 1, r = dims.rank(); i < r; ++i) slice_size *= dims[i];

    int32_t* indices = node->sub_expression(0)
        .template result_data_as<int32_t>();
    DT* sdata = node->sub_expression(1).template result_data_as<DT>();
    DT* rdata = node->result_data();

    // TODO(matthiasspringer): Replace loop with efficient GPU kernel.
    for (int j = 0; j < num_rows; ++j) {
      DT* sdata_slice = sdata + indices[j]*slice_size;
      device->memcpy(rdata, sdata_slice, slice_size * sizeof(DT));
      rdata += slice_size;
    }
  }
};

#ifdef GOOGLE_CUDA
template<typename DT>
class EigenKernel<GpuDevice, nodes::Gather<DT>> {
 public:
  static void Kernel(nodes::Gather<DT>* node, Graph* /*graph*/,
                     GpuDevice* device) {
    DCHECK(node->has_result());
    if (node->axis_ != 0) {
      LOG(FATAL) << "Gather only supported on axis 0 at this time.";
    }

    const auto& dims = node->dimensions();
    Dimensions::IndexType num_rows = dims[0];  // axis_ is 0
    size_t slice_size = 1;
    for (int i = 1, r = dims.rank(); i < r; ++i) slice_size *= dims[i];

    // Copy indices to host.
    int32_t* indices = node->sub_expression(0)
        .template result_data_as<int32_t>();
    size_t indices_bytes = num_rows*sizeof(int32_t);
    auto* indices_cpu = reinterpret_cast<int32_t*>(malloc(indices_bytes));
    device->memcpyDeviceToHost(indices_cpu, indices, indices_bytes);
    device->synchronize();

    DT* sdata = node->sub_expression(1).template result_data_as<DT>();
    DT* rdata = node->result_data();

    for (int j = 0; j < num_rows; ++j) {
      DT* sdata_slice = sdata + indices_cpu[j]*slice_size;
      device->memcpy(rdata, sdata_slice, slice_size * sizeof(DT));
      rdata += slice_size;
    }

    device->synchronize();
    free(indices_cpu);
  }
};
#endif

template<typename DT>
class EigenKernel<DefaultDevice, nodes::Scatter<DT>> {
 public:
  static void Kernel(nodes::Scatter<DT>* node, Graph* /*graph*/,
                     DefaultDevice* /*device*/) {
    // This version only works on CPU.
    DCHECK(node->has_result());
    if (node->axis_ != 0) {
      LOG(FATAL) << "Scatter only supported on axis 0 at this time.";
    }

    auto& dims = node->sub_expression(1).output_type(0).dimensions();
    Dimensions::IndexType num_rows = dims[0];  // axis_ is 0.
    size_t slice_size = 1;
    for (int i = 1, r = dims.rank(); i < r; ++i) slice_size *= dims[i];

    int32_t* indices = node->sub_expression(0)
        .template result_data_as<int32_t>();
    DT* sdata = node->sub_expression(1).template result_data_as<DT>();
    DT* rdata = node->result_data();

    // Initialize output to zero.
    typename ETypes<DT, 1>::Dimensions rdims(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor r(reinterpret_cast<DT*>(rdata), rdims);
    r.setZero();

    // Now scatter.
    for (int j = 0; j < num_rows; ++j) {
      DT* rdata_slice = rdata + indices[j]*slice_size;
      // We add the data here, because indices may contain duplicates.
      for (int k = 0; k < slice_size; ++k) rdata_slice[k] += sdata[k];
      sdata += slice_size;
    }
  }
};

#ifdef GOOGLE_CUDA
template<typename DT>
__global__ void ScatterCUDAKernel(DT* rdata_slice, DT* sdata_slice,
                                  size_t slice_size) {
  auto tid = blockDim.x*blockIdx.x + threadIdx.x;
  auto num_threads = blockDim.x*gridDim.x;

  // See comment of cuda_block_size for work distribution.
  for (int i = 0; i < slice_size; i += num_threads) {
    if (tid + i < slice_size) {
      // We add the data here, because indices may contain duplicates.
      rdata_slice[tid + i] += sdata_slice[tid + i];
    }
  }
}

template<typename DT>
class EigenKernel<GpuDevice, nodes::Scatter<DT>> {
 public:
  static void Kernel(nodes::Scatter<DT>* node, Graph* /*graph*/,
                     GpuDevice* device) {
    DCHECK(node->has_result());
    if (node->axis_ != 0) {
      LOG(FATAL) << "Scatter only supported on axis 0 at this time.";
    }

    auto& dims = node->sub_expression(1).output_type(0).dimensions();
    Dimensions::IndexType num_rows = dims[0];  // axis_ is 0.
    size_t slice_size = 1;
    for (int i = 1, r = dims.rank(); i < r; ++i) slice_size *= dims[i];

    // Copy indices to host.
    int32_t* indices = node->sub_expression(0)
        .template result_data_as<int32_t>();
    size_t indices_bytes = num_rows*sizeof(int32_t);
    auto* indices_cpu = reinterpret_cast<int32_t*>(malloc(indices_bytes));
    device->memcpyDeviceToHost(indices_cpu, indices, indices_bytes);
    device->synchronize();

    DT* sdata = node->sub_expression(1).template result_data_as<DT>();
    DT* rdata = node->result_data();

    // Initialize output to zero.
    typename ETypes<DT, 1>::Dimensions rdims(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor r(reinterpret_cast<DT*>(rdata), rdims);
    r.device(*device) = r.constant(static_cast<DT>(0));

    // Now scatter.
    // TODO(matthiasspringer): Kernel is not efficient.
    for (int j = 0; j < num_rows; ++j) {
      DT* rdata_slice = rdata + indices_cpu[j]*slice_size;

      ScatterCUDAKernel<DT><<<cuda_grid_size(slice_size),
                              cuda_block_size(slice_size),
                              0, device->stream()>>>(
          rdata_slice, sdata, slice_size);
      device->synchronize();
      sdata += slice_size;
    }

    free(indices_cpu);
  }
};
#endif

template<class Device, typename DT>
class EigenKernel<Device, nodes::Negative<DT>> {
 public:
  static void Kernel(nodes::Negative<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());

    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = -a;
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Reciprocal<DT>> {
 public:
  static void Kernel(nodes::Reciprocal<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());

    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = a.inverse();
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Add<DT>> {
 public:
  static void Kernel(nodes::Add<DT>* node, Graph* /*graph*/, Device* device) {
    DCHECK(node->has_result());

    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    DT* b_data = node->sub_expression(1).template result_data_as<DT>();
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor b(b_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = a + b;
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::AssignAdd<DT>> {
 public:
  static void Kernel(nodes::AssignAdd<DT>* node, Graph* /*graph*/,
                     Device* device) {
    Variable<DT>* var = node->variable();
    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    DT* r_data = var->data(node->device());

    typename ETypes<DT, 1>::Dimensions d(var->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) += a;
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Multiply<DT>> {
 public:
  static void Kernel(nodes::Multiply<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());

    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    DT* b_data = node->sub_expression(1).template result_data_as<DT>();
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor b(b_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = a * b;
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Matmul<DT>> {
 public:
  static void Kernel(nodes::Matmul<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());

    auto& sa = node->sub_expression(0).template as<DT>();
    auto& sb = node->sub_expression(1).template as<DT>();
    DT* r_data = node->result_data();

    typename ETypes<DT, 2>::Dimensions dim_a(sa.dimension(0), sa.dimension(1));
    typename ETypes<DT, 2>::Dimensions dim_b(sb.dimension(0), sb.dimension(1));
    typename ETypes<DT, 2>::Dimensions dim_r(node->dimension(0),
                                             node->dimension(1));

    typename ETypes<DT, 2>::Tensor a(sa.result_data(), dim_a);
    typename ETypes<DT, 2>::Tensor b(sb.result_data(), dim_b);
    typename ETypes<DT, 2>::Tensor r(r_data, dim_r);

    Eigen::array<Eigen::IndexPair<int>, 1> product_dims =
        { Eigen::IndexPair<int>(1, 0) };
    r.device(*device) = a.contract(b, product_dims);
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Relu<DT>> {
 public:
  static void Kernel(nodes::Relu<DT>* node, Graph* /*graph*/, Device* device) {
    DCHECK(node->has_result());

    auto a_data = node->sub_expression(0).template result_data_as<DT>();
    auto r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = a.cwiseMax(static_cast<DT>(0));
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::ReluGrad<DT>> {
 public:
  static void Kernel(nodes::ReluGrad<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());

    auto error_data = node->sub_expression(0).template result_data_as<DT>();
    auto a_data = node->sub_expression(1).template result_data_as<DT>();
    auto r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor error(error_data, d);
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = (a > static_cast<DT>(0)).select(
        error, error.constant(static_cast<DT>(0)));
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Sigmoid<DT>> {
 public:
  static void Kernel(nodes::Sigmoid<DT>* node, Graph* /*graph*/,
                     Device* device) {
    DCHECK(node->has_result());

    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = a.sigmoid();
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Tanh<DT>> {
 public:
  static void Kernel(nodes::Tanh<DT>* node, Graph* /*graph*/, Device* device) {
    DCHECK(node->has_result());

    DT* a_data = node->sub_expression(0).template result_data_as<DT>();
    DT* r_data = node->result_data();

    typename ETypes<DT, 1>::Dimensions d(node->dimensions().num_elements());
    typename ETypes<DT, 1>::Tensor a(a_data, d);
    typename ETypes<DT, 1>::Tensor r(r_data, d);

    r.device(*device) = a.tanh();
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::Softmax<DT>> {
 public:
  static void Kernel(nodes::Softmax<DT>* node, Graph* graph, Device* device) {
    auto& sa = node->sub_expression(0).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH(sa.dimensions().rank(), KernelR,
                             node, graph, device);
  }

 private:
  template<int R>
  static void KernelR(nodes::Softmax<DT>* node, Graph* /*graph*/,
                      Device* device) {
    DCHECK(node->has_result());

    auto& sa = node->sub_expression(0).template as<DT>();

    // Dimensions of input tensor a.
    typename ETypes<DT, R>::Dimensions dims_a;
    initialize_dims<DT, R>(&dims_a, sa.dimensions());

    typename ETypes<DT, R>::Tensor a(sa.result_data(), dims_a);
    typename ETypes<DT, R>::Tensor r(node->result_data(), dims_a);

    // Same dimensions as tensor a, but last dimension is 1.
    typename ETypes<DT, R>::Dimensions dims_reduced_full(dims_a);
    dims_reduced_full[R - 1] = 1;

    // Broadcast dimensions. Determines how often to copy data in a dimension.
    // Only the last dimension is copied here.
    Eigen::array<Dimensions::IndexType, R> dims_broadcast;

    for (int i = 0; i < R - 1; ++i) {
      dims_broadcast[i] = 1;
    }

    dims_broadcast[R - 1] = sa.dimension(R - 1);

    // Reduce dimensions. Determines in which dimensions data is reduced.
    typename Eigen::array<int, 1> dims_reduce({R - 1});

    // Compute softmax(a) = exp(a - max(a) - log(sum(e(a_i - max(a))))). This
    // formula is more numerically stable.

    // Compute the maximum value (for every element in the batch).
    auto max = a.maximum(dims_reduce).eval();
    auto max_bc = max.reshape(dims_reduced_full).broadcast(dims_broadcast);
    // Compute log(sum(e(a))).
    auto logsumexp = max + (a - max_bc).exp().sum(dims_reduce).eval().log();
    auto logsumexp_bc = logsumexp.reshape(dims_reduced_full)
        .broadcast(dims_broadcast);
    // Compute softmax.
    r.device(*device) = (a - logsumexp_bc).exp();
  }
};

template<class Device, typename DT>
class EigenKernel<Device, nodes::SoftmaxCrossEntropy<DT>> {
 public:
  static void Kernel(nodes::SoftmaxCrossEntropy<DT>* node, Graph* graph,
                     Device* device) {
    auto& sa = node->sub_expression(0).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH(sa.dimensions().rank(), KernelR,
                             node, graph, device);
  }

 private:
  template<int R>
  static void KernelR(nodes::SoftmaxCrossEntropy<DT>* node, Graph* /*graph*/,
                      Device* device) {
    DCHECK(node->has_result());

    auto& s_labels = node->sub_expression(0).template as<DT>();
    auto& s_softmax = node->sub_expression(1).template as<DT>();

    // Dimensions of input tensors.
    typename ETypes<DT, R>::Dimensions dims;
    initialize_dims<DT, R>(&dims, s_labels.dimensions());

    // Dimensions of result tensor: last dimension should be 1.
    typename ETypes<DT, R>::Dimensions dims_r(dims);
    dims_r[R - 1] = 1;

    typename ETypes<DT, R>::Tensor labels(s_labels.result_data(), dims);
    typename ETypes<DT, R>::Tensor softmax(s_softmax.result_data(), dims);
    typename ETypes<DT, R>::Tensor r(node->result_data(), dims_r);

    // Reduce dimensions. Determines in which dimensions data is reduced.
    typename Eigen::array<int, 1> dims_reduce({R - 1});

    // Compute cross-entropy(Y, P) = -sum(Y * log(P)).
    // TODO(matthiasspringer): This can be more efficient if softmax is recomputed.
    r.device(*device) = -(labels * softmax.log())
        .sum(dims_reduce).reshape(dims_r);
  }
};

template<typename DT>
class EigenKernel<DefaultDevice, nodes::SoftmaxSparseCrossEntropy<DT>> {
 public:
  static void Kernel(nodes::SoftmaxSparseCrossEntropy<DT>* node, Graph* graph,
                     DefaultDevice* device) {
    auto& sb = node->sub_expression(1).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH_2(sb.dimensions().rank(), KernelR,
                               node, graph, device);
  }

 private:
  // Store data for "Softmax Sparse Cross-Entropy" in the result tensor.
  // This is a separate function, because different code is required based on
  // whether this is a rank 1 or rank 2 (with batching) operation.
  static void sparse_cross_entropy_store_data(
      typename ETypes<DT, 2>::Tensor* r,
      const typename ETypes<DT, 2>::Dimensions& dims,
      const typename ETypes<DT, 2>::Tensor& softmax,
      int* labels_data, DefaultDevice* /*device*/) {
    typename ETypes<DT, 2>::Tensor& result = *r;
    for (int i = 0; i < dims[0]; ++i) {
      result(i, 0) = -std::log(softmax(i, labels_data[i]));
    }
  }

  static void sparse_cross_entropy_store_data(
      typename ETypes<DT, 1>::Tensor* r,
      const typename ETypes<DT, 1>::Dimensions dims,
      const typename ETypes<DT, 1>::Tensor softmax,
      int* labels_data, DefaultDevice* /*device*/) {
    (*r)(0) = -std::log(softmax(labels_data[0]));
  }

  template<int R>
  static void KernelR(nodes::SoftmaxSparseCrossEntropy<DT>* node,
                      Graph* /*graph*/, DefaultDevice* device) {
    static_assert(R <= 2, "Cross entropy probabilities rank must be <= 2");
    DCHECK(node->has_result());

    auto& s_labels = node->sub_expression(0).template as<int>();
    auto& s_softmax = node->sub_expression(1).template as<DT>();

    // Dimensions of input tensors.
    typename ETypes<DT, R>::Dimensions dims_s;
    initialize_dims<DT, R>(&dims_s, s_softmax.dimensions());

    // Dimensions of result tensor: last dimension should be 1.
    typename ETypes<DT, R>::Dimensions dims_r(dims_s);
    dims_r[R - 1] = 1;

    typename ETypes<DT, R>::Tensor softmax(s_softmax.result_data(), dims_s);
    typename ETypes<DT, R>::Tensor r(node->result_data(), dims_r);

    // Gather probabilities specified by s_labels.
    sparse_cross_entropy_store_data(&r, dims_s, softmax,
                                    s_labels.result_data(), device);
  }
};

#ifdef GOOGLE_CUDA
template<typename DT>
__global__ void SoftmaxSparseCrossEntropyCUDAKernel(
    DT* result, const int* labels, const DT* softmax,
    Dimensions::IndexType batch_size, Dimensions::IndexType num_labels) {
  auto tid = blockDim.x*blockIdx.x + threadIdx.x;
  auto num_threads = blockDim.x*gridDim.x;

  // See comment of cuda_block_size for work distribution.
  for (int i = 0; i < batch_size; i += num_threads) {
    if (tid + i < batch_size) {
      // softmax_index is index of softmax[tid, labels[tid]] in row major.
      auto softmax_index = num_labels*(tid + i) + labels[tid + i];
      result[tid + i] = -log(softmax[softmax_index]);
    }
  }
}

template<typename DT>
class EigenKernel<GpuDevice, nodes::SoftmaxSparseCrossEntropy<DT>> {
 public:
  static void Kernel(nodes::SoftmaxSparseCrossEntropy<DT>* node, Graph* graph,
                     GpuDevice* device) {
    DCHECK(node->has_result());

    auto& s_labels = node->sub_expression(0).template as<int>();
    auto& s_softmax = node->sub_expression(1).template as<DT>();
    CHECK_LE(s_softmax.dimensions().rank(), 2);

    auto result_size = node->dimensions().num_elements();
    SoftmaxSparseCrossEntropyCUDAKernel<DT><<<cuda_grid_size(result_size),
                                              cuda_block_size(result_size),
                                              0, device->stream()>>>(
        node->result_data(), s_labels.result_data(), s_softmax.result_data(),
        result_size, s_softmax.dimension(1));
    device->synchronize();
  }
};
#endif

template<typename DT>
class EigenKernel<DefaultDevice, nodes::SoftmaxSparseCrossEntropyGrad<DT>> {
 public:
  static void Kernel(nodes::SoftmaxSparseCrossEntropyGrad<DT>* node,
                     Graph* graph, DefaultDevice* device) {
    auto& sb = node->sub_expression(1).template as<DT>();

    // Switch based on runtime rank.
    LLGTM_RANK_KERNEL_SWITCH_2(sb.dimensions().rank(),
                               KernelR, node, graph, device);
  }

 private:
  // Store data for "Softmax Sparse Cross-Entropy Gradient" in the result
  // tensor This is a separate function, because different code is required
  // based on whether this is a rank 1 or rank 2 (with batching) operation.
  static void sparse_cross_entropy_grad_store_data(
      typename ETypes<DT, 2>::Tensor* r,
      const typename ETypes<DT, 2>::Dimensions& dims,
      const typename ETypes<DT, 2>::Tensor& softmax,
      int* labels_data, DefaultDevice* /*device*/) {
    // (d cross-entropy(labels, softmax(logits)) / d logits)[i] =
    //     softmax(logits)[i] - labels[i]  (for dense version)
    typename ETypes<DT, 2>::Tensor& result = *r;
    result = softmax;
    for (int i = 0; i < dims[0]; ++i) {
      DT& value = result(i, labels_data[i]);
      value = value - static_cast<DT>(1);
    }
  }

  static void sparse_cross_entropy_grad_store_data(
      typename ETypes<DT, 1>::Tensor* r,
      const typename ETypes<DT, 1>::Dimensions dims,
      const typename ETypes<DT, 1>::Tensor softmax,
      int* labels_data, DefaultDevice* /*device*/) {
    typename ETypes<DT, 1>::Tensor& result = *r;
    result = softmax;
    DT& value = result(labels_data[0]);
    value = value - static_cast<DT>(1);
  }

  template<int R>
  static void KernelR(nodes::SoftmaxSparseCrossEntropyGrad<DT>* node,
                      Graph* /*graph*/, DefaultDevice* device) {
    static_assert(R <= 2, "Cross entropy probabilities rank must be <= 2");
    DCHECK(node->has_result());

    auto& s_labels = node->sub_expression(0).template as<int>();
    auto& s_softmax = node->sub_expression(1).template as<DT>();

    // Dimensions of input tensors.
    typename ETypes<DT, R>::Dimensions dims_s;
    initialize_dims<DT, R>(&dims_s, s_softmax.dimensions());

    typename ETypes<DT, R>::Tensor softmax(s_softmax.result_data(), dims_s);
    typename ETypes<DT, R>::Tensor r(node->result_data(), dims_s);

    // Gather probabilities specified by s_labels.
    sparse_cross_entropy_grad_store_data(&r, dims_s, softmax,
                                         s_labels.result_data(), device);
  }
};

#ifdef GOOGLE_CUDA
// TODO(matthiasspringer): It might be better to use `batch_size` many threads and
// add a loop inside the kernel.
template<typename DT>
__global__ void SoftmaxSparseCrossEntropyGradCUDAKernel(
    DT* result, const int* labels, const DT* softmax,
    Dimensions::IndexType result_size,
    Dimensions::IndexType batch_size,
    Dimensions::IndexType num_labels) {
  auto tid = blockDim.x*blockIdx.x + threadIdx.x;
  auto num_threads = blockDim.x*gridDim.x;

  // See comment of cuda_block_size for work distribution.
  for (int i = 0; i < result_size; i += num_threads) {
    if (tid + i < result_size) {
      auto batch_index = (tid + i) / num_labels;
      auto label_index = (tid + i) % num_labels;
      if (label_index == labels[batch_index]) {
        result[tid + i] = softmax[tid + i] - static_cast<DT>(1);
      } else {
        result[tid + i] = softmax[tid + i];
      }
    }
  }
}

template<typename DT>
class EigenKernel<GpuDevice, nodes::SoftmaxSparseCrossEntropyGrad<DT>> {
 public:
  static void Kernel(nodes::SoftmaxSparseCrossEntropyGrad<DT>* node,
                     Graph* graph, GpuDevice* device) {
    DCHECK(node->has_result());

    auto& s_labels = node->sub_expression(0).template as<int>();
    auto& s_softmax = node->sub_expression(1).template as<DT>();
    CHECK_LE(s_softmax.dimensions().rank(), 2);

    auto result_size = node->dimensions().num_elements();

    SoftmaxSparseCrossEntropyGradCUDAKernel<DT><<<cuda_grid_size(result_size),
                                                  cuda_block_size(result_size),
                                                  0, device->stream()>>>(
        node->result_data(), s_labels.result_data(), s_softmax.result_data(),
        result_size, s_softmax.dimension(0), s_softmax.dimension(1));
    device->synchronize();
  }
};
#endif

#undef LLGTM_OPTIONAL_VA_ARGS
#undef LLGTM_RANK_KERNEL_CASE
#undef LLGTM_RANK_KERNEL_SWITCH
#undef LLGTM_RANK_KERNEL_SWITCH_DEFAULT
#undef LLGTM_RANK_KERNEL_SWITCH_2

}  // namespace


GraphImplementation* EigenEvaluator::NewGraphImpl() {
  // Graph will take ownership.
  return new EigenGraphImplementation(default_device_, gpu_device_);
}


void EigenEvaluator::InitializeDevices() {
    default_device_ = new Eigen::DefaultDevice();
#ifdef GOOGLE_CUDA
    cuda_stream_device_ = new Eigen::CudaStreamDevice();
    gpu_device_ = new Eigen::GpuDevice(cuda_stream_device_);
#endif
}


EigenEvaluator::~EigenEvaluator() {
  delete default_device_;
#ifdef GOOGLE_CUDA
  delete gpu_device_;
  delete cuda_stream_device_;
#endif
}


void* EigenEvaluator::AllocateDeviceMemory(size_t bytes, DeviceID device) {
  switch (device) {
    case kDeviceIDCPU:
      return default_device_->allocate(bytes);
    case kDeviceIDGPU:
#ifdef GOOGLE_CUDA
      return gpu_device_->allocate(bytes);
#else
      LOG(FATAL) << "LLGTM built without CUDA support. Use --config=cuda.";
#endif  // GOOGLE_CUDA
    default:
      LOG(FATAL) << "Device not supported: " << device_name(device) << ".";
  }
}


void EigenEvaluator::FreeDeviceMemory(void* ptr, DeviceID device) {
  switch (device) {
    case kDeviceIDCPU:
      return default_device_->deallocate(ptr);
    case kDeviceIDGPU:
#ifdef GOOGLE_CUDA
      return gpu_device_->deallocate(ptr);
#else
      LOG(FATAL) << "LLGTM built without CUDA support. Use --config=cuda.";
#endif  // GOOGLE_CUDA
    default:
      LOG(FATAL) << "Device not supported: " << device_name(device) << ".";
  }
}


void EigenEvaluator::MemcpyHostToDevice(void* destination, void* source,
                                        size_t bytes, DeviceID device) {
  switch (device) {
    case kDeviceIDCPU:
      memcpy(destination, source, bytes);
      break;
    case kDeviceIDGPU:
#ifdef GOOGLE_CUDA
      gpu_device_->memcpyHostToDevice(destination, source, bytes);
      gpu_device_->synchronize();
#else
      LOG(FATAL) << "LLGTM built without CUDA support. Use --config=cuda.";
#endif  // GOOGLE_CUDA
      break;
    default:
      LOG(FATAL) << "Device not supported: " << device_name(device) << ".";
  }
}


template<typename DT>
class EigenKernelDeviceSelector<nodes::CopyToDevice<DT>> {
 public:
  static void Kernel(nodes::CopyToDevice<DT>* node, Graph* graph) {
    const auto& sa = node->sub_expression(0).template as<DT>();

    if (node->device() == sa.device()) {
      // Do not allocate any memory.
      node->set_result_data(const_cast<DT*>(sa.result_data()));
      return;
    }

    // Allocate result memory.
    auto* graph_impl = reinterpret_cast<EigenGraphImplementation*>(
        graph->graph_implementation());
    // Avoid virtual method call since we know the exact type.
    graph_impl->EigenGraphImplementation::AllocateResultData(node);

#ifdef GOOGLE_CUDA
    auto* evaluator = reinterpret_cast<EigenEvaluator*>(graph->evaluator());
    // The template Device argument is the device onto which data should
    // be copied, but the transfer must always be initiated from the GPU
    // device (assuming no other devices than CPU or GPU).
    auto* gpu_device = evaluator->gpu_device();

    // TODO(matthiasspringer): This code can handle only CPU and GPU.
    if (node->device() == kDeviceIDCPU) {
      // Copy GPU device -> host.
      CHECK(sa.device() == kDeviceIDGPU);
      gpu_device->memcpyDeviceToHost(node->result_data(), sa.result_data(),
                                     sa.tensor_type().memory_size());
      gpu_device->synchronize();
    } else if (node->device() == kDeviceIDGPU) {
      // Copy host -> GPU device.
      CHECK(sa.device() == kDeviceIDCPU);
      gpu_device->memcpyHostToDevice(node->result_data(), sa.result_data(),
                                     sa.tensor_type().memory_size());
      gpu_device->synchronize();
    } else {
      LOG(FATAL) << "Device " << static_cast<int>(node->device())
                 << " not supported.";
    }
#else
    LOG(FATAL) << "LLGTM built without CUDA support. Use --config=cuda.";
#endif  // GOOGLE_CUDA
  }
};


template<typename NodeType>
void EigenKernelDeviceSelector<NodeType>::Kernel(NodeType* node,
                                                 Graph* graph) {
  auto* evaluator = reinterpret_cast<EigenEvaluator*>(graph->evaluator());
  auto* graph_impl = reinterpret_cast<EigenGraphImplementation*>(
      graph->graph_implementation());
  // Avoid virtual method call since we know the exact type.
  graph_impl->EigenGraphImplementation::AllocateResultData(node);

  // TODO(matthiasspringer): Implement multi GPU support. For now, CPU and GPU have
  // hard-coded IDs.
  switch (node->device()) {
    case kDeviceIDCPU: {
      auto* dev_default = evaluator->default_device();
      EigenKernel<Eigen::DefaultDevice, NodeType>::Kernel(node, graph,
                                                          dev_default);
      break;
    }
    case kDeviceIDGPU: {
#ifdef GOOGLE_CUDA
#ifndef NDEBUG
#error "Cannot run Eigen with CUDA support in debug mode. Run in opt mode.";
#endif
      auto* dev_gpu = evaluator->gpu_device();
      EigenKernel<Eigen::GpuDevice, NodeType>::Kernel(node, graph, dev_gpu);
#else
      LOG(FATAL) << "LLGTM built without CUDA support. Use --config=cuda.";
#endif  // GOOGLE_CUDA
      break;
    }
    case kDeviceIDUnspecified:
      // This is a programming error and should never happen.
      LOG(FATAL) << "Found unspecified device ID in Eigen backend.";
    default:
      LOG(FATAL) << "Device not supported.";
  }
}

// This macro explicitly instantiates an EigenKernel for a given node class for
// all data types (DT) that are supported by LLGTM. This ensures that the
// resulting object file contains all possible specializations of EigenKernels,
// i.e., every node and every data type. Clients include eigen_evaluator.h,
// which contains declarations of all EigenKernels, and link against the object
// file generated for this file.
#define LLGTM_NODE_DEFINITION(NODETYPE) \
  template class EigenKernelDeviceSelector<NODETYPE<float>>; \
  template class EigenKernelDeviceSelector<NODETYPE<int32_t>>;
#define LLGTM_NODE_DEFINITION_FP(NODETYPE) \
  template class EigenKernelDeviceSelector<NODETYPE<float>>;
#include "tensorflow_fold/llgtm/backend/llgtm_nodes.inc"
#undef LLGTM_NODE_DEFINITION

template class EigenKernelDeviceSelector<nodes::GetOutput>;

}  // namespace llgtm
