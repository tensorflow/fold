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

// Implements the dimensions of a tensor.

#ifndef TENSORFLOW_FOLD_LLGTM_DIMENSIONS_H_
#define TENSORFLOW_FOLD_LLGTM_DIMENSIONS_H_

#include <array>
#include <iostream>

#include "tensorflow_fold/llgtm/platform/platform.h"
#include "tensorflow_fold/llgtm/tensor_opcodes.h"


namespace llgtm  {

// Dimensions for a tensor of the given rank.
class Dimensions {
 public:
  // TODO(delesley):  Support tensors with rank > 4.
  // Currently supported maximum tensor rank, sufficent for most neural nets.
  static const int kMaxRank = 4;
  typedef int64 IndexType;

  Dimensions(const Dimensions&) = default;

  template<typename... IndexType>
  explicit Dimensions(IndexType... dimensions)
      : rank_(sizeof...(dimensions)), dtype_(kDTvoid),
        dimensions_({{ dimensions... }}) {
    static_assert(sizeof...(dimensions) <= kMaxRank,
                  "Maximum number of dimensions exceeded.");
  }

  Dimensions(std::initializer_list<IndexType> indices)
      : rank_(indices.size()), dtype_(kDTvoid) {
    CHECK_LE(indices.size(), kMaxRank);

    int index = 0;
    for (const auto& el : indices) {
      dimensions_[index++] = el;
    }
  }

  template<std::size_t Rank>
  explicit Dimensions(const std::array<IndexType, Rank>& dimensions)
    : rank_(Rank), dimensions_(dimensions)
  { static_assert(Rank < kMaxRank, "Rank is too large."); }

  int rank() const { return rank_; }

  template<int Rank>
  const std::array<IndexType, Rank>& values() const {
    DCHECK_EQ(rank_, Rank);
    return *reinterpret_cast<const std::array<IndexType, Rank>*>(&dimensions_);
  }

  IndexType operator[](int i) const {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, rank_);
    return dimensions_[i];
  }

  const IndexType* as_ptr() const { return &dimensions_[0]; }

  void set_dimension(int i, IndexType v) {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, rank_);
    dimensions_[i] = v;
  }

  bool operator==(const Dimensions& d) const {
    if (rank_ != d.rank_) return false;
    for (int i = 0; i < rank_; ++i) {
      if (dimensions_[i] != d.dimensions_[i]) return false;
    }
    return true;
  }

  bool operator!=(const Dimensions& d) const {
    return !(*this == d);
  }

  int64 num_elements() const {
    int64 num_elems = 1;
    for (int i = 0; i < rank_; ++i) { num_elems *= dimensions_[i]; }
    return num_elems;
  }

  // Get the offset into the underlying data, for a given element.
  int64 offset(IndexType i) const {
    DCHECK_EQ(rank_, 1);
    DCHECK(i >= 0 && i < dimensions_[0]);
    return i;
  }
  int64 offset(IndexType i, IndexType j) const {
    DCHECK_EQ(rank_, 2);
    DCHECK(i >= 0 && j >= 0);
    DCHECK(i < dimensions_[0] && j < dimensions_[1]);
    return i*dimensions_[1] + j;
  }
  int64 offset(IndexType i, IndexType j, IndexType k) const {
    DCHECK_EQ(rank_, 3);
    DCHECK(i >= 0 && j >= 0 && k >= 0);
    DCHECK(i < dimensions_[0] && j < dimensions_[1] && k < dimensions_[2]);
    return dimensions_[2]*(i*dimensions_[1] + j) + k;
  }

  string str() const {
    std::ostringstream outstr;
    outstr << "(";
    const char* sep = "";
    for (int i = 0; i < rank_; ++i) {
      outstr << sep << dimensions_[i];
      sep = ", ";
    }
    outstr << ")";
    return outstr.str();
  }

 private:
  friend class TensorType;

  // Access to as_array().
  friend class TfGraphEvaluator;

  Dimensions(int rank, const std::array<IndexType, kMaxRank>& dims)
    : rank_(rank), dtype_(kDTvoid), dimensions_(dims) {}

  // Constructor used by TensorType.
  Dimensions(TensorDataType dt, const Dimensions& dims)
      : rank_(dims.rank_), dtype_(dt), dimensions_(dims.dimensions_) {}

  template<typename DT>
  std::array<DT, kMaxRank> as_array() const {
    std::array<DT, kMaxRank> result;
    for (int i = 0; i < kMaxRank; ++i) {
      result[i] = static_cast<DT>(dimensions_[i]);
    }
    return result;
  }

  const uint8_t rank_;  // Rank of the tensor. (8 bits)

  // Dimensions are usually part of a TensorType, so the dtype_ field
  // is stored in Dimensions for compactness.
  const TensorDataType dtype_;  // Data type of tensor. (8 bits)

  std::array<IndexType, kMaxRank> dimensions_;  // Dimensions of the tensor.
};

// Dimensions is also used as a datatype for a list of up to kMaxRank many
// indices. E.g., the indices of a transposition. In such a case, we use
// DimensionIndices instead Dimensions to make clear that such an object does
// not represent the dimensions of a tensor.
using DimensionIndices = Dimensions;


// A TensorType holds both the data type and dimensions of a Tensor.
class TensorType {
 public:
  static const size_t kResultAlignment = 16;  // Must be a power of two.

  TensorType() = default;
  TensorType(const TensorType&) = default;

  TensorType(TensorDataType dt, const Dimensions& dims)
      : dimensions_(dt, dims) {}

  TensorType& operator=(const TensorType&) = delete;

  TensorDataType dtype() const { return dimensions_.dtype_; }
  const Dimensions& dimensions() const { return dimensions_; }
  Dimensions::IndexType dimension(int i) const { return dimensions_[i]; }

  int rank() const { return dimensions_.rank(); }
  int64 num_elements() const { return dimensions_.num_elements(); }

  // Round sz upwards so that it is a multiple of alignment.
  static size_t aligned_size(size_t sz, size_t alignment) {
    size_t remainder = sz & (alignment - 1);
    if (remainder > 0) sz += alignment - remainder;
    return sz;
  }

  size_t memory_size() const {
    return num_elements() * sizeof_dtype(dtype());
  }

  size_t aligned_memory_size() const {
    return aligned_size(memory_size(), kResultAlignment);
  }

  bool operator==(const TensorType& t) const {
    if (dtype() != t.dtype()) return false;
    return dimensions_ == t.dimensions_;
  }

  bool operator!=(const TensorType& t) const {
    return !(*this == t);
  }

  // Returns true if ttype is equal to ttype, except for the batch size.
  // The batch size must always be the first dimension.
  bool matches(const TensorType& t) const {
    if (dtype() != t.dtype()) return false;
    if (rank() != t.rank()) return false;
    for (int i = 1, r = rank(); i < r; ++i) {   // ignore first dimension
      if (dimensions_[i] != t.dimensions_[i]) return false;
    }
    return true;
  }

  // Check if data is aligned properly.
  static inline bool is_aligned(uintptr_t data_ptr) {
    return ((uintptr_t) data_ptr & (kResultAlignment - 1)) == 0;
  }

  // Return offset into a list of tensors, stored contiguously in memory.
  static inline size_t memory_offset(int i, const TensorType* types);

  // Find total size of a list of tensors, stored contiguously in memory.
  static size_t total_memory_size(int num_tensors,
                                  const TensorType* types) {
    return memory_offset(num_tensors, types);
  }

  inline string str() const;

 private:
  Dimensions dimensions_;
};


// Print Dimensions on a stream.
inline std::ostream& operator<<(std::ostream& o, const Dimensions& dims) {
  return o << dims.str();
}


// Print TensorType on a stream.
inline std::ostream& operator<<(std::ostream& o, const TensorType& ttype) {
  return o << ttype.str();
}


string TensorType::str() const {
  std::ostringstream outstr;
  outstr << "(" << dtype_name(dtype()) << ", " << dimensions().str() << ")";
  return outstr.str();
}


// Return the offset into result_data for the i^th output.
// Results are allocated sequentially in memory.
size_t TensorType::memory_offset(int i, const TensorType* types) {
  if (types == nullptr) return 0;
  size_t offset = 0;
  for (int k = 0; k < i; ++k) {
    offset += types[k].aligned_memory_size();
  }
  return offset;
}


}  // namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_DIMENSIONS_H_
