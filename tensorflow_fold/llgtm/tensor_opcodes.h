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

// Enumerations and helper functions that are used in tensor.h.

#ifndef TENSORFLOW_FOLD_LLGTM_TENSOR_OPCODES_H_
#define TENSORFLOW_FOLD_LLGTM_TENSOR_OPCODES_H_

#include "tensorflow_fold/llgtm/platform/platform.h"
#include "absl/strings/string_view.h"

namespace llgtm  {

// TODO(delesley): Consider using tensorflow/core/framework/types.proto
// For now, we are trying to limit dependencies on tensorflow.
enum TensorDataType : uint8_t {
  kDTvoid,
  kDTbool,
  kDTint8,
  kDTint16,
  kDTint32,
  kDTint64,
  // TODO(delesley): decide if we want to support unsigned ints.
  kDTfloat32,
  kDTfloat64,
  kMaximumTensorDataType
};


// Return the size of the given dtype.
inline int sizeof_dtype(TensorDataType dtype) {
  switch (dtype) {
    case kDTvoid: return 0;
    case kDTbool: return sizeof(bool);
    case kDTint8: return sizeof(int8_t);
    case kDTint16: return sizeof(int16_t);
    case kDTint32: return sizeof(int32_t);
    case kDTint64: return sizeof(int64_t);
    case kDTfloat32: return sizeof(float);
    case kDTfloat64: return sizeof(double);
    case kMaximumTensorDataType:
      break;
  }
  LOG(FATAL) << "Invalid DType.";
}


// Template function converts a C++ type to TensorDataType at compile time.
template<typename DT> struct CppTypeToDType;

#define TYPE2DTYPEDEF(Type, DTypeVal) \
template<> struct CppTypeToDType<Type> { \
  static const TensorDataType dtype = DTypeVal; \
};

TYPE2DTYPEDEF(bool,     kDTbool);
TYPE2DTYPEDEF(int8_t,   kDTint8);
TYPE2DTYPEDEF(int16_t,  kDTint16);
TYPE2DTYPEDEF(int32_t,  kDTint32);
TYPE2DTYPEDEF(int64_t,  kDTint64);
TYPE2DTYPEDEF(float,    kDTfloat32);
TYPE2DTYPEDEF(double,   kDTfloat64);

#undef TYPE2DTYPEDEF


enum TensorOpcode : int8_t {
  kOpGetOutput,             // Get the nth output.
  kOpTuple,                 // Collect a set of gradients into a single node.
  kOpCopyToDevice,          // Copy result data to a different device.

  // Constants.
  kOpValue,                 // A tensor value (fully allocated in memory).
  kOpVariable,              // A reference to a variable.
  kOpAssignAdd,             // Adds a gradient to variable.
  kOpZeros,                 // A tensor full of zeros.
  kOpConstantFromFunction,  // A tensor constant, initialized from a function.
  kOpConstantFromScalar,    // A tensor initialized with a scalar.
  kOpUniformRandom,         // A tensor of random numbers (uniform distr.).
  kOpNormalRandom,          // A tensor of random numbers (normal distr.).

  // Tensor operations.
  kOpBroadcast,
  kOpReduceSum,
  kOpTranspose,
  kOpReshape,
  kOpConcat,
  kOpSplit,
  kOpGather,
  kOpScatter,

  // Element-wise unary arithmetic operations
  kOpNegative,
  kOpReciprocal,

  // Element-wise arithmetic operations.
  kOpAdd,
  kOpMultiply,

  // Matrix operations.
  kOpMatmul,

  // Neural network activations.
  kOpRelu,
  kOpReluGrad,
  kOpSigmoid,
  kOpTanh,
  kOpSoftmax,
  kOpSoftmaxCrossEntropy,
  kOpSoftmaxSparseCrossEntropy,
  kOpSoftmaxSparseCrossEntropyGrad,

  kMaximumTensorOpcode
};

// Return the name of the given opcode.
absl::string_view opcode_name(TensorOpcode op);

// Return the name of the given type.
absl::string_view dtype_name(TensorDataType type);

}  // end namespace llgtm

#endif  // TENSORFLOW_FOLD_LLGTM_TENSOR_OPCODES_H_
