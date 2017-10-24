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

#include "tensorflow_fold/llgtm/tensor_opcodes.h"

namespace llgtm  {

#define LLGTM_OPCODE_NAME_CASE(OP) case kOp ## OP: return #OP;

absl::string_view opcode_name(TensorOpcode op) {
  switch (op) {
    LLGTM_OPCODE_NAME_CASE(Tuple);
    LLGTM_OPCODE_NAME_CASE(GetOutput);
    LLGTM_OPCODE_NAME_CASE(CopyToDevice);

    LLGTM_OPCODE_NAME_CASE(Value);
    LLGTM_OPCODE_NAME_CASE(Variable);
    LLGTM_OPCODE_NAME_CASE(AssignAdd);
    LLGTM_OPCODE_NAME_CASE(Zeros);
    LLGTM_OPCODE_NAME_CASE(ConstantFromFunction);
    LLGTM_OPCODE_NAME_CASE(ConstantFromScalar);
    LLGTM_OPCODE_NAME_CASE(UniformRandom);
    LLGTM_OPCODE_NAME_CASE(NormalRandom);

    LLGTM_OPCODE_NAME_CASE(Broadcast);
    LLGTM_OPCODE_NAME_CASE(ReduceSum);
    LLGTM_OPCODE_NAME_CASE(Transpose);
    LLGTM_OPCODE_NAME_CASE(Reshape);
    LLGTM_OPCODE_NAME_CASE(Concat);
    LLGTM_OPCODE_NAME_CASE(Split);
    LLGTM_OPCODE_NAME_CASE(Gather);
    LLGTM_OPCODE_NAME_CASE(Scatter);

    LLGTM_OPCODE_NAME_CASE(Negative);
    LLGTM_OPCODE_NAME_CASE(Reciprocal);

    LLGTM_OPCODE_NAME_CASE(Add);
    LLGTM_OPCODE_NAME_CASE(Multiply);

    LLGTM_OPCODE_NAME_CASE(Matmul);

    LLGTM_OPCODE_NAME_CASE(Relu);
    LLGTM_OPCODE_NAME_CASE(ReluGrad);
    LLGTM_OPCODE_NAME_CASE(Sigmoid);
    LLGTM_OPCODE_NAME_CASE(Tanh);
    LLGTM_OPCODE_NAME_CASE(Softmax);
    LLGTM_OPCODE_NAME_CASE(SoftmaxCrossEntropy);
    LLGTM_OPCODE_NAME_CASE(SoftmaxSparseCrossEntropy);
    LLGTM_OPCODE_NAME_CASE(SoftmaxSparseCrossEntropyGrad);

    case kMaximumTensorOpcode: return "Invalid";
  }
}
#undef LLGTM_OPCODE_NAME_CASE


#define LLGTM_DTYPE_NAME_CASE(TYPE) case kDT ## TYPE: return #TYPE;

absl::string_view dtype_name(TensorDataType type) {
  switch (type) {
    LLGTM_DTYPE_NAME_CASE(void);
    LLGTM_DTYPE_NAME_CASE(bool);
    LLGTM_DTYPE_NAME_CASE(int8);
    LLGTM_DTYPE_NAME_CASE(int16);
    LLGTM_DTYPE_NAME_CASE(int32);
    LLGTM_DTYPE_NAME_CASE(int64);
    LLGTM_DTYPE_NAME_CASE(float32);
    LLGTM_DTYPE_NAME_CASE(float64);

    case kMaximumTensorDataType: return "Invalid";
  }
}

#undef LLGTM_DTYPE_NAME_CASE

}  // end namespace llgtm
