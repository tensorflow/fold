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

#ifndef TENSORFLOW_FOLD_LOOM_WEAVER_OP_BASE_H_
#define TENSORFLOW_FOLD_LOOM_WEAVER_OP_BASE_H_

#include "tensorflow_fold/loom/weaver.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace fold {

/// This macro is a specialization of REGISTER_OP intended for ops subclassed
/// from WeaverOpBase.  Outputs are preset, but the user may add
/// additional attributes and inputs.
#define REGISTER_WEAVER_OP(_name_)                                           \
  REGISTER_OP(_name_)                                                        \
      .Attr("metadata: string")                                              \
      .Attr("constant_types: list(type) >= 0")                               \
      .Attr("num_type_shapes: int >= 0")                                     \
      .Output("out_0_arg_wiring_concat: int32")                              \
      .Output("out_1_arg_wiring_slice_starts: int32")                        \
      .Output("out_2_arg_wiring_slice_sizes: int32")                         \
      .Output("out_3_output_wiring: num_type_shapes * int32")                \
      .Output("out_4_constants: constant_types")                             \
      .SetShapeFn([](tensorflow::shape_inference::InferenceContext *c) {     \
        int32 num_type_shapes;                                               \
        TF_RETURN_IF_ERROR(c->GetAttr("num_type_shapes", &num_type_shapes)); \
        int32 i = 0;                                                         \
        for (i = 0; i < num_type_shapes + 3; ++i) {                          \
          c->set_output(i, c->Vector(c->UnknownDim()));                      \
        }                                                                    \
        for (int32 j = 0; j < num_type_shapes; ++j) {                        \
          c->set_output(i++, c->UnknownShape());                             \
        }                                                                    \
                                                                             \
        return tensorflow::Status::OK();                                     \
      })

/// `WeaverOpBase` is a base class for writing TensorFlow ops kernels that
/// schedule ops for Loom.
///
/// Operations created as subclasses of `WeaverOpBase` should be registered with
/// the `REGISTER_WEAVER_OP` macro.  For example, `DeserializingWeaverOp` is
/// registered using:
///
/*! \verbatim
```c++
REGISTER_WEAVER_OP("DeserializingWeaver").Input("weaver_messages: string");
```
\endverbatim
*/
///
///
/// And
///
/*! \verbatim
```c++
REGISTER_KERNEL_BUILDER(
    Name("DeserializingWeaver").Device(tensorflow::DEVICE_CPU),
    DeserializingWeaverOp);
```
\endverbatim
*/

class WeaverOpBase : public tensorflow::OpKernel {
 public:
  /// Reads the `metadata`, `constant_types`, and `num_types_shapes`
  /// attributes and makes sure they're consistent.  Dies if they're
  /// not.
  explicit WeaverOpBase(tensorflow::OpKernelConstruction* c);

  /// Weave is a virtual method, to be subclassed. Weave's responsibility is to
  /// read the ops inputs and use the weaver to schedule LoomOps to be executed
  /// on the loom.  `Weave` should not call `Weaver::Finalize`.
  virtual tensorflow::Status Weave(
      tensorflow::OpKernelContext *c, Weaver *weaver) = 0;

  /// Dispatches to `Weave` to build a `Weaver`, which is then used to build
  /// the wiring diagram and constant tensors that the loom needs.
  void Compute(tensorflow::OpKernelContext *c) override;

 protected:
  /// Returns the metadata for the Loom as a serialized string.
  const string &serialized_metadata() const {
    return metadata_str_;
  }

  /// Returns the metadata for the Loom as a message.
  const LoomMetadata &metadata() const {
    return metadata_;
  }

  /// Finds the ID of an op inside of the LoomMetadata.  To be invoked in the
  /// constructor of subclasses.
  tensorflow::Status FindOp(const string &op_name, tensor_idx_t *op_idx) const;

  /// Finds the TypeShape index and named_tensor index of a named tensor
  /// inside of the LoomMetadata.  To be invoked in the constructor of
  /// subclasses.
  tensorflow::Status FindNamedTensor(
      const string &tensor_name, tensor_idx_t *ts_idx,
      tensor_idx_t *named_tensor_idx) const;

  /// Finds the ID of an TypeShape inside of the / LoomMetadata.  To be invoked
  /// in the constructor of subclasses.
  tensorflow::Status FindTypeShape(
      const string &tag, tensor_idx_t *ts_idx) const;

 private:
  string metadata_str_;
  std::vector<tensorflow::DataType> constant_types_;
  tensor_idx_t num_type_shapes_;
  LoomMetadata metadata_;
};

}  // namespace fold
}  // namespace tensorflow

#endif  // TENSORFLOW_FOLD_LOOM_WEAVER_OP_BASE_H_
