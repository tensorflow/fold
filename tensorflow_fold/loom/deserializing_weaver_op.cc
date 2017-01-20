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

#include "tensorflow_fold/loom/weaver.h"
#include "tensorflow_fold/loom/weaver_op_base.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace fold {

REGISTER_WEAVER_OP("DeserializingWeaver")
  .Input("weaver_messages: string");

// A Weaver op which:
// 1. Reads one or more serialized WeaverMessages from `weaver_messages`, its
//   input tensor.
// 2. Merges them if there are more than one, and
// 3. Creates output tensors that can drive the Loom using the resulting Weaver.
//
// (Item 3 is handled by WeaverOpBase.)
//
// Note: the reason merges are supported in this op to allow the user to
// pre-compute many WeaverMessages (for example, one per element of the training
// set) and then group them together into random mini-batches at run-time.
//
// A second reason merges are supported is that for large input examples, and
// large batch sizes, merges done in advance of `DeserializingWeaverOp` could
// push the resulting `WeaverMessage` over the protocol buffer size limit.
class DeserializingWeaverOp : public WeaverOpBase {
 public:
  explicit DeserializingWeaverOp(tensorflow::OpKernelConstruction *c)
      : WeaverOpBase(c) {}

  tensorflow::Status Weave(
      tensorflow::OpKernelContext *c, Weaver* weaver) override {
    auto weaver_messages = c->input(0).flat<string>();
    if (weaver_messages.size() < 1) {
      return tensorflow::errors::InvalidArgument(
          "weaver_messages must contain at least one value.");
    }
    if (!weaver->Deserialize(weaver_messages(0))) {
      return tensorflow::errors::Internal(
          "Failed to deserialize WeaverMessage: ", weaver->error_string());
    }

    // Note: If necessary, this loop could be sped up by merging the messages in
    // a multi-threaded way instead of in sequence.
    for (int64 i = 1; i < weaver_messages.size(); ++i) {
      if (!weaver->MergeFromSerialized(weaver_messages(i))) {
        return tensorflow::errors::Internal(
            "Failed to merge WeaverMessage", i, ":", weaver->error_string());
      }
    }
    return tensorflow::Status::OK();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("DeserializingWeaver").Device(tensorflow::DEVICE_CPU),
    DeserializingWeaverOp);

}  // namespace fold
}  // namespace tensorflow
