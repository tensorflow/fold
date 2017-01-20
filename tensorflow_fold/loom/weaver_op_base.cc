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

#include "tensorflow_fold/loom/weaver_op_base.h"

#include <vector>

#include "tensorflow_fold/loom/weaver.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {
namespace fold {

using tensorflow::errors::InvalidArgument;

namespace {

Status AddVectorToOutputList(
    const std::vector<tensor_idx_t>& v, tensor_idx_t index,
    OpOutputList *output_list) {
  tensor_idx_t v_len = v.size();
  TensorShape shape({v_len});
  Tensor *output_tensor;
  Status s = output_list->allocate(
      index, shape, &output_tensor);
  if (!s.ok()) return s;
  std::copy(v.begin(), v.end(), output_tensor->flat<tensor_idx_t>().data());
  return s;  // OK
}

Status OutputTensorIdxVector(
    OpKernelContext* c, tensor_idx_t output_idx,
    const std::vector<tensor_idx_t> &v) {
  tensor_idx_t v_len = v.size();
  TensorShape shape({v_len});
  Tensor *output_tensor;
  Status s = c->allocate_output(output_idx, shape, &output_tensor);
  if (!s.ok()) return s;
  std::copy(v.begin(), v.end(), output_tensor->flat<tensor_idx_t>().data());
  return s;  // OK
}

}  // namespace

WeaverOpBase::WeaverOpBase(OpKernelConstruction *c)
    : OpKernel(c) {
  OP_REQUIRES_OK(c, c->GetAttr("metadata", &metadata_str_));
  OP_REQUIRES_OK(c, c->GetAttr("constant_types", &constant_types_));
  OP_REQUIRES_OK(c, c->GetAttr("num_type_shapes", &num_type_shapes_));

  OP_REQUIRES(c, metadata_.ParseFromString(metadata_str_),
              InvalidArgument("Failed to parse metadata for Loom Op."));
  string error_string;
  OP_REQUIRES(c, VerifyLoomMetadata(metadata_, &error_string), InvalidArgument(
      "Couldn't verify loom metadata:", error_string));

  tensor_idx_t metadata_num_type_shapes = metadata_.type_shape_metadata_size();
  OP_REQUIRES(
      c, metadata_num_type_shapes == num_type_shapes_, InvalidArgument(
          "num_type_shapes is wrong.  Got ", num_type_shapes_,
          " but metadata claimed num_type_shapes=",
          metadata_num_type_shapes, "."));
  OP_REQUIRES(
      c, constant_types_.size() == num_type_shapes_, InvalidArgument(
          "metadata has ", num_type_shapes_, " TypeShapes, but ",
          "constant_types has ", constant_types_.size()))
}

void WeaverOpBase::Compute(OpKernelContext *c) {
  Weaver weaver(metadata_str_);
  OP_REQUIRES(c, weaver.error_string().empty(), InvalidArgument(
      "Couldn't initialize weaver from metadata: ", weaver.error_string()));
  OP_REQUIRES_OK(c, Weave(c, &weaver));
  weaver.Finalize();

  // Output the Weaver's wirings:
  std::vector<tensor_idx_t> arg_wiring_concat;
  std::vector<tensor_idx_t> arg_wiring_slice_starts;
  std::vector<tensor_idx_t> arg_wiring_slice_sizes;
  tensor_idx_t max_depth = weaver.MaxDepth();
  tensor_idx_t num_ops = weaver.NumOps();
  std::vector<tensor_idx_t> num_args;
  for (tensor_idx_t op_idx = 0; op_idx < num_ops; ++op_idx) {
    num_args.push_back(weaver.InputTypeShapes(op_idx).size());
  }
  for (tensor_idx_t depth = 1; depth <= max_depth; ++depth) {
    for (tensor_idx_t op_idx = 0; op_idx < num_ops; ++op_idx) {
      for (tensor_idx_t arg_idx = 0; arg_idx < num_args[op_idx]; ++arg_idx) {
        arg_wiring_slice_starts.push_back(arg_wiring_concat.size());
        const std::vector<tensor_idx_t> &wiring = weaver.GetWiring(
            depth, op_idx, arg_idx);
        arg_wiring_slice_sizes.push_back(wiring.size());
        arg_wiring_concat.insert(
            arg_wiring_concat.end(), wiring.begin(), wiring.end());
      }
    }
  }

  OP_REQUIRES_OK(c, OutputTensorIdxVector(c, 0, arg_wiring_concat));
  OP_REQUIRES_OK(c, OutputTensorIdxVector(c, 1, arg_wiring_slice_starts));
  OP_REQUIRES_OK(c, OutputTensorIdxVector(c, 2, arg_wiring_slice_sizes));

  OpOutputList output_wiring_list;
  OP_REQUIRES_OK(c, c->output_list("out_3_output_wiring", &output_wiring_list));
  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
    OP_REQUIRES_OK(c, AddVectorToOutputList(
        weaver.GetOutputWiring(ts_idx),
        ts_idx, &output_wiring_list));
  }

  // Output the constants:
  OpOutputList constants_list;
  OP_REQUIRES_OK(c, c->output_list("out_4_constants", &constants_list));
  for (tensor_idx_t ts_idx = 0; ts_idx < num_type_shapes_; ++ts_idx) {
    constants_list.set(ts_idx, weaver.BatchConstantValues(ts_idx));
  }
}

tensorflow::Status WeaverOpBase::FindOp(
    const string &op_name, tensor_idx_t *op_idx) const {
  for (tensor_idx_t op = 0; op < metadata_.op_metadata_size(); ++op) {
    if (metadata_.op_metadata(op).name() == op_name) {
      *op_idx = op;
      return tensorflow::Status::OK();
    }
  }
  return InvalidArgument(
      "Weaver could not find an op named: ", op_name);
}

tensorflow::Status WeaverOpBase::FindNamedTensor(
    const string &tensor_name, tensor_idx_t *ts_idx,
    tensor_idx_t *named_tensor_idx) const {
  for (tensor_idx_t ts = 0; ts < metadata_.type_shape_metadata_size(); ++ts) {
    auto ts_metadata = metadata_.type_shape_metadata(ts);
    for (tensor_idx_t nt = 0; nt < ts_metadata.tensor_names_size(); ++nt) {
      if (ts_metadata.tensor_names(nt) == tensor_name) {
        *ts_idx = ts;
        *named_tensor_idx = nt;
        return tensorflow::Status::OK();
      }
    }
  }
  return InvalidArgument(
      "Weaver could not find a named tensor named: ", tensor_name);
}

tensorflow::Status WeaverOpBase::FindTypeShape(
    const string &tag, tensor_idx_t *ts_idx) const {
  for (tensor_idx_t ts = 0; ts < metadata_.type_shape_metadata_size(); ++ts) {
    if (metadata_.type_shape_metadata(ts).tag() == tag) {
      *ts_idx = ts;
      return tensorflow::Status::OK();
    }
  }
  return InvalidArgument(
      "Weaver could not find a TypeShape tagged with: ", tag);
}

}  // namespace fold
}  // namespace tensorflow
