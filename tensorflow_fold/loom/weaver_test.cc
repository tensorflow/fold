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

#include <stdio.h>

#include <gtest/gtest.h>

#include "tensorflow_fold/loom/loom.pb.h"
#include "tensorflow_fold/loom/weaver.h"
#include "tensorflow/core/framework/types.pb.h"

namespace tensorflow {
namespace fold {

namespace {

Tensor FloatVectorTensor(const std::vector<float> &vec) {
  TensorShape shape;
  shape.AddDim(vec.size());
  Tensor tensor(DT_FLOAT, shape);
  auto flat = tensor.template flat<float>();
  for (size_t i = 0; i < vec.size(); ++i) {
    flat(i) = vec[i];
  }
  return tensor;
}

}  // namespace

TEST(WeaverTest, BuildMetadata) {
  LoomMetadata metadata;
  metadata.set_max_depth(-1);

  // TypeShape #0 : float vector of length 3.
  auto *type_shape_metadata = metadata.add_type_shape_metadata();
  type_shape_metadata->set_dtype(DT_FLOAT);
  type_shape_metadata->add_shape(3);
  type_shape_metadata->set_name("3vec");
  type_shape_metadata->set_is_batch_input(false);

  // Op #0 : mandatory pass through op for TypeShape #0
  auto *passthrough0_op_metadata = metadata.add_op_metadata();
  passthrough0_op_metadata->set_name("pass-through 0");
  passthrough0_op_metadata->add_input_ts_idx(0);
  passthrough0_op_metadata->add_output_ts_idx(0);

  // Op #1: binary op 'plus'.
  auto *plus_op_metadata = metadata.add_op_metadata();
  plus_op_metadata->set_name("plus");
  plus_op_metadata->add_input_ts_idx(0);
  plus_op_metadata->add_input_ts_idx(0);
  plus_op_metadata->add_output_ts_idx(0);

  string error_string;
  CHECK(VerifyLoomMetadata(metadata, &error_string)) << error_string;

  string metadata_str;
  metadata.SerializeToString(&metadata_str);
  Weaver w(metadata_str);

  std::vector<tensor_idx_t> constant_results;
  for (size_t i = 0; i < 6; ++i) {
    constant_results.push_back(w.MakeConstant(0, FloatVectorTensor(
        {1.0f * i, 2.0f * i, 3.0f * i})));
  }

  std::vector<tensor_idx_t> pair_results;
  for (size_t i = 0; i < 3; ++i) {
    pair_results.push_back(w.CallOp(1, {
      constant_results[2 * i], constant_results[2 * i + 1]})[0]);
  }

  tensor_idx_t first_four_result = w.CallOp(1, {
    pair_results[0], pair_results[1]})[0];
  tensor_idx_t total_result = w.CallOp(
      1, {first_four_result, pair_results[2]})[0];
  w.AddOutput(total_result);
  w.Finalize();

  Tensor constants_tensor = w.BatchConstantValues(0);
  auto constants = constants_tensor.flat_inner_dims<float>();
  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(constants(i, j),  i * (j + 1));
    }
  }

  // Level 1 is where the pair_results get constructed.
  EXPECT_TRUE(w.GetWiring(1, 0, 0).empty());  // No passthroughs
  const std::vector<tensor_idx_t> &level1_plus_arg0 = w.GetWiring(1, 1, 0);
  const std::vector<tensor_idx_t> &level1_plus_arg1 = w.GetWiring(1, 1, 1);

  EXPECT_EQ(3, level1_plus_arg0.size());
  EXPECT_EQ(3, level1_plus_arg1.size());
  EXPECT_EQ(0, level1_plus_arg0[0]);
  EXPECT_EQ(1, level1_plus_arg1[0]);
  EXPECT_EQ(2, level1_plus_arg0[1]);
  EXPECT_EQ(3, level1_plus_arg1[1]);
  EXPECT_EQ(4, level1_plus_arg0[2]);
  EXPECT_EQ(5, level1_plus_arg1[2]);

  // Level 2 has one call to plus (for pair_results[0] and pair_results[1]), and
  // a pass through (for pair_results[2])
  const std::vector<tensor_idx_t> &level2_passthrough_arg0 = w.GetWiring(
      2, 0, 0);
  const std::vector<tensor_idx_t> &level2_plus_arg0 = w.GetWiring(2, 1, 0);
  const std::vector<tensor_idx_t> &level2_plus_arg1 = w.GetWiring(2, 1, 1);
  EXPECT_EQ(1, level2_passthrough_arg0.size());
  EXPECT_EQ(1, level2_plus_arg0.size());
  EXPECT_EQ(1, level2_plus_arg1.size());
  EXPECT_EQ(0, level2_plus_arg0[0]);
  EXPECT_EQ(1, level2_plus_arg1[0]);
  EXPECT_EQ(2, level2_passthrough_arg0[0]);

  // Level 3 has one call to plus.  Its arguments are in reverse order because
  // the pass-through has a lower op-index than plus, so the result from
  // pair_results[2] ends up in position 0, while first_four_result ends up in
  // position 1.
  EXPECT_TRUE(w.GetWiring(3, 0, 0).empty());  // No passthroughs
  const std::vector<tensor_idx_t> &level3_plus_arg0 = w.GetWiring(3, 1, 0);
  const std::vector<tensor_idx_t> &level3_plus_arg1 = w.GetWiring(3, 1, 1);
  EXPECT_EQ(1, level3_plus_arg0.size());
  EXPECT_EQ(1, level3_plus_arg1.size());
  EXPECT_EQ(1, level3_plus_arg0[0]);
  EXPECT_EQ(0, level3_plus_arg1[0]);

  // Output Wiring: straight forward as only one output has been marked and only
  // one value exists at level 3.
  const std::vector<tensor_idx_t> &output_wiring = w.GetOutputWiring(0);
  EXPECT_EQ(1, output_wiring.size());
  EXPECT_EQ(0, output_wiring[0]);
}

}  // namespace fold
}  // namespace tensorflow
