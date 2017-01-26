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

#ifndef TENSORFLOW_FOLD_LOOM_WEAVER_H_
#define TENSORFLOW_FOLD_LOOM_WEAVER_H_

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow_fold/loom/loom.pb.h"

#if !defined(SWIG)
#include "tensorflow_fold/loom/platform.h"
#endif

namespace tensorflow {
namespace fold {

// We use int32 to reduce memory pressure and improve GPU performance.
// If you want to change this to int64 then you must also update
// loom.proto and REGISTER_WEAVER_OP in weaver_op_base.h.
typedef int32 tensor_idx_t;

// START_SKIP_DOXYGEN
// The Weaver object uses `LoomResult`s internally to represent the results of
// computations.  Weaver uses integer result IDs to refer to these results in
// its API.  These are opaque handles representing results.
//
// Results can be either constants, named tensors, or the results of calling
// loom functions on previous results.
struct LoomResult {
  // depth: Which step this operation happens at (0 for constants and named
  //   Tensors, positive numbers for ops.)
  tensor_idx_t depth;

  // ts_idx: An integer ID representing the TypeShape of the result.
  tensor_idx_t ts_idx;

  // op_idx: Which op was called to generate this result
  tensor_idx_t op_idx;

  // op_output_idx: Which output of this op the result refers to.
  tensor_idx_t op_output_idx;

  // pos_idx: Which call of this op the result refers to, or which constant it
  //   refers of this shape it refers to.
  tensor_idx_t pos_idx;

  // cached_passthrough: which LoomResult ID we got by calling pass_through on
  // the current LoomResult (-1 if we never did.)
  tensor_idx_t cached_passthrough = -1;
};


// The Weaver object uses these internally to represent TypeShape metadata.
struct TypeShape {
  tensorflow::DataType dtype;
  tensorflow::TensorShape shape;
  string name;
};
// END_SKIP_DOXYGEN

/// Validates a LoomMetadata proto.
///
/// Args:
///   metadata: a LoomMetadata proto to validate.
///
/// Returns false and sets the error string if:
///  - max_depth is less than -1
///  - Any of the elements of op_input_ts_idx or op_output_ts_idx are invalid
///    typeshapes (cannot be used as indices for type_shapes_.)
///  - The first num_type_shapes_ ops aren't pass-through ops for the
///    corresponding typeshape.  (num_type_shapes_ is the length of
///    type_shapes_.)
bool VerifyLoomMetadata(const LoomMetadata &metadata, string *error_string);

/// *Weaver* builds a schedule to be run by a Loom, producing vectors of
/// integers (wiring diagrams) to be fed to Loom's `tf.gather` ops in order to
/// drive the loom according to the schedule the user specified.
///
/// A `Weaver` is constructed by passing in a serialized `LoomMetadata` proto
/// (among other things, this specifies the types and operations supported by
/// the loom.)
///
/// To build up a schedule, the user calls `Weaver::MakeConstant`,
/// `Weaver::MakeConstantSerialized`, `Weaver::NamedTensor`, or
/// `Weaver::BatchInput` in order to create the terminal nodes.
///
/// These methods all return integer values (hereafter, result IDs) that
/// represent nodes in the partially completed graph.  The user can then grow
/// the graph by feeding preexisting result IDs as arguments to
/// `Weaver::CallOp`, which returns result IDs refering to the return values of
/// that call to the operation.
///
/// Once the graph has been built up, the user calls `Weaver::AddOutput` in
/// order tag nodes that represent computations that ought to be passed out into
/// the output tensor for the appropriate TypeShape.  Once the graph is complete
/// and all outputs have been tagged the user calls `Weaver::Finalize`, which
/// compiles the graph into a collection of integer vectors to be passed to
/// Loom's gather operations.
///
/// After `Weaver::Finalize` has been called, the finished wiring diagram can be
/// extractied using `Weaver::GetWiring` and `Weaver::GetOutputWiring`.  The
/// constants to be fed into the Loom as initial state can be extracted using
/// `Weaver::BatchConstantValues`.
///
/// Weaver also supports serialization and deserialization to `WeaverMessage`
/// via `Weaver::Seriailize` and `Weaver::Deserialize`.  (See loom.proto for the
/// definition of `WeaverMessage`.)
class Weaver {
 public:
  ///   `serialized_loom_metadata`: A serialized `LoomMetatdata` proto.  See
  ///   loom.proto for details.
  ///
  /// Sets a non-empty error string if either the metadata fails to
  /// deserialize to a `LoomMetadata` proto, or the `LoomMetadata` is invalid
  /// according to `VerifyLoomMetadata`.
  ///
  /// The caller must check for a status after constructing the Weaver before
  /// using it for anything.
  explicit Weaver(const string &serialized_loom_metadata);

  /// Returns the error string (non empty if any previous operation has failed.)
  const string &error_string() {
    return error_string_;
  }

  /// Resets the weaver back to the state it was in right when the constructor
  /// was called.
  void Reset();

  /// Returns an N-dimensional array containing the constant values for the
  /// given typeshape, stacked in a batch.
  tensorflow::Tensor BatchConstantValues(tensor_idx_t ts_idx) const;

  /// Serializes this Weaver into a string (a serialized
  /// `WeaverMessage`.)
  ///
  /// Returns the empty string and sets an error string if serialization fails.
  string Serialize() const;

  /// Overwrites this Weaver from the string (a serialized `WeaverMessage`.)
  ///
  /// Returns true unless an error occurs during deserialization.  If an error
  /// occurs, returns false and sets the error string.  In the event an error
  /// occurs, no guarantees are made about the Weaver's future behavior.
  ///
  /// WARNING: does almost no checking as to whether the contents of
  /// `serialized_weaver` are valid.
  bool Deserialize(const string &serialized_weaver);

  /// Returns the maximum depth of this scheduler.
  tensor_idx_t MaxDepth() const {
    return max_depth_;
  }

  /// Returns the number of typeshapes this scheduler has.
  tensor_idx_t NumTypeShapes() const {
    return num_type_shapes_;
  }

  /// Returns the largest depth of any operation scheduled so far.
  tensor_idx_t Deepest() const {
    return deepest_;
  }

  /// Returns the number of operations this scheduler supports.
  tensor_idx_t NumOps() const {
    return num_ops_;
  }

  /// Returns the name of an op given its index.
  const string& OpName(tensor_idx_t op_idx) const {
    return op_names_[op_idx];
  }

  /// Returns the TypeShape indices of an operation's arguments.
  const std::vector<tensor_idx_t>& InputTypeShapes(tensor_idx_t op_idx) const {
    return op_input_ts_idx_[op_idx];
  }

  /// Returns the TypeShape indices of an operation's return values.
  const std::vector<tensor_idx_t>& OutputTypeShapes(tensor_idx_t op_idx) const {
    return op_input_ts_idx_[op_idx];
  }

  /// Returns the depth of the node `result_id`.
  ///
  /// Returns -1 if `result_id` is invalid.
  tensor_idx_t Depth(tensor_idx_t result_id) const;

  /// Returns the TypeShape ID of the node `result_id`.
  ///
  /// Returns -1 if `result_id` is invalid.
  tensor_idx_t GetTypeShape(tensor_idx_t result_id) const;

  /// Creates a result ID refering to the `named_tensor_idx`th NamedTensor
  /// with TypeShape `ts_idx` (these were passed to the Loom when it was
  /// constructed.)
  /// Returns -1 and sets the error string if either `ts_idx` or
  /// `named_tensor_idx` is invalid.
  ///
  /// Note: Repeated calls to `GetNamedTensor` can bloat the schedule
  /// with copies of the tensor.  Writers of C++ Weaver Ops should
  /// call `GetNamedTensor` once for each named tensor they wish to use.
  tensor_idx_t GetNamedTensor(
      tensor_idx_t ts_idx, tensor_idx_t named_tensor_idx);

  /// `MakeConstantSerialized` creates a new result ID representing an input
  /// value of TypeShape `ts_idx` using the serialized contents of
  /// `tensor_bytes`.  (It's the Weaver's responsibility to hold the value)
  ///
  /// Returns -1 and sets the error string if `ts_idx` is invalid or if that
  /// TypeShape is in batch-mode or if the value to be set is invalid.
  tensor_idx_t MakeConstantSerialized(
      tensor_idx_t ts_idx, const string &tensor_bytes);

  /// `MakeConstant` creates a new result ID representing an input
  /// value of TypeShape `ts_idx`  using the serialized contents of
  /// `tensor_proto`.  (It's the Weaver's responsibility to hold the value)
  ///
  /// Returns -1 and sets the error string if `ts_idx` is invalid or if that
  /// TypeShape is in batch-mode or if the value to be set is invalid.
  tensor_idx_t MakeConstant(
      tensor_idx_t ts_idx, const tensorflow::TensorProto &tensor_proto);

  /// `MakeConstant` creates a new result ID representing an input
  /// value of TypeShape `ts_idx`  using the serialized contents of
  /// `tensor`.  (It's the Weaver's responsibility to hold the value)
  ///
  /// Returns -1 and sets the error string if `ts_idx` is invalid or if that
  /// TypeShape is in batch-mode or if the value to be set is invalid.
  tensor_idx_t MakeConstant(
      tensor_idx_t ts_idx, const tensorflow::Tensor &tensor);

  /// `BatchInput` creates a new result ID representing the `batch_idx`th row of
  /// the batch input tensor for TypeShape `ts_idx` (a single batch tensor
  /// provided to the loom on construction.)
  ///
  /// Returns -1 and sets the error string if `ts_idx` is invalid or if that
  /// TypeShape is not in batch-mode.
  ///
  /// Runs with no errors if `batch_idx` is out of range (that will result in a
  /// gather error when the loom is run; this is because the weaver doesn't know
  /// how large the batch will be.)
  tensor_idx_t BatchInput(tensor_idx_t ts_idx, tensor_idx_t batch_idx);

  /// Creates result IDs representing return values from the `op_idx`th op.
  ///
  /// `op_idx` is the ID of an op, and `args` is a vector of result IDs.
  ///
  /// Returns an empty vector and sets the error string if `op_idx` is an
  /// invalid op ID or if any of args are invalid result IDs.
  ///
  /// Returns the a vector of result IDs representing the return values.
  std::vector<tensor_idx_t> CallOp(
      tensor_idx_t op_idx, const std::vector<tensor_idx_t> &args);

  /// Adds `result_id` to the list of results to pass on to the Loom's output
  /// tensors.
  ///
  /// Returns false and sets the error string if `result_id` is invalid.
  bool AddOutput(tensor_idx_t result_id);

  /// Merges the wiring and set outputs from `other` into this Weaver.
  ///
  /// May return false and set the error string in some cases in which the merge
  /// is impossible.
  ///
  /// WARNING: does not check whether `other` has the same set of loom ops, same
  /// set of type-shapes, etc.  Unpredictable behavior may ensue if you call
  /// `MergeFromSerialized` with a serialized scheduler with a different op set.
  bool MergeFromSerialized(const string &other);

  /// Compiles this graph into a wiring diagram which can be accessed using
  /// `Weaver::GetWiring` and `Weaver::GetOutputWiring`.
  ///
  /// Only does anything the first time it's called.
  void Finalize();

  /// Returns the wiring for the gather operation at `depth` for the
  /// `op_arg_idx`th argument of the `op_idx`th operation.  (This gather
  /// operation will select rows from the tensor of the appropriate Typeshape.)
  ///
  /// Should only be called after `Weaver::Finalize` has been called.
  const std::vector<tensor_idx_t> &GetWiring(
      tensor_idx_t depth, tensor_idx_t op_idx, tensor_idx_t op_arg_idx) const {
    auto key = std::make_tuple(depth, op_idx, op_arg_idx);
    return final_wiring_.find(key)->second;
  }

  /// Returns the wiring for the gather operation after the while loop which
  /// selects which values should go to the output tensor for `ts_idx`.
  ///
  /// Should only be called after `Weaver::Finalize` has been called.
  const std::vector<tensor_idx_t> &GetOutputWiring(tensor_idx_t ts_idx) const {
    return final_output_wiring_.at(ts_idx);
  }

 private:
  // Internal portion of CallOp which assumes all args have the same depth, and
  // have the correct typeshape.
  std::vector<tensor_idx_t> AlignedCallOp(
      tensor_idx_t op_idx, const std::vector<tensor_idx_t> &args);

  // Repeatedly call the pass-through op of the appropriate type_shape until
  // the result with ID `result_id` has been promoted to `target_depth`.
  //
  // Returns the result ID of the new result (which will have depth
  // `target_depth`.
  tensor_idx_t Deepen(tensor_idx_t result_id, tensor_idx_t target_depth);


  // The Metadata for this loom.
  LoomMetadata metadata_;

  // The maximum allowable depth that LoomOps can be nested.
  tensor_idx_t max_depth_;

  // The number of TypeShapes this loom supports.
  tensor_idx_t num_type_shapes_;

  // The number of operations this loom supports.
  tensor_idx_t num_ops_;

  // The names of the LoomOps (used for user-friendly error messages.)
  std::vector<string> op_names_;

  // Type shapes.
  std::vector<TypeShape> type_shapes_;

  // How many named tensors there are for a given TypeShape.
  std::vector<tensor_idx_t> num_named_tensors_by_ts_idx_;

  // op_input_ts_idx_[op_idx] is a list of the typeshape IDs of the arguments
  // accepted by op number op_idx.
  std::vector<std::vector<tensor_idx_t> > op_input_ts_idx_;

  // op_input_ts_idx_[op_idx] is a list of the typeshape IDs of the outputs
  // emitted by op number op_idx.
  std::vector<std::vector<tensor_idx_t> > op_output_ts_idx_;

  // The maximum depth of any LoomResult seen so far.
  tensor_idx_t deepest_;

  // Has Finalize() been called yet?
  bool finalized_;

  // How many constants have been created for a given type_shape.
  std::vector<tensor_idx_t> num_constants_by_type_shape_;

  // constant_values_by_type_shape_ Constants values by TypeShape.
  // The elements of constant_values_by_type_shape_[i] have
  // dtype metadata.type_shape_metadata.dtype() and shape
  // metadata.type_shape_metadata.shape()
  std::vector<std::vector<tensorflow::Tensor> > constant_values_by_type_shape_;

  // loom_results_: The LoomResults created thus far.  Result IDs index into
  // this vector.
  std::vector<LoomResult> loom_results_;

  // The result IDs of the LoomResults that have been slated to be sent to the
  // Loom's output tensors using the `AddOutput` function.
  std::vector<tensor_idx_t> output_result_ids_;

  // _wiring: maps (depth, op_idx, arg_idx) to  a vector of result_ids.  This
  // vector shows which results are being used as the inputs of which op calls.
  std::map<std::tuple<tensor_idx_t, tensor_idx_t, tensor_idx_t>,
           std::vector<tensor_idx_t> > wiring_results_;

  // final_wiring_: maps (depth, op_idx, arg_idx) to a vector of integers to be
  // used by the loom.  Populated by finalize.
  std::map<std::tuple<tensor_idx_t, tensor_idx_t, tensor_idx_t>,
           std::vector<tensor_idx_t> > final_wiring_;

  // final_output_wiring_: maps ts_idx to a vector of integers to be
  // used by the loom in the gathers at the end of the loom that produce the
  // Loom's output tensors.  Populated by finalize.
  std::vector<std::vector<tensor_idx_t> > final_output_wiring_;

  // error_string_: This variable gets set if any operation fails.
  mutable string error_string_;
};

}  // namespace fold
}  // namespace tensorflow

#endif  // TENSORFLOW_FOLD_LOOM_WEAVER_H_
