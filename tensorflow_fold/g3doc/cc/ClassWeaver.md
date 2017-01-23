<!-- This file is machine generated: DO NOT EDIT! -->

# `class tensorflow::fold::Weaver`



Weaver builds a schedule to be run by a Loom, producing vectors of integers (wiring diagrams) to be fed to Loom&apos;s `tf.gather` ops in order to drive the loom according to the schedule the user specified.

A ` Weaver ` is constructed by passing in a serialized `LoomMetadata` proto (among other things, this specifies the types and operations supported by the loom.)

To build up a schedule, the user calls ` Weaver::MakeConstant `, ` Weaver::MakeConstantSerialized `, `Weaver::NamedTensor`, or ` Weaver::BatchInput ` in order to create the terminal nodes.

These methods all return integer values (hereafter, result IDs) that represent nodes in the partially completed graph. The user can then grow the graph by feeding preexisting result IDs as arguments to ` Weaver::CallOp `, which returns result IDs refering to the return values of that call to the operation.

Once the graph has been built up, the user calls ` Weaver::AddOutput ` in order tag nodes that represent computations that ought to be passed out into the output tensor for the appropriate TypeShape . Once the graph is complete and all outputs have been tagged the user calls ` Weaver::Finalize `, which compiles the graph into a collection of integer vectors to be passed to Loom&apos;s gather operations.

After ` Weaver::Finalize ` has been called, the finished wiring diagram can be extractied using ` Weaver::GetWiring ` and ` Weaver::GetOutputWiring `. The constants to be fed into the Loom as initial state can be extracted using ` Weaver::BatchConstantValues `.

Weaver also supports serialization and deserialization to `WeaverMessage` via `Weaver::Seriailize` and ` Weaver::Deserialize `. (See loom.proto for the definition of `WeaverMessage`.)

###Member Details

#### `tensorflow::fold::Weaver::Weaver(const string &serialized_loom_metadata)`
(#tensorflow_fold_Weaver_Weaver)



serialized_loom_metadata: A serialized LoomMetatdata proto. See loom.proto for details.

Sets a non-empty error string if either the metadata fails to deserialize to a LoomMetadata proto, or the LoomMetadata is invalid according to VerifyLoomMetadata.

The caller must check for a status after constructing the Weaver before using it for anything.

#### `const string& tensorflow::fold::Weaver::error_string()`
(#const_string_tensorflow_fold_Weaver_error_string)

Returns the error string (non empty if any previous operation has failed.)



#### `void tensorflow::fold::Weaver::Reset()`
(#void_tensorflow_fold_Weaver_Reset)



Resets the weaver back to the state it was in right when the constructor was called.

#### `Tensor tensorflow::fold::Weaver::BatchConstantValues(tensor_idx_t ts_idx) const`
(#Tensor_tensorflow_fold_Weaver_BatchConstantValues)



Returns an N-dimensional array containing the constant values for the given typeshape, stacked in a batch.

#### `string tensorflow::fold::Weaver::Serialize() const`
(#string_tensorflow_fold_Weaver_Serialize)



Serializes this Weaver into a string (a serialized WeaverMessage.)

Returns the empty string and sets an error string if serialization fails.

#### `bool tensorflow::fold::Weaver::Deserialize(const string &serialized_weaver)`
(#bool_tensorflow_fold_Weaver_Deserialize)



Overwrites this Weaver from the string (a serialized WeaverMessage.)

Returns true unless an error occurs during deserialization. If an error occurs, returns false and sets the error string. In the event an error occurs, no guarantees are made about the Weaver &apos;s future behavior.

WARNING: does almost no checking as to whether the contents of serialized_weaver are valid.

#### `tensor_idx_t tensorflow::fold::Weaver::MaxDepth() const`
(#tensor_idx_t_tensorflow_fold_Weaver_MaxDepth)

Returns the maximum depth of this scheduler.



#### `tensor_idx_t tensorflow::fold::Weaver::NumTypeShapes() const`
(#tensor_idx_t_tensorflow_fold_Weaver_NumTypeShapes)

Returns the number of typeshapes this scheduler has.



#### `tensor_idx_t tensorflow::fold::Weaver::Deepest() const`
(#tensor_idx_t_tensorflow_fold_Weaver_Deepest)

Returns the largest depth of any operation scheduled so far.



#### `tensor_idx_t tensorflow::fold::Weaver::NumOps() const`
(#tensor_idx_t_tensorflow_fold_Weaver_NumOps)

Returns the number of operations this scheduler supports.



#### `const string& tensorflow::fold::Weaver::OpName(tensor_idx_t op_idx) const`
(#const_string_tensorflow_fold_Weaver_OpName)

Returns the name of an op given its index.



#### `const std::vector<tensor_idx_t>& tensorflow::fold::Weaver::InputTypeShapes(tensor_idx_t op_idx) const`
(#const_std_vector_tensor_idx_t_tensorflow_fold_Weaver_InputTypeShapes)

Returns the TypeShape indices of an operation&apos;s arguments.



#### `const std::vector<tensor_idx_t>& tensorflow::fold::Weaver::OutputTypeShapes(tensor_idx_t op_idx) const`
(#const_std_vector_tensor_idx_t_tensorflow_fold_Weaver_OutputTypeShapes)

Returns the TypeShape indices of an operation&apos;s return values.



#### `tensor_idx_t tensorflow::fold::Weaver::Depth(tensor_idx_t result_id) const`
(#tensor_idx_t_tensorflow_fold_Weaver_Depth)



Returns the depth of the node `result_id`.

Returns -1 if `result_id` is invalid.

#### `tensor_idx_t tensorflow::fold::Weaver::GetTypeShape(tensor_idx_t result_id) const`
(#tensor_idx_t_tensorflow_fold_Weaver_GetTypeShape)



Returns the TypeShape ID of the node `result_id`.

Returns -1 if `result_id` is invalid.

#### `tensor_idx_t tensorflow::fold::Weaver::GetNamedTensor(tensor_idx_t ts_idx, tensor_idx_t named_tensor_idx)`
(#tensor_idx_t_tensorflow_fold_Weaver_GetNamedTensor)



Creates a result ID refering to the `named_tensor_idx`th NamedTensor with TypeShape `ts_idx` (these were passed to the Loom when it was constructed.) Returns -1 and sets the error string if either `ts_idx` or `named_tensor_idx` is invalid.

Note: Repeated calls to GetNamedTensor can bloat the schedule with copies of the tensor. Writers of C++ Weaver Ops should call GetNamedTensor once for each Named Tensor they wish to use.

#### `tensor_idx_t tensorflow::fold::Weaver::MakeConstantSerialized(tensor_idx_t ts_idx, const string &tensor_bytes)`
(#tensor_idx_t_tensorflow_fold_Weaver_MakeConstantSerialized)



MakeConstantSerialized creates a new result ID representing an input value of TypeShape `ts_idx` using the serialized contents of `tensor_bytes`. (It&apos;s the Weaver &apos;s responsibility to hold the value)

Returns -1 and sets the error string if `ts_idx` is invalid or if that TypeShape is in batch-mode or if the value to be set is invalid.

#### `tensor_idx_t tensorflow::fold::Weaver::MakeConstant(tensor_idx_t ts_idx, const tensorflow::TensorProto &tensor_proto)`
(#tensor_idx_t_tensorflow_fold_Weaver_MakeConstant)



MakeConstant creates a new result ID representing an input value of TypeShape `ts_idx` using the serialized contents of `tensor_proto`. (It&apos;s the Weaver &apos;s responsibility to hold the value)

Returns -1 and sets the error string if `ts_idx` is invalid or if that TypeShape is in batch-mode or if the value to be set is invalid.

#### `tensor_idx_t tensorflow::fold::Weaver::MakeConstant(tensor_idx_t ts_idx, const tensorflow::Tensor &tensor)`
(#tensor_idx_t_tensorflow_fold_Weaver_MakeConstant)



MakeConstant creates a new result ID representing an input value of TypeShape `ts_idx` using the serialized contents of `tensor`. (It&apos;s the Weaver &apos;s responsibility to hold the value)

Returns -1 and sets the error string if `ts_idx` is invalid or if that TypeShape is in batch-mode or if the value to be set is invalid.

#### `tensor_idx_t tensorflow::fold::Weaver::BatchInput(tensor_idx_t ts_idx, tensor_idx_t batch_idx)`
(#tensor_idx_t_tensorflow_fold_Weaver_BatchInput)



BatchInput creates a new result ID representing the `batch_idx`th row of the batch input tensor for TypeShape `ts_idx` (a single batch tensor provided to the loom on construction.)

Returns -1 and sets the error string if `ts_idx` is invalid or if that TypeShape is not in batch-mode.

Runs with no errors if batch_idx is out of range (that will result in a gather error when the loom is run; this is because the weaver doesn&apos;t know how large the batch will be.)

#### `std::vector< tensor_idx_t > tensorflow::fold::Weaver::CallOp(tensor_idx_t op_idx, const std::vector< tensor_idx_t > &args)`
(#std_vector_tensor_idx_t_tensorflow_fold_Weaver_CallOp)



Creates result IDs representing return values from the `op_idx`th op.

`op_idx` is the ID of an op, and `args` is a vector of result IDs.

Returns an empty vector and sets the error string if `op_idx` is an invalid op ID or if any of args are invalid result IDs.

Returns the a vector of result IDs representing the return values.

#### `bool tensorflow::fold::Weaver::AddOutput(tensor_idx_t result_id)`
(#bool_tensorflow_fold_Weaver_AddOutput)



Adds `result_id` to the list of results to pass on to the Loom&apos;s output tensors.

Returns false and sets the error string if `result_id` is invalid.

#### `bool tensorflow::fold::Weaver::MergeFromSerialized(const string &other)`
(#bool_tensorflow_fold_Weaver_MergeFromSerialized)



Merges the wiring and set outputs from `other` into this Weaver .

May return false and set the error string in some cases in which the merge is impossible.

WARNING: does not check whether `other` has the same set of loom ops, same set of type-shapes, etc. Unpredictable behavior may ensue if you call MergeFromSerialized with a serialized scheduler with a different op set.

#### `void tensorflow::fold::Weaver::Finalize()`
(#void_tensorflow_fold_Weaver_Finalize)



Compiles this graph into a wiring diagram which can be accessed using ` Weaver::GetWiring ` and ` Weaver::GetOutputWiring `.

Only does anything the first time it&apos;s called.

#### `const std::vector<tensor_idx_t>& tensorflow::fold::Weaver::GetWiring(tensor_idx_t depth, tensor_idx_t op_idx, tensor_idx_t op_arg_idx) const`
(#const_std_vector_tensor_idx_t_tensorflow_fold_Weaver_GetWiring)



Returns the wiring for the gather operation at `depth` for the `op_arg_idx`th argument of the `op_idx`th operation. (This gather operation will select rows from the tensor of the appropriate Typeshape.)

Should only be called after ` Weaver::Finalize ` has been called.

#### `const std::vector<tensor_idx_t>& tensorflow::fold::Weaver::GetOutputWiring(tensor_idx_t ts_idx) const`
(#const_std_vector_tensor_idx_t_tensorflow_fold_Weaver_GetOutputWiring)



Returns the wiring for the gather operation after the while loop which selects which values should go to the output tensor for `ts_idx`.

Should only be called after ` Weaver::Finalize ` has been called.
