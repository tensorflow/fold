# Protocol buffers

## Introduction

The `tensorflow_fold.util.proto_tools` Python C++ extension provides a function
called `serialized_message_to_tree` which takes the message type of a protocol
buffer and its content (serialized as a string) and converts it into a nested
native Python data-structure composed of dictionaries and lists.  This
function's behavior is analogous to `json.loads` (except that enum values are
treated specially, which will be described in more detail below.)

The Fold blocks API provides a 
[`SerializedMessageToTree`](py/td.md#tdserializedmessagetotreemessage_type_name)
block that serves as a convenient wrapper for this function.

## Rationale

The outputs of `serialized_message_to_tree` can be traversed faster than the
Python protocol buffer API, the resulting traversal code is more Pythonic; in
particular, it eliminates the need for separate Fold blocks for dealing with
protocol buffers and data loaded from JSON or other sources.

## Setup

Before `serialized_message_to_tree` can be called, `proto_tools` must be told
the locations of the `.proto` files (which define schema of the the protocol
buffers) using `map_proto_source_tree_path(virtual_path, disk_path)`.  One or
more calls to `map_proto_source_tree_path` will build up a virtual source tree
(in a manner analgous to Unix's `mount` command with the arguments reversed.)
If all your proto files are in a single directory and their absolute import
statements are written relative to that directory, then a single call to:
`map_proto_source_tree_path("", dir_path)` will suffice.

Next, the protocol buffer message types that you care about should be imported
using `proto_tools.import_proto_file(viritual_path)`.  One of the calls to
`map_proto_source_tree_path` must have taken a virtual path which is a prefix of
`virtual_path` for the import to resolve.  `virtual_path` should point to a
valid `.proto` file (after the path has been resolved), as should any paths in
any import statements the `.proto` file might contain, etc.

Once this is done, `proto_tools.serialized_message_to_tree(message_type,
str)` should work properly with any protocol buffer message types declared in
the imported proto files.  (Here `message_type` is the fully qualified message
type which includes the package name, e.g. `tensorflow.fold.LoomMetadata`.)

See [util/proto\_test.py](../util/proto_test.py) for example usages.

## Outputs

Most types of proto fields are dealt with straight-forwardly.  Strings fields
become Python strings, integers become Python integers, floats become Python
floats and so on.  Repeated fields are rendered as Python lists.  Submessages
are rendered as Python dictionaries whose keys are strings.

Enum fields are converted into Python dictionaries containing the name, index,
and number of the enum value (keyed by "name", "index", and "number"
respectively.)
