# Protocol buffers

## Introduction

The `tensorflow_fold.util.proto_tools` python C++ extension provides a function
called `serialized_message_to_tree` which takes the type of a protocol buffer
and its content (serialized as a string) and converts it into a nested native
Python datastructure composed of dictionaries and lists.  This function's
behavior is analogous to `json.loads` (except that enum values are treated
specially, which will be described later.)

## Rationale

The outputs of `serialized_message_to_tree` can be traversed faster than the
python protocol buffer API, the resulting traversal code is more Pythonic; in
particular, it eliminates the need for separate Fold blocks for dealing with
protocol buffers and data loaded from JSON or other sources.

## Setup

Before `serialized_message_to_tree` can be called, `proto_tools` must be told
where the `.proto` files defining the protocol buffers are using
`map_proto_source_tree_path(virtual_path, disk_path)`.  One or more calls to
`map_proto_source_tree_path` will build up a virtual source tree (in a manner
analgous to Unix's `mount` command with the arguments reversed.)  If all your
proto files are in a single directory and their absolute import statements are
written relative to that directory, then a single call to:
`map_proto_source_tree_path("", dir_path)` will suffice.

Next, the protocol buffer message types that you care about should be imported
using `proto_tools.import_proto_file(viritual_path)`.  One of the calls to
`map_proto_source-tree_path` must have taken a virtual path which is a prefix of
`virtual_path` for the import to resolve.

Once this is done, `proto_tools.serialized_message_to_tree(message_type,
str)` should work properly with any protocol buffer message types declared in
the imported proto files.

See `util/proto_test.py` for example usages.

## Outputs

Most types of proto fields are dealt with straight-forwardly.  Strings fields
become python strings, integers become python integers, floats become python
floats and so on.  Repeated fields are rendered as python lists.  Submessages
are rendered as Python dictionaries whose keys are strings.

Enum fields are converted into Python dictionaries containing the name, index,
and number of the enum value.
