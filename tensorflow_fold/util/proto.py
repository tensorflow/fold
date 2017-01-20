# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility code for protocol buffer introspection."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import google3
from tensorflow_fold.util import proto_tools


def MapProtoPath(virtual_path, disk_path):
  proto_tools.map_proto_source_tree_path(virtual_path, disk_path)


def ImportProtoFile(filename):
  proto_tools.import_proto_file(filename)


def SerializedMessageToTree(message_type, message_str):
  """Converts a serialized protobuf into a tree of python dicts and lists.

  Args:
    message_type: A string containing the full name of the message's type.
     Note that you need to link in the C++ version of that protobuf so the C++
     implementation of MessageToTree can see it.
    message_str: A message of type `message_type` serialized as a string.

  Returns:
    A dictionary of the message's values by fieldname, where the function
    renders repeated fields as lists, submessages via recursion and field values
    in a straight forward way (except for enums which turn into dictionaries
    whose keys are "name" and "index" and "value".)
  """
  return proto_tools.serialized_message_to_tree(message_type, message_str)
