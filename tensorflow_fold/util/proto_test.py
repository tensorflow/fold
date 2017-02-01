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

"""Tests for tensorflow_fold.util.proto."""
import os

# import google3
import tensorflow as tf
from tensorflow_fold.util import proto_tools
from tensorflow_fold.util import test3_pb2
from tensorflow_fold.util import test_pb2
from google.protobuf import text_format


# Make sure SerializedMessageToTree can see our proto files.

proto_tools.map_proto_source_tree_path("", os.getcwd())
# Note: Tests run in the bazel root directory, which we will use as the root for
# our source protos.

proto_tools.import_proto_file("tensorflow_fold/util/test.proto")
proto_tools.import_proto_file("tensorflow_fold/util/test3.proto")


def MakeCyclicProto(message_str):
  return text_format.Parse(message_str, test_pb2.CyclicType())


def MakeCyclicProto3(message_str):
  return text_format.Parse(message_str, test3_pb2.CyclicType3())


def MakeOneAtomProto(message_str):
  return text_format.Parse(message_str, test_pb2.OneAtom())


class ProtoTest(tf.test.TestCase):

  def testSerializedMessageToTree(self):
    example = MakeCyclicProto(
        "some_same<"
        "  many_int32: 1"
        "  many_int32: 2"
        "  some_same<"
        "    many_int32: 3"
        "    many_int32: 4"
        "    some_bool: false"
        "  >"
        ">"
        "some_enum: THAT")
    result = proto_tools.serialized_message_to_tree(
        "tensorflow.fold.CyclicType", example.SerializeToString())
    self.assertEqual(result["some_same"]["many_int32"], [1, 2])
    self.assertEqual(result["some_same"]["some_same"]["many_int32"], [3, 4])
    self.assertEqual(result["some_same"]["some_same"]["some_bool"], False)
    self.assertEqual(result["many_bool"], [])
    self.assertEqual(result["some_bool"], None)
    self.assertEqual(result["some_same"]["many_bool"], [])
    self.assertEqual(result["some_same"]["some_bool"], None)
    self.assertEqual(result["some_enum"]["name"], "THAT")
    self.assertEqual(result["some_enum"]["index"], 1)
    self.assertEqual(result["some_enum"]["number"], 1)

  def testSerializedMessageToTreeProto3(self):
    example = MakeCyclicProto3(
        "some_same<"
        "  many_int32: 1"
        "  many_int32: 2"
        "  some_same<"
        "    many_int32: 3"
        "    many_int32: 4"
        "    some_bool: false"
        "  >"
        ">"
        "some_enum: THAT")
    result = proto_tools.serialized_message_to_tree(
        "tensorflow.fold.CyclicType3", example.SerializeToString())
    self.assertEqual(result["some_same"]["many_int32"], [1, 2])
    self.assertEqual(result["some_same"]["some_same"]["many_int32"], [3, 4])
    self.assertEqual(result["some_same"]["some_same"]["some_bool"], False)
    self.assertEqual(result["many_bool"], [])
    self.assertEqual(result["some_bool"], False)
    self.assertEqual(result["some_same"]["many_bool"], [])
    self.assertEqual(result["some_same"]["some_bool"], False)
    self.assertEqual(result["some_enum"]["name"], "THAT")
    self.assertEqual(result["some_enum"]["index"], 1)
    self.assertEqual(result["some_enum"]["number"], 1)

  def testSerializedMessageToTreeOneofEmpty(self):
    empty_proto = MakeOneAtomProto("").SerializeToString()
    empty_result = proto_tools.serialized_message_to_tree(
        "tensorflow.fold.OneAtom", empty_proto)
    self.assertEqual(empty_result["atom_type"], None)
    self.assertEqual(empty_result["some_int32"], None)
    self.assertEqual(empty_result["some_int64"], None)
    self.assertEqual(empty_result["some_uint32"], None)
    self.assertEqual(empty_result["some_uint64"], None)
    self.assertEqual(empty_result["some_double"], None)
    self.assertEqual(empty_result["some_float"], None)
    self.assertEqual(empty_result["some_bool"], None)
    self.assertEqual(empty_result["some_enum"], None)
    self.assertEqual(empty_result["some_string"], None)

  def testSerializedMessageToTreeOneof(self):
    empty_proto = MakeOneAtomProto("some_string: \"x\"").SerializeToString()
    empty_result = proto_tools.serialized_message_to_tree(
        "tensorflow.fold.OneAtom", empty_proto)
    self.assertEqual(empty_result["atom_type"], "some_string")
    self.assertEqual(empty_result["some_int32"], None)
    self.assertEqual(empty_result["some_int64"], None)
    self.assertEqual(empty_result["some_uint32"], None)
    self.assertEqual(empty_result["some_uint64"], None)
    self.assertEqual(empty_result["some_double"], None)
    self.assertEqual(empty_result["some_float"], None)
    self.assertEqual(empty_result["some_bool"], None)
    self.assertEqual(empty_result["some_enum"], None)
    self.assertEqual(empty_result["some_string"], "x")

  def testNonConsecutiveEnum(self):
    name = "tensorflow.fold.NonConsecutiveEnumMessage"
    msg = test_pb2.NonConsecutiveEnumMessage(
        the_enum=test_pb2.NonConsecutiveEnumMessage.THREE)
    self.assertEqual(
        {"the_enum": {"name": "THREE", "index": 1, "number": 3}},
        proto_tools.serialized_message_to_tree(name, msg.SerializeToString()))
    msg.the_enum = test_pb2.NonConsecutiveEnumMessage.SEVEN
    self.assertEqual(
        {"the_enum": {"name": "SEVEN", "index": 0, "number": 7}},
        proto_tools.serialized_message_to_tree(name, msg.SerializeToString()))


if __name__ == "__main__":
  tf.test.main()
