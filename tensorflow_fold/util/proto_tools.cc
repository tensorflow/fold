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

#include <string>
#include <vector>
#include <map>

#include <Python.h>

#include "google/protobuf/descriptor.h"
#include "google/protobuf/dynamic_message.h"
#include "google/protobuf/message.h"
#include "google/protobuf/reflection.h"
#include "google/protobuf/compiler/importer.h"

#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace fold {
namespace proto_tools {

using google::protobuf::Descriptor;
using google::protobuf::DescriptorPool;
using google::protobuf::DynamicMessageFactory;
using google::protobuf::FieldDescriptor;
using google::protobuf::FileDescriptor;
using google::protobuf::OneofDescriptor;
using google::protobuf::Message;
using google::protobuf::Reflection;
using google::protobuf::EnumValueDescriptor;
using google::protobuf::compiler::DiskSourceTree;
using google::protobuf::compiler::Importer;
using google::protobuf::compiler::MultiFileErrorCollector;
using CppType = google::protobuf::FieldDescriptor::CppType;
using tensorflow::strings::StrCat;


PyObject* PyString_FromCPPString(const string& s) {
#if PY_MAJOR_VERSION >= 3
  return PyUnicode_FromStringAndSize(s.c_str(), s.size());
#else
  return PyString_FromStringAndSize(s.c_str(), s.size());
#endif
}

PyObject* EnumValueToNameAndIndex(const EnumValueDescriptor* desc) {
  PyObject *result = PyDict_New();

  PyObject *name = PyString_FromCPPString(desc->name());
  PyDict_SetItemString(result, "name", name);
  Py_DECREF(name);

  PyObject *index = PyLong_FromLong(desc->index());
  PyDict_SetItemString(result, "index", index);
  Py_DECREF(index);

  PyObject *number = PyLong_FromLong(desc->number());
  PyDict_SetItemString(result, "number", number);
  Py_DECREF(number);

  return result;
}

// Recursively converts a Message into a Tree of Python objects.
//
// Returns nullptr and sets PyErr if the conversion fails at any point.
//
// Messages become dicts mapping field names to their values.
//
// Integral fields become PyLongs.
// Real fields become PyFloats.
// Strings become PyStrings.
// Enums become dictionaries of the form:
//   {"name": "enum_value_name", "index": <enum_value_index>}
// Repeated fields become PyLists.
PyObject* MessageToTree(
    const Message &message,
    string* error_pos);

PyObject* FieldToTree(
    const Message &message,
    const Reflection &reflection,
    const FieldDescriptor &field,
    bool is_proto3,
    string *error_pos) {

  auto message_handler = std::bind(
      MessageToTree, std::placeholders::_1, error_pos);

  // We need always_has because if the proto syntax is proto3, for singular
  // non-message fields, HasField returns false if the default value is set.
  // However, we want to return the default value instead of a None in this
  // case.
  bool always_has = is_proto3 && field.cpp_type() != CppType::CPPTYPE_MESSAGE;

  switch (field.cpp_type()) {
#define HANDLE_CASE(_cpp_type_, _type_name_, _py_constructor_) \
    case CppType::_cpp_type_: \
      if (field.is_repeated()) { \
        int field_size = reflection.FieldSize(message, &field); \
        PyObject *result = PyList_New(field_size); \
        for (int i = 0; i < field_size; ++i) { \
          PyObject *item = _py_constructor_( \
              reflection.GetRepeated##_type_name_(message, &field, i)); \
          if (!item) { \
            Py_DECREF(result); /* Prevents a memory leak. */ \
            *error_pos = StrCat("[", i, "]", *error_pos); \
            return nullptr; \
          } \
          PyList_SetItem(result, i, item); \
        } \
        return result; \
      } else { \
        PyObject *result;\
        if (always_has || reflection.HasField(message, &field)) { \
          result = _py_constructor_(\
              reflection.Get##_type_name_(message, &field)); \
        } else { \
          result = Py_None; \
        } \
        if (!result) { \
          return nullptr; \
        } \
        return result; \
      }

      HANDLE_CASE(CPPTYPE_INT32, Int32, PyLong_FromLong);
      HANDLE_CASE(CPPTYPE_INT64, Int64, PyLong_FromLongLong);
      HANDLE_CASE(CPPTYPE_UINT32, UInt32, PyLong_FromUnsignedLong);
      HANDLE_CASE(CPPTYPE_UINT64, UInt64, PyLong_FromUnsignedLongLong);
      HANDLE_CASE(CPPTYPE_DOUBLE, Double, PyFloat_FromDouble);
      HANDLE_CASE(CPPTYPE_FLOAT, Float, PyFloat_FromDouble);
      HANDLE_CASE(CPPTYPE_BOOL, Bool, PyBool_FromLong);
      HANDLE_CASE(CPPTYPE_STRING, String, PyString_FromCPPString);
      HANDLE_CASE(CPPTYPE_ENUM, Enum, EnumValueToNameAndIndex);
      HANDLE_CASE(CPPTYPE_MESSAGE, Message, message_handler);
#undef HANDLE_CASE

    default:
      *error_pos = StrCat(": Unknown field type: ", field.cpp_type());
      return nullptr;
  }
}

PyObject* MessageToTree(
    const Message &message,
    string* error_pos) {
  PyObject* result = PyDict_New();
  const Reflection &reflection = *message.GetReflection();
  const Descriptor &descriptor = *message.GetDescriptor();

  bool is_proto3 = descriptor.file()->syntax() == FileDescriptor::SYNTAX_PROTO3;

  int field_count = descriptor.field_count();
  for (int i = 0; i < field_count; ++i) {
    const FieldDescriptor* field = descriptor.field(i);
    PyObject *field_value = FieldToTree(
        message, reflection, *field, is_proto3, error_pos);
    if (!field_value) {
      Py_DECREF(result);  // We no longer need 'result' due to the error.

      // We build up error_pos from back to front as the nullptrs propagate down
      // the stack in the event of an error.
      *error_pos = StrCat(".", field->name(), *error_pos);

      return nullptr;
    }
    PyDict_SetItemString(result, field->name().c_str(), field_value);

    // PyDict_SetItemString increments the reference counter, and we want to
    // give the dictionary ownership.
    if (field_value != Py_None) {  // None isn't ref-counted.
      Py_DECREF(field_value);
    }
  }

  int oneof_decl_count = descriptor.oneof_decl_count();
  for (int i = 0; i < oneof_decl_count; ++i) {
    const OneofDescriptor* oneof = descriptor.oneof_decl(i);

    const FieldDescriptor* field = reflection.GetOneofFieldDescriptor(
        message, oneof);
    if (field == nullptr) {
      PyDict_SetItemString(result, oneof->name().c_str(), Py_None);
    } else {
      PyObject *field_name = PyString_FromCPPString(field->name());
      PyDict_SetItemString(result, oneof->name().c_str(), field_name);
      Py_DECREF(field_name);
    }
  }

  return result;
}

class LoggingErrorCollector : public MultiFileErrorCollector {
 public:
  void AddError(const string& filename, int line, int column,
                const string& message) override {
    LOG(FATAL) << "Error parsing: " << filename << " at "
               << line << ":" << column << ", " << message;
  }

  void AddWarning(const string& filename, int line, int column,
                  const string& message) override {
    LOG(WARNING) << "Warning parsing: " << filename << " at "
                 << line << ":" << column << ", " << message;
  }
};

class MessagePrototypeFactory{
 public:
  MessagePrototypeFactory() :
    source_tree_(),
    error_collector_(),
    importer_(&source_tree_, &error_collector_),
    message_factory_() {}

  void MapPath(const string &virtual_path, const string &disk_path) {
    source_tree_.MapPath(virtual_path, disk_path);
  }

  void Import(const string &filename) {
    importer_.Import(filename);
  }

  const Message *GetPrototype(const string &message_type) {
    mutex_lock lock(mu_);
    if (prototype_cache_.count(message_type)) {
      return prototype_cache_.at(message_type);
    }

    const Descriptor *descriptor =
        importer_.pool()->FindMessageTypeByName(message_type);

    const Message *prototype = nullptr;
    if (descriptor) {
      prototype = message_factory_.GetPrototype(descriptor);
      prototype_cache_[message_type] = prototype;
    }
    return prototype;
  }

 private:
  DiskSourceTree source_tree_;
  LoggingErrorCollector error_collector_;
  Importer importer_;
  DynamicMessageFactory message_factory_;
  mutex mu_;
  std::map<string, const Message*> prototype_cache_;

  TF_DISALLOW_COPY_AND_ASSIGN(MessagePrototypeFactory);
};

std::unique_ptr<MessagePrototypeFactory> singleton_prototype_factory;

MessagePrototypeFactory *SingletonPrototypeFactory() {
  if (!singleton_prototype_factory) {
    singleton_prototype_factory.reset(new MessagePrototypeFactory());
  }
  return singleton_prototype_factory.get();
}

PyObject* MapProtoSourceTreePath(PyObject *self, PyObject *args) {
  const char *virtual_path;
  const char *disk_path;

  if (!PyArg_ParseTuple(args, "ss", &virtual_path, &disk_path)) {
    // PyArg_ParseTuple already sets an error for us.
    return nullptr;
  }
  LOG(INFO) << "Mapping Vitual Path '" << virtual_path
            << "' to disk_path '" << disk_path << "'";
  SingletonPrototypeFactory()->MapPath(virtual_path, disk_path);
  LOG(INFO) << "DONE Mapping Vitual Path '" << virtual_path
            << "' to disk_path '" << disk_path << "'";
  return Py_None;
}

PyObject* ImportProtoFile(PyObject *self, PyObject *args) {
  const char *filename;
  LOG(INFO) << "Importing Proto file: about to parse args.";

  if (!PyArg_ParseTuple(args, "s", &filename)) {
    // PyArg_ParseTuple already sets an error for us.
    return nullptr;
  }
  LOG(INFO) << "Importing Proto file " << filename;
  SingletonPrototypeFactory()->Import(filename);
  LOG(INFO) << "Done Importing Proto file " << filename;
  return Py_None;
}


// Converts a message into a tree of Python objects.
PyObject* SerializedMessageToTree(PyObject *self, PyObject *args) {
#define FAIL_VALUE_ERROR(...) \
    PyErr_SetString(PyExc_ValueError, StrCat(__VA_ARGS__).c_str()); \
    return nullptr;
  const char *message_type_cstr;
  const char *serialized_message_ptr;
  int serialized_message_len;
  if (!PyArg_ParseTuple(args, "ss#",
                        &message_type_cstr,
                        &serialized_message_ptr, &serialized_message_len)) {
    // PyArg_ParseTuple already sets an error for us.
    return nullptr;
  }
  string message_type(message_type_cstr);
  const Message *prototype =
      SingletonPrototypeFactory()->GetPrototype(message_type);
  if (!prototype) {
    FAIL_VALUE_ERROR("No descriptor for proto:", message_type,
                     "\nfound. Try Importing the .proto file.");
  }
  std::unique_ptr<Message> mutable_message(prototype->New());
  if (!mutable_message->ParseFromArray(
      serialized_message_ptr, serialized_message_len)) {
    FAIL_VALUE_ERROR("Failed to parse message of type:", message_type);
  }
  string error_pos;
  PyObject *result = MessageToTree(*mutable_message, &error_pos);
  if (result == nullptr) {
    FAIL_VALUE_ERROR("message_to_tree failed at  ROOT", error_pos);
  }
  return result;
#undef FAIL_VALUE_ERROR
}

PyMethodDef ProtoToolsMethods[] = {
  {"map_proto_source_tree_path", MapProtoSourceTreePath, METH_VARARGS,
    "Specifies a mapping from a virtual path to a UNIX file path "
    "from which to load .proto files."},
  {"import_proto_file", ImportProtoFile, METH_VARARGS,
    "Imports the .proto file on a given virtual path."},
  {"serialized_message_to_tree", SerializedMessageToTree, METH_VARARGS,
    "Turn a serialized proto message into a tree of Python objects. "
    "The proto definitions are set up using map_proto_source_tree_path and "
    "import_proto_file."},
  {nullptr, nullptr, 0, nullptr},  /* Sentinel. */
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "proto_tools",
  nullptr,
  0,
  ProtoToolsMethods,
  nullptr,
  nullptr,
  nullptr,
  nullptr
};
#endif

}  // namespace proto_tools
}  // namespace fold
}  // namespace tensorflow


PyMODINIT_FUNC
#if PY_MAJOR_VERSION >= 3
PyInit_proto_tools(void)
#else
initproto_tools(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
  PyObject* mod =  PyModule_Create(&tensorflow::fold::proto_tools::moduledef);
#else
  PyObject* mod = Py_InitModule(
      "proto_tools",
      tensorflow::fold::proto_tools::ProtoToolsMethods);
#endif
  if (mod == nullptr) {
    LOG(FATAL) << "Failed to build proto_tools python module.\n";
  }
#if PY_MAJOR_VERSION >= 3
  return mod;
#endif
}
