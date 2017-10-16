licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow_fold:fold.bzl",
    "fold_cc_library",
    "fold_proto_library",
    "fold_py_extension",
    "fold_py_library",
    "fold_py_test",
)

package(
    default_visibility = [
        "//visibility:public",
    ],
)

fold_proto_library(
    srcs = ["test.proto"],
    cc_name = "test_proto",
    py_name = "test_py_pb2",
)

fold_proto_library(
    srcs = ["test3.proto"],
    cc_name = "test3_proto",
    py_name = "test3_py_pb2",
)

filegroup(
    name = "test_proto_files",
    srcs = [
        "test.proto",
        "test3.proto",
    ],
)

# main() for C++ tests
fold_cc_library(
    name = "test_main",
    testonly = 1,
    srcs = ["test_main.cc"],
    linkopts = ["-lm"],
    deps = [
        "@com_google_googletest//:gtest",
        "@org_tensorflow//tensorflow/core:framework_lite",
        "@org_tensorflow//tensorflow/core:testlib",
    ],
)

fold_py_extension(
    name = "proto_tools",
    srcs = ["proto_tools.cc"],
    outs = ["proto_tools.so"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//util/python:python_headers",
        "@protobuf_archive//:protobuf",
    ],
)

fold_py_library(
    name = "proto",
    srcs = [],
    cc_deps = [":proto_tools"],
)

fold_py_test(
    name = "proto_test",
    srcs = ["proto_test.py"],
    data = [
        "test.proto",
        "test3.proto",
    ],
    deps = [
        ":proto",
        ":test3_py_pb2",
        ":test_py_pb2",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_library(
    name = "cpp_test_proto_lib",
    srcs = [],
    deps = [
        ":test3_py_pb2",
        ":test_py_pb2",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "//tensorflow_fold:headers",
        "//tensorflow_fold/public:blocks",
        "//tensorflow_fold/public:loom",
        "//tensorflow_fold/public:loom_cc",
    ],
)
