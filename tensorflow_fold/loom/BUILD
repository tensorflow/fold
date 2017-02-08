# Loom enables TensorFlow to be easily and efficiently used to train
# networks whose shape differs on a per training example basis.  This allows
# TensorFlow to handle architectures including Tree RNNs, high dimensional
# sparse variants of GridLSTM and arbitrary architectures containing multiple
# LSTMs of example-dependant depth.  Loom is implemented as a library on
# top of TensorFlow.

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow_fold:fold.bzl",
    "fold_cc_library",
    "fold_cc_test",
    "fold_proto_library",
    "fold_py_binary",
    "fold_py_library",
    "fold_py_test",
    "fold_py_wrap_cc",
    "fold_tf_op_py",
)

package(
    default_visibility = [
        "//tensorflow_fold/public:__subpackages__",
    ],
)

fold_proto_library(
    srcs = ["loom.proto"],
    cc_deps = [
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
    cc_name = "loom_proto",
    py_deps = [
        "@org_tensorflow//tensorflow/core:protos_all_py",
    ],
    py_name = "loom_py_pb2",
)

fold_cc_library(
    name = "platform",
    hdrs = ["platform.h"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

fold_cc_library(
    name = "weaver",
    srcs = ["weaver.cc"],
    hdrs = ["weaver.h"],
    deps = [
        ":loom_proto",
        ":platform",
        "@org_tensorflow//third_party/eigen3",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

fold_cc_test(
    name = "weaver_test",
    srcs = ["weaver_test.cc"],
    deps = [
        ":loom_proto",
        ":weaver",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)

fold_py_wrap_cc(
    name = "pywrapweaver",
    srcs = ["python/weaver.swig"],
    deps = [
        ":platform",
        ":weaver",
    ],
)

fold_cc_library(
    name = "weaver_op_base",
    srcs = ["weaver_op_base.cc"],
    hdrs = ["weaver_op_base.h"],
    deps = [
        ":weaver",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
)

fold_py_library(
    name = "weaver_op_base_py",
    srcs = ["weaver_op_base.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

#  The following two BUILD rules are an example of how you would use
#  weaver_op_base to define an op.

fold_cc_library(
    name = "deserializing_weaver_op_cc",
    srcs = ["deserializing_weaver_op.cc"],
    deps = [
        ":weaver",
        ":weaver_op_base",
        "@org_tensorflow//tensorflow/core:framework_headers_lib",
    ],
    alwayslink = 1,
)

fold_tf_op_py(
    name = "deserializing_weaver_op",
    srcs = ["deserializing_weaver_op.py"],
    cc_deps = [":deserializing_weaver_op_cc"],
    py_deps = [
        ":weaver_op_base_py",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_library(
    name = "loom",
    srcs = ["loom.py"],
    deps = [
        ":deserializing_weaver_op",
        ":loom_py_pb2",
        ":pywrapweaver",
        # numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_test(
    name = "loom_test",
    srcs = ["loom_test.py"],
    deps = [
        ":loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
