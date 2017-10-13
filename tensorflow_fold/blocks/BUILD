# TensorFlow Fold is a high-level library for constructing models that is built
# on top of TensorLoom.

licenses(["notice"])  # Apache 2.0

load("//tensorflow_fold:fold.bzl", "fold_py_library", "fold_py_test")

package(
    default_visibility = [
        "//tensorflow_fold/blocks/examples:__subpackages__",
        "//tensorflow_fold/public:__subpackages__",
    ],
)

fold_py_library(
    name = "blocks",
    srcs = [
        "block_compiler.py",
        "blocks.py",
        "layers.py",
        "loom_ops.py",
        "metrics.py",
    ],
    cc_deps = [
        "//tensorflow_fold/util:proto_tools",
    ],
    deps = [
        ":result_types",
        ":util",
        # numpy",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/public:loom",
    ],
)

fold_py_library(
    name = "test_lib",
    testonly = 1,
    srcs = [
        "test_lib.py",
    ],
    cc_deps = [
        "//tensorflow_fold/util:proto_tools",
    ],
    data = [
        "//tensorflow_fold/util:test_proto_files",
    ],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_test(
    name = "blocks_test",
    srcs = ["blocks_test.py"],
    data = [
        "//tensorflow_fold/util:test_proto_files",
    ],
    deps = [
        ":blocks",
        ":test_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/util:cpp_test_proto_lib",
        "//tensorflow_fold/util:test_py_pb2",
    ],
)

fold_py_test(
    name = "block_compiler_test",
    srcs = ["block_compiler_test.py"],
    deps = [
        ":blocks",
        ":test_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_test(
    name = "layers_test",
    srcs = ["layers_test.py"],
    deps = [
        ":blocks",
        ":test_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [
        ":blocks",
        ":test_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_library(
    name = "result_types",
    srcs = ["result_types.py"],
    deps = [
        # numpy",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/public:loom",
    ],
)

fold_py_test(
    name = "result_types_test",
    srcs = ["result_types_test.py"],
    deps = [
        ":result_types",
        ":test_lib",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_library(
    name = "util",
    srcs = ["util.py"],
    deps = [
        # numpy",
    ],
)

fold_py_test(
    name = "util_test",
    srcs = ["util_test.py"],
    deps = [
        ":test_lib",
        ":util",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_library(
    name = "plan",
    srcs = ["plan.py"],
    deps = [
        ":util",
        # numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/debug:debug_py",
    ],
)

fold_py_test(
    name = "plan_test",
    srcs = ["plan_test.py"],
    deps = [
        ":blocks",
        ":plan",
        ":test_lib",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
