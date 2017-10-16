# Tree RNN smoke test (learn a calculator.)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow_fold:fold.bzl",
    "fold_proto_library",
    "fold_py_binary",
    "fold_py_extension",
    "fold_py_library",
    "fold_py_test",
)

package(
    default_visibility = [
        "//tensorflow_fold:__subpackages__",
    ],
)

fold_proto_library(
    srcs = ["calculator.proto"],
    cc_name = "calculator_proto",
    py_name = "calculator_py_pb2",
)

filegroup(
    name = "calculator_proto_file",
    srcs = ["calculator.proto"],
)

fold_py_extension(
    name = "cpp_calculator_proto",
    srcs = [],
    outs = [],
    deps = [":calculator_proto"],
)

fold_py_library(
    name = "calculator",
    srcs = ["calculator.py"],
    deps = [
        ":calculator_py_pb2",
    ],
)

fold_py_test(
    name = "calculator_test",
    srcs = ["calculator_test.py"],
    deps = [
        ":calculator",
        ":calculator_py_pb2",
        "@protobuf_archive//:protobuf_python",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_binary(
    name = "make_dataset",
    srcs = ["make_dataset.py"],
    deps = [
        ":calculator",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        ":calculator",
        ":calculator_py_pb2",
        # numpy",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/public:loom",
    ],
)

fold_py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    deps = [
        ":calculator",
        ":model",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_library(
    name = "helpers",
    srcs = ["helpers.py"],
    deps = [
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":calculator_py_pb2",
        ":helpers",
        ":model",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":calculator_py_pb2",
        ":helpers",
        ":model",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
