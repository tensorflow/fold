# Calculator example for TensorFlow Fold.

licenses(["notice"])  # Apache 2.0

load("//tensorflow_fold:fold.bzl", "fold_py_binary", "fold_py_library")

fold_py_library(
    name = "model",
    srcs = ["model.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/public:blocks",
    ],
)

fold_py_binary(
    name = "train",
    srcs = ["train.py"],
    data = [
        "//tensorflow_fold/loom/calculator_example:calculator_proto_file",
    ],
    deps = [
        ":model",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/loom/calculator_example:calculator_py_pb2",
        # "//tensorflow_fold/loom/calculator_example:cpp_calculator_proto",
        "//tensorflow_fold/public:blocks",
        "//tensorflow_fold/util:proto",
    ],
)
