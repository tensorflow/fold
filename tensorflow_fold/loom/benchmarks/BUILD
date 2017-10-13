# Description:
# Bechmarks for the Loom library.

licenses(["notice"])  # Apache 2.0

load("//tensorflow_fold:fold.bzl", "fold_py_binary")

fold_py_binary(
    name = "iclr_2017_benchmark",
    srcs = ["iclr_2017_benchmark.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/public:loom",
    ],
)
