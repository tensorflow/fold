# Description:
# Bechmarks for the Loom library.

load("//tensorflow_fold:fold.bzl", "fold_py_binary")

fold_py_binary(
    name = "iclr_2017_benchmark",
    srcs = ["iclr_2017_benchmark.py"],
    deps = [
        "//tensorflow_fold/public:loom",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
