# MNIST example for TensorFlow Fold.

licenses(["notice"])  # Apache 2.0

load("//tensorflow_fold:fold.bzl", "fold_py_binary")

fold_py_binary(
    name = "language_id",
    srcs = ["language_id.py"],
    deps = [
        "//tensorflow_fold/public:blocks",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
