# MNIST example for TensorFlow Fold.

licenses(["notice"])  # Apache 2.0

load("//tensorflow_fold:fold.bzl", "fold_py_binary")

fold_py_binary(
    name = "mnist",
    srcs = ["mnist.py"],
    deps = [
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/blocks/examples:plan",
        "//tensorflow_fold/public:blocks",
    ],
)
