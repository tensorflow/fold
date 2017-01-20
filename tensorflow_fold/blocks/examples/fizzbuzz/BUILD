# Fizzbuzz example for TensorFlow Fold.

licenses(["notice"])  # Apache 2.0

load("//tensorflow_fold:fold.bzl", "fold_py_binary")

fold_py_binary(
    name = "fizzbuzz",
    srcs = ["fizzbuzz.py"],
    deps = [
        "//tensorflow_fold/public:blocks",
        "@six_archive//:six",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
