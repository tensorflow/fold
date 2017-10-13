# Experiments applying TensorFlow Fold to Stanford Sentiment Treebank data
# <nlp.stanford.edu/sentiment/treebank.html>

licenses(["notice"])  # Apache 2.0

load("//tensorflow_fold:fold.bzl", "fold_py_binary", "fold_py_library")

fold_py_library(
    name = "sentiment",
    srcs = ["sentiment.py"],
    deps = [
        # nltk
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/public:blocks",
    ],
)

fold_py_binary(
    name = "train",
    srcs = ["train.py"],
    deps = [
        ":sentiment",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "//tensorflow_fold/public:blocks",
    ],
)

fold_py_binary(
    name = "eval",
    srcs = ["eval.py"],
    deps = [
        ":sentiment",
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)

fold_py_binary(
    name = "filter_glove",
    srcs = ["filter_glove.py"],
    deps = [
        "@org_tensorflow//tensorflow:tensorflow_py",
    ],
)
