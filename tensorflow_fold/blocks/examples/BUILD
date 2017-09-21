package(
    default_visibility = [
        "//tensorflow_fold/blocks/examples:__subpackages__",
    ],
)

load("//tensorflow_fold:fold.bzl", "fold_py_binary")

fold_py_binary(
    name = "plan",
    srcs = ["plan.py"],
    deps = [
        "//tensorflow_fold/blocks:util",
        # numpy",
        "@org_tensorflow//tensorflow:tensorflow_py",
        "@org_tensorflow//tensorflow/python/debug:debug_py",
    ],
)
