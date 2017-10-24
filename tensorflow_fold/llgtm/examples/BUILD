# Example models for LLGTM.

licenses(["notice"])  # Apache 2.0

cc_binary(
    name = "character_rnn",
    srcs = [
        "character_rnn.cc",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow_fold/llgtm:llgtm_eigen",
    ],
)

cc_binary(
    name = "tree_rnn",
    srcs = [
        "parsetree.h",
        "tree_rnn.cc",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "//tensorflow_fold/llgtm:llgtm_eigen",
    ],
)
