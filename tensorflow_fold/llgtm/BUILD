# The Low-level Library for Gradients, Tensors, and Matrices (LLGTM)

licenses(["notice"])  # Apache 2.0

load(
    "//tensorflow_fold:fold.bzl",
    "fold_cc_library",
    "fold_cc_test",
    "fold_cuda_library",
    "fold_cuda_cc_test",
    "fold_fake_cc_binary",
    "fold_py_nocompile_test",
    "if_cuda",
)

package(
    default_visibility = [
        "//tensorflow_fold/llgtm:__subpackages__",
    ],
)

fold_cc_library(
    name = "platform_external",
    hdrs = [
        "platform/external.h",
        "platform/platform.h",
    ],
    defines = ["LLGTM_PLATFORM_EXTERNAL"],
    deps = [
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
    ],
)

fold_cc_library(
    name = "platform_google",
    hdrs = [
        "platform/google.h",
        "platform/platform.h",
    ],
    defines = ["LLGTM_PLATFORM_GOOGLE"],
    deps = [
        "//base",
    ],
)

# Open source build: Run with --define platform=external
config_setting(
    name = "platform_external_mode",
    values = {"define": "platform=external"},
)

config_setting(
    name = "platform_google_mode",
    values = {"define": "platform=google"},
)

fold_cc_library(
    name = "llgtm",
    srcs = [
        "device.cc",
        "gradients.cc",
        "graph.cc",
        "layers.cc",
        "tensor.cc",
        "tensor_opcodes.cc",
        "trainer.cc",
    ],
    hdrs = [
        "device.h",
        "dimensions.h",
        "gradients.h",
        "graph.h",
        "graph_evaluator.h",
        "graph_implementation.h",
        "layers.h",
        "tensor.h",
        "tensor_opcodes.h",
        "tensor_ops.h",
        "tensor_ops_impl.h",
        "trainer.h",
        "util.h",
        "variable_initializers.h",
    ],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
    ] + select({
        ":platform_external_mode": [":platform_external"],
        ":platform_google_mode": [":platform_google"],
        "//conditions:default": [":platform_external"],
    }),
)

fold_cuda_library(
    name = "llgtm_eigen_backend",
    srcs = [
        "backend/eigen_evaluator.cc",
        "backend/eigen_graph_implementation.cc",
    ],
    hdrs = [
        "backend/eigen_evaluator.h",
        "backend/eigen_graph_implementation.h",
    ],
    textual_hdrs = [
        "backend/llgtm_nodes.inc",
    ],
    deps = [
        ":llgtm",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//third_party/eigen3",
    ],
)

fold_cc_library(
    name = "llgtm_tf_backend",
    srcs = [
        "backend/tf_evaluator.cc",
    ],
    hdrs = [
        "backend/tf_evaluator.h",
    ],
    textual_hdrs = [
        "backend/tf_nodes.inc",
    ],
    deps = [
        ":llgtm",
        ":llgtm_eigen_backend",
        "@com_google_absl//absl/memory",
        "@org_tensorflow//tensorflow/c:c_api",
        "@org_tensorflow//tensorflow/c:c_api_internal",
        "@org_tensorflow//tensorflow/c/eager:runtime",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:core_cpu_internal",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:session_options",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
)

fold_cc_library(
    name = "llgtm_eigen",
    hdrs = [
        "backend/eigen_evaluator_client.h",
        "llgtm.h",
    ],
    deps = [
        ":llgtm",
        ":llgtm_eigen_backend",
    ],
)

fold_cc_library(
    name = "llgtm_tf",
    hdrs = [
        "backend/tf_evaluator_client.h",
        "llgtm.h",
    ],
    textual_hdrs = [
        "backend/llgtm_nodes.inc",
    ],
    deps = [
        ":llgtm",
        ":llgtm_tf_backend",
    ],
)

fold_cc_library(
    name = "test_framework",
    testonly = 1,
    hdrs = [
        "test/test_framework.h",
    ],
    deps = [
        ":llgtm",
        "@org_tensorflow//tensorflow/core:test",
    ],
)

fold_cc_test(
    name = "graph_test",
    size = "small",
    srcs = ["test/graph_test.cc"],
    deps = [
        ":llgtm_eigen",
        ":test_framework",
        "@com_google_absl//absl/memory",
    ],
)

fold_cc_test(
    name = "evaluator_test_eigen",
    size = "small",
    srcs = [
        "test/evaluator_test.h",
        "test/evaluator_test_eigen.cc",
    ],
    deps = [
        ":llgtm_eigen",
        ":test_framework",
    ],
)

fold_cuda_cc_test(
    name = "evaluator_test_eigen_gpu",
    size = "small",
    srcs = [
        "test/evaluator_test.h",
        "test/evaluator_test_eigen_gpu.cc",
    ],
    deps = [
        ":llgtm_eigen",
        ":test_framework",
    ],
)

fold_cc_test(
    name = "evaluator_test_tf",
    srcs = [
        "test/evaluator_test.h",
        "test/evaluator_test_tf.cc",
    ],
    deps = [
        ":llgtm_tf",
        ":test_framework",
    ],
)

fold_cc_test(
    name = "gradients_test_eigen",
    size = "small",
    srcs = [
        "test/gradients_test.h",
        "test/gradients_test_eigen.cc",
    ],
    deps = [
        ":llgtm_eigen",
        ":test_framework",
    ],
)

fold_cuda_cc_test(
    name = "gradients_test_eigen_gpu",
    size = "small",
    srcs = [
        "test/gradients_test.h",
        "test/gradients_test_eigen_gpu.cc",
    ],
    deps = [
        ":llgtm_eigen",
        ":test_framework",
    ],
)

fold_cc_test(
    name = "gradients_test_tf",
    srcs = [
        "test/gradients_test.h",
        "test/gradients_test_tf.cc",
    ],
    deps = [
        ":llgtm_tf",
        ":test_framework",
    ],
)

fold_fake_cc_binary(
    name = "graph_nocompile",
    srcs = ["test/graph_nocompile.cc"],
    deps = [
        ":llgtm",
        ":llgtm_eigen",
        ":llgtm_tf",
    ],
)

fold_py_nocompile_test(
    name = "graph_nocompile_test",
    size = "large",
    srcs = ["test/graph_nocompile_test.py"],
    data = ["graph_nocompile"],
    tags = ["non_compile_test"],
    deps = [],
)

test_suite(
    name = "cpu_tests",
    tests = [
        "evaluator_test_eigen",
        # "evaluator_test_tf",  # disabled due to tf visibility
        "gradients_test_eigen",
        # "gradients_test_tf",  # disabled due to tf visibility
        "graph_nocompile_test",
        "graph_test",
    ],
)

test_suite(
    name = "gpu_tests",
    tests = [
        "evaluator_test_eigen_gpu",
        "gradients_test_eigen_gpu",
    ],
)
