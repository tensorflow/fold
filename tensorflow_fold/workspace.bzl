# TensorFlow Fold external dependencies that can be loaded in WORKSPACE
# files.

load('@org_tensorflow//tensorflow:workspace.bzl', 'tf_workspace')

# All TensorFlow Fold external dependencies.
# workspace_dir is the absolute path to the TensorFlow Fold repo. If linked
# as a submodule, it'll likely be '__workspace_dir__ + "/fold"'
def tf_fold_workspace():
  tf_workspace(tf_repo_name = "org_tensorflow")

  # ===== gRPC dependencies =====
  native.bind(
    name = "libssl",
    actual = "@boringssl//:ssl",
  )

  native.bind(
      name = "zlib",
      actual = "@zlib_archive//:zlib",
  )

  native.bind(
      name = "gmock",
      actual = "@gmock_archive//:gmock",
  )
