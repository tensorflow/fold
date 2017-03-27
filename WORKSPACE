workspace(name = "org_tensorflow_fold")

# There should be a symlink from /tensorflow to the local directory where
# tensorflow has been cloned from github.
local_repository(
  name = "org_tensorflow",
  path = "tensorflow",
)

# TensorFlow depends on "io_bazel_rules_closure" so we need this here.
# Needs to be kept in sync with the same target in TensorFlow's WORKSPACE file.
http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "60fc6977908f999b23ca65698c2bb70213403824a84f7904310b6000d78be9ce",
    strip_prefix = "rules_closure-5ca1dab6df9ad02050f7ba4e816407f88690cf7d",
    urls = [
        "http://bazel-mirror.storage.googleapis.com/github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",  # 2017-02-03
        "https://github.com/bazelbuild/rules_closure/archive/5ca1dab6df9ad02050f7ba4e816407f88690cf7d.tar.gz",
    ],
)

# Import all of the tensorflow dependencies.
load('//tensorflow_fold:workspace.bzl', 'tf_fold_workspace')
tf_fold_workspace()
