workspace(name = "org_tensorflow_fold")

# There should be a symlink from /tensorflow to the local directory where
# tensorflow has been cloned from github.
local_repository(
  name = "org_tensorflow",
  path = "tensorflow",
)

# Import all of the tensorflow dependencies.
load('//tensorflow_fold:workspace.bzl', 'tf_fold_workspace')
tf_fold_workspace()
