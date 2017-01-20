#!/bin/bash
#
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Script for building a pip package.
#
# Based on tensorflow/tools/pip_package/build_pip_package.sh.
set -e

function main() {
  if [ $# -lt 1 ] ; then
    echo "No destination dir provided"
    exit 1
  fi

  DEST=$1
  TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXX)

  echo $(date) : "=== Using tmpdir: ${TMPDIR}"

  if [ ! -d bazel-bin/tensorflow_fold ]; then
    echo "Could not find bazel-bin.  Did you run from the root of the build tree?"
    exit 1
  fi

  cp -R \
    bazel-bin/tensorflow_fold/util/build_pip_package.runfiles/org_tensorflow_fold/tensorflow_fold \
    "${TMPDIR}"

  cp -f "${TMPDIR}"/tensorflow_fold/public/blocks.py "${TMPDIR}"/tensorflow_fold/__init__.py
  cp -f "${TMPDIR}"/tensorflow_fold/public/loom.py "${TMPDIR}"/tensorflow_fold/loom/__init__.py

  cp tensorflow_fold/util/setup.py ${TMPDIR}

  # Before we leave the top-level directory, make sure we know how to
  # call python.
  source tensorflow/tools/python_bin_path.sh

  pushd ${TMPDIR}
  echo $(date) : "=== Building wheel"
  "${PYTHON_BIN_PATH:-python}" setup.py bdist_wheel >/dev/null
  mkdir -p ${DEST}
  cp dist/* ${DEST}
  popd
  rm -rf ${TMPDIR}
  echo $(date) : "=== Output wheel file is in: ${DEST}"
}

main "$@"
