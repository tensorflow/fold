#!/usr/bin/env bash
# Based on tensorflow/tools/ci_build/builds/pip.sh.
#
# Build the Python PIP installation package for TensorFlow Fold.
#
# Usage:
#   pip.sh CONTAINER_TYPE [--mavx] [--mavx2]
#
# When executing the Python unit tests, the script obeys the shell
# variables: TF_BUILD_BAZEL_CLEAN.
#
# TF_BUILD_BAZEL_CLEAN, if set to any non-empty and non-0 value, directs the
# script to perform bazel clean prior to main build.

set -e

# Fixed naming patterns for wheel (.whl) files given different python versions
if [[ $(uname) == "Linux" ]]; then
  declare -A WHL_TAGS
  WHL_TAGS=(["2.7"]="cp27-none" ["3.4"]="cp34-cp34m" ["3.5"]="cp35-cp35m")
fi

source tensorflow/tensorflow/tools/ci_build/builds/builds_common.sh

# Get the command line arguments
CONTAINER_TYPE=$( echo "$1" | tr '[:upper:]' '[:lower:]' )

if [[ ! -z "${TF_BUILD_BAZEL_CLEAN}" ]] && \
   [[ "${TF_BUILD_BAZEL_CLEAN}" != "0" ]]; then
  echo "TF_BUILD_BAZEL_CLEAN=${TF_BUILD_BAZEL_CLEAN}: Performing 'bazel clean'"
  bazel clean
fi
MAVX_FLAG=""
while true; do
  if [[ "${1}" == "--mavx" ]]; then
    MAVX_FLAG="--copt=-mavx"
  elif [[ "${1}" == "--mavx2" ]]; then
    MAVX_FLAG="--copt=-mavx2"
  fi

  shift
  if [[ -z "${1}" ]]; then
    break
  fi
done

if [[ ! -z "${MAVX_FLAG}" ]]; then
  echo "Using MAVX flag: ${MAVX_FLAG}"
fi

PIP_BUILD_TARGET="//tensorflow_fold/util:build_pip_package"
GPU_FLAG=""
if [[ ${CONTAINER_TYPE} == "cpu" ]] || \
   [[ ${CONTAINER_TYPE} == "debian.jessie.cpu" ]]; then
  bazel build -c opt ${MAVX_FLAG} ${PIP_BUILD_TARGET} || \
      die "Build failed."
elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
  bazel build -c opt --config=cuda ${MAVX_FLAG} ${PIP_BUILD_TARGET} || \
      die "Build failed."
  GPU_FLAG="--gpu"
else
  die "Unrecognized container type: \"${CONTAINER_TYPE}\""
fi

MAC_FLAG=""
if [[ $(uname) == "Darwin" ]]; then
  MAC_FLAG="--mac"
fi


# If still in a virtualenv, deactivate it first
if [[ ! -z "$(which deactivate)" ]]; then
  echo "It appears that we are already in a virtualenv. Deactivating..."
  deactivate || die "FAILED: Unable to deactivate from existing virtualenv"
fi

# Obtain the path to Python binary
source tensorflow/tools/python_bin_path.sh

# Assume: PYTHON_BIN_PATH is exported by the script above
if [[ -z "$PYTHON_BIN_PATH" ]]; then
  die "PYTHON_BIN_PATH was not provided. Did you run configure?"
fi

# Determine the major and minor versions of Python being used (e.g., 2.7)
# This info will be useful for determining the directory of the local pip
# installation of Python
PY_MAJOR_MINOR_VER=$(${PYTHON_BIN_PATH} -V 2>&1 | awk '{print $NF}' | cut -d. -f-2)
if [[ -z "${PY_MAJOR_MINOR_VER}" ]]; then
  die "ERROR: Unable to determine the major.minor version of Python"
fi

echo "Python binary path to be used in PIP install: ${PYTHON_BIN_PATH} "\
"(Major.Minor version: ${PY_MAJOR_MINOR_VER})"

# Build PIP Wheel file
PIP_TEST_ROOT="pip_test"
PIP_WHL_DIR="${PIP_TEST_ROOT}/whl"
PIP_WHL_DIR=$(realpath ${PIP_WHL_DIR})  # Get absolute path
rm -rf ${PIP_WHL_DIR} && mkdir -p ${PIP_WHL_DIR}

bazel-bin/tensorflow_fold/util/build_pip_package ${PIP_WHL_DIR} ${GPU_FLAG} || \
    die "build_pip_package FAILED"

WHL_PATH=$(ls ${PIP_WHL_DIR}/tensorflow*.whl)
if [[ $(echo ${WHL_PATH} | wc -w) -ne 1 ]]; then
  die "ERROR: Failed to find exactly one built Fold .whl file in "\
"directory: ${PIP_WHL_DIR}"
fi

# Rename the whl file properly so it will have the python
# version tags and platform tags that won't cause pip install issues.
if [[ $(uname) == "Linux" ]]; then
  PY_TAGS=${WHL_TAGS[${PY_MAJOR_MINOR_VER}]}
  PLATFORM_TAG=$(to_lower "$(uname)_$(uname -m)")
# MAC has bash v3, which does not have associative array
elif [[ $(uname) == "Darwin" ]]; then
  if [[ ${PY_MAJOR_MINOR_VER} == "2.7" ]]; then
    PY_TAGS="py2-none"
  elif [[ ${PY_MAJOR_MINOR_VER} == "3.5" ]]; then
    PY_TAGS="py3-none"
  fi
  PLATFORM_TAG="any"
fi

WHL_DIR=$(dirname "${WHL_PATH}")
WHL_BASE_NAME=$(basename "${WHL_PATH}")

if [[ ! -z "${PY_TAGS}" ]]; then
  NEW_WHL_BASE_NAME=$(echo ${WHL_BASE_NAME} | cut -d \- -f 1)-\
$(echo ${WHL_BASE_NAME} | cut -d \- -f 2)-${PY_TAGS}-${PLATFORM_TAG}.whl

  if [[ ! -f "${WHL_DIR}/${NEW_WHL_BASE_NAME}" ]]; then
    cp "${WHL_DIR}/${WHL_BASE_NAME}" "${WHL_DIR}/${NEW_WHL_BASE_NAME}" && \
      echo "Copied wheel file: ${WHL_BASE_NAME} --> ${NEW_WHL_BASE_NAME}" || \
      die "ERROR: Failed to copy wheel file to ${NEW_WHL_BASE_NAME}"
  fi
fi

if [[ $(uname) == "Linux" ]]; then
  AUDITED_WHL_NAME="${WHL_DIR}/$(echo ${WHL_BASE_NAME} | sed "s/linux/manylinux1/")"

  # Repair the wheels for cpu manylinux1
  if [[ ${CONTAINER_TYPE} == "cpu" ]]; then
    echo "auditwheel repairing ${WHL_PATH}"
    auditwheel repair -w ${WHL_DIR} ${WHL_PATH}

    if [[ -f ${AUDITED_WHL_NAME} ]]; then
      WHL_PATH=${AUDITED_WHL_NAME}
      echo "Repaired manylinx1 wheel file at: ${WHL_PATH}"
    else
      die "ERROR: Cannot find repaired wheel."
    fi
  # Copy and rename for gpu manylinux as we do not want auditwheel to package in libcudart.so
  elif [[ ${CONTAINER_TYPE} == "gpu" ]]; then
    WHL_PATH=${AUDITED_WHL_NAME}
    cp ${WHL_DIR}/${WHL_BASE_NAME} ${WHL_PATH}
    echo "Copied manylinx1 wheel file at ${WHL_PATH}"
  fi
fi
