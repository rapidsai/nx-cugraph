#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

# TODO: uncomment when we're ready to begin uploading artifacts in CI
# rapids-configure-conda-channels

# configuring sccache shouldn't be necessary in a pure-Python project build
# source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

rapids-generate-version > ./VERSION
export RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)

rapids-logger "Begin py build"

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/pylibcugraph

rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/cugraph

# NOTE: nothing in nx-cugraph is CUDA-specific, but it is built on each CUDA
# platform to ensure it is included in each set of artifacts, since test
# scripts only install from one set of artifacts based on the CUDA version used
# for the test run.
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/nx-cugraph

# removed mambabuild for cugraph-service

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

# removed mambabuild steps for conda

# removed mambabuild for cugraph equivariant

# TODO: uncomment when we're ready to upload artifacts in CI
# rapids-upload-conda-to-s3 python
