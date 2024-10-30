#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-date-string

rapids-print-env

rapids-generate-version > ./VERSION
export RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)

rapids-logger "Begin py build"

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly

# NOTE: nothing in nx-cugraph is CUDA-specific, but it is built on each CUDA
# platform to ensure it is included in each set of artifacts, since test
# scripts only install from one set of artifacts based on the CUDA version used
# for the test run.
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${RAPIDS_CONDA_BLD_OUTPUT_DIR}" \
  conda/recipes/nx-cugraph

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

# TODO: uncomment when we're ready to upload artifacts in CI
# rapids-upload-conda-to-s3 python