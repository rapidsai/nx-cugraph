#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-date-string

rapids-print-env

# TODO: revert this once we start publishing nightly packages from the
#       'nx-cugraph' repo and stop publishing them from the 'cugraph' repo
# rapids-generate-version > ./VERSION
echo "24.12.00a1000" > ./VERSION

rapids-logger "Begin py build"

# TODO: Remove `--no-test` flags once importing on a CPU
# node works correctly

# NOTE: nothing in nx-cugraph is CUDA-specific, but it is built on each CUDA
# platform to ensure it is included in each set of artifacts, since test
# scripts only install from one set of artifacts based on the CUDA version used
# for the test run.
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  conda/recipes/nx-cugraph

rapids-upload-conda-to-s3 python
