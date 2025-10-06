#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir=$1

source rapids-date-string
source rapids-init-pip

rapids-generate-version > ./VERSION

cd "${package_dir}"

rapids-pip-retry wheel \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    -v \
    --no-deps \
    --disable-pip-version-check \
    --extra-index-url https://pypi.nvidia.com \
    .
