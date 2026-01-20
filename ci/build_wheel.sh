#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir=$1

source rapids-date-string
source rapids-init-pip

rapids-generate-version > ./VERSION

cd "${package_dir}"

RAPIDS_PIP_WHEEL_ARGS=(
  -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
  -v
  --no-deps
  --disable-pip-version-check
  --extra-index-url https://pypi.nvidia.com
)

# unset PIP_CONSTRAINT (set by rapids-init-pip)... it doesn't affect builds as of pip 25.3, and
# results in an error from 'pip wheel' when set and --build-constraint is also passed
unset PIP_CONSTRAINT
rapids-pip-retry wheel \
    "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
    .
