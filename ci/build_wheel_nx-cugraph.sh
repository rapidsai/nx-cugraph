#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

# Only use stable ABI package for Python >= 3.11
if [[ "${RAPIDS_PY_VERSION}" != "3.10" ]]; then
  source ./ci/use_upstream_sabi_wheels.sh
fi

./ci/build_wheel.sh .
./ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PACKAGE_NAME="$(rapids-package-name wheel_python nx-cugraph --pure --cuda "$RAPIDS_CUDA_VERSION")"
export RAPIDS_PACKAGE_NAME
