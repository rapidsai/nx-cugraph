#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

./ci/build_wheel.sh nx-cugraph .
./ci/validate_wheel.sh "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"
