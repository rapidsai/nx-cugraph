#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

wheel_dir=${RAPIDS_WHEEL_BLD_OUTPUT_DIR}

./ci/build_wheel.sh nx-cugraph .
./ci/validate_wheel.sh "${wheel_dir}"
