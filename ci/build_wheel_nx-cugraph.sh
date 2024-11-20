#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

./ci/build_wheel.sh nx-cugraph .
./ci/validate_wheel.sh dist
