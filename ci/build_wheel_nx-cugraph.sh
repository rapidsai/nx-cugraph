#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

#                   pkg-name   pkg-dir
./ci/build_wheel.sh nx-cugraph nx_cugraph
