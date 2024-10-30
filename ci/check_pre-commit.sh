#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Install"
pip install --yes pre-commit

rapids-logger "Running pre-commit"
pre-commit install
pre-commit run --all-files
