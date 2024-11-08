#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support invoking run_nx_cugraph_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/..

# Only be verbose and display print information for the first pytest command here
NX_CUGRAPH_USE_COMPAT_GRAPHS=False pytest --capture=no --verbose --cache-clear --benchmark-disable "$@" ./nx_cugraph/tests
NX_CUGRAPH_USE_COMPAT_GRAPHS=True pytest --cache-clear --benchmark-disable "$@" ./nx_cugraph/tests
