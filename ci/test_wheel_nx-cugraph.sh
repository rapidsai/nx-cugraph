#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -eoxu pipefail

package_name="nx-cugraph"
python_package_name=$(echo ${package_name}|sed 's/-/_/g')

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# nx-cugraph is a pure wheel, which is part of generating the download path
RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" RAPIDS_PY_WHEEL_PURE="1" rapids-download-wheels-from-s3 ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
    "$(echo ./dist/${python_package_name}*.whl)[test]"

# Run smoke tests for aarch64 pull requests
arch=$(uname -m)
if [[ "${arch}" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_${package_name}.py
else
    # --import-mode=append. See test_python.sh for details.
    # FIXME: Adding PY_IGNORE_IMPORTMISMATCH=1 to workaround conftest.py import
    # mismatch error seen by nx-cugraph after using pytest 8 and
    # --import-mode=append.
    RAPIDS_DATASET_ROOT_DIR=`pwd`/datasets \
    PY_IGNORE_IMPORTMISMATCH=1 \
    NX_CUGRAPH_USE_COMPAT_GRAPHS=False \
    python -m pytest \
       -v \
       --import-mode=append \
       --benchmark-disable \
       ./${python_package_name}/tests
fi
