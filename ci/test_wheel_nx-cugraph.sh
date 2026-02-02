#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -eoxu pipefail

source rapids-init-pip

package_name="nx-cugraph"
python_package_name=${package_name//-/_}

# nx-cugraph is a pure wheel, which is part of generating the download path
NX_CUGRAPH_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" "$package_name" --pure --cuda "$RAPIDS_CUDA_VERSION")")

# echo to expand wildcard before adding `[extra]` requires for pip
#
# '--extra-index-url pypi.nvidia.com' can be removed when 'pylibcugraph' and
# its dependencies are available from pypi.org
rapids-pip-retry install \
    --extra-index-url https://pypi.nvidia.com \
    "$(echo "${NX_CUGRAPH_WHEELHOUSE}"/"${python_package_name}"*.whl)[test]"

# Run smoke tests for aarch64 pull requests
arch=$(uname -m)
if [[ "${arch}" == "aarch64" && ${RAPIDS_BUILD_TYPE} == "pull-request" ]]; then
    python ./ci/wheel_smoke_test_${package_name}.py
else
    # --import-mode=append. See test_python.sh for details.
    # FIXME: Adding PY_IGNORE_IMPORTMISMATCH=1 to workaround conftest.py import
    # mismatch error seen by nx-cugraph after using pytest 8 and
    # --import-mode=append.
    RAPIDS_DATASET_ROOT_DIR=$(pwd)/datasets \
    PY_IGNORE_IMPORTMISMATCH=1 \
    NX_CUGRAPH_USE_COMPAT_GRAPHS=False \
    python -m pytest \
       -v \
       --import-mode=append \
       --benchmark-disable \
       ./"${python_package_name}"/tests
fi
