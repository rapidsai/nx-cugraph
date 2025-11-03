#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Support invoking test_python.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-package-name "conda_python" nx-cugraph --pure)")

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest nx-cugraph"
./ci/run_nx_cugraph_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-nx-cugraph.xml" \
  --cov=nx_cugraph \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/nx-cugraph-coverage.xml" \
  --cov-report=term

rapids-logger "pytest networkx using nx-cugraph backend"

pushd nx_cugraph
../run_nx_tests.sh --durations=10

# run_nx_tests.sh outputs coverage data, so check that total coverage is >0.0%
# in case nx-cugraph failed to load but fallback mode allowed the run to pass.
_coverage=$(coverage report | grep "^TOTAL")

echo "nx-cugraph coverage from networkx tests: $_coverage"
echo "$_coverage" | awk '{ if ($NF == "0.0%") exit 1 }'

# Ensure all algorithms were called by comparing covered lines to function lines.
# Run our tests again (they're fast enough) to add their coverage, then create coverage.json
NX_CUGRAPH_USE_COMPAT_GRAPHS=False pytest \
  --pyargs nx_cugraph \
  --config-file=../pyproject.toml \
  --cov-config=../pyproject.toml \
  --cov=nx_cugraph \
  --cov-append \
  --cov-report=

coverage report \
  --include="*/nx_cugraph/algorithms/*" \
  --omit=__init__.py \
  --show-missing \
  --rcfile=../pyproject.toml

coverage json --rcfile=../pyproject.toml

python -m nx_cugraph.tests.ensure_algos_covered

# Exercise (and show results of) scripts that show implemented networkx algorithms
python -m nx_cugraph.scripts.print_tree --dispatch-name --plc --incomplete --different
python -m nx_cugraph.scripts.print_table
popd

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}
