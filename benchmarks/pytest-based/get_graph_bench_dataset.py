# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""
Checks if a particular dataset has been downloaded inside the datasets dir
(RAPIDS_DATASET_ROOT_DIR). If not, the file will be downloaded using the
datasets API.

Positional Arguments:
    1) dataset name (e.g. 'email_Eu_core', 'cit-patents')
       available datasets can be found here:
        - `python/cugraph/cugraph/datasets/__init__.py`
"""

import sys

import cugraph.datasets as cgds

if __name__ == "__main__":
    # download and store dataset (csv) by using the Datasets API
    dataset = sys.argv[1].replace("-", "_")
    dataset_obj = getattr(cgds, dataset)

    if not dataset_obj.get_path().exists():
        dataset_obj.get_edgelist(download=True)
