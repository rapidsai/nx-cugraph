# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from nx_cugraph.convert_matrix import to_scipy_sparse_array
from nx_cugraph.utils import networkx_algorithm

__all__ = ["adjacency_matrix"]


@networkx_algorithm(version_added="25.06")
def adjacency_matrix(G, nodelist=None, dtype=None, weight="weight"):
    return to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight)
