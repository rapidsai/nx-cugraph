# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from nx_cugraph import _nxver
from nx_cugraph.convert_matrix import to_scipy_sparse_array
from nx_cugraph.utils import networkx_algorithm

__all__ = ["adjacency_matrix"]


_adjacency_matrix_extra_params = None
if _nxver < (3, 7):
    _adjacency_matrix_extra_params = {
        "format : str, optional (default='csr')": (
            "The type of matrix to return. See `to_scipy_sparse_array` for "
            "supported values."
        )
    }


@networkx_algorithm(extra_params=_adjacency_matrix_extra_params, version_added="25.06")
def adjacency_matrix(G, nodelist=None, dtype=None, weight="weight", *, format="csr"):
    return to_scipy_sparse_array(
        G, nodelist=nodelist, dtype=dtype, weight=weight, format=format
    )
