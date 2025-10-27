# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import networkx as nx

import nx_cugraph as nxcg
from nx_cugraph import _nxver

from ..utils import networkx_algorithm
from ._utils import (
    _common_small_graph,
    _complete_graph_indices,
    _ensure_int,
    _ensure_nonnegative_int,
)

__all__ = [
    "caveman_graph",
]


@networkx_algorithm(version_added="23.12")
def caveman_graph(l, k):  # noqa: E741
    l = _ensure_int(l)  # noqa: E741
    k = _ensure_int(k)
    N = _ensure_nonnegative_int(k * l)
    if l == 0 or k < 1:
        return _common_small_graph(N, None, None)
    k = _ensure_nonnegative_int(k)
    src_clique, dst_clique = _complete_graph_indices(k)
    src_cliques = [src_clique]
    dst_cliques = [dst_clique]
    src_cliques.extend(src_clique + i * k for i in range(1, l))
    dst_cliques.extend(dst_clique + i * k for i in range(1, l))
    src_indices = cp.hstack(src_cliques)
    dst_indices = cp.hstack(dst_cliques)
    use_compat_graph = _nxver < (3, 3) or nx.config.backends.cugraph.use_compat_graphs
    return nxcg.CudaGraph.from_coo(
        l * k, src_indices, dst_indices, use_compat_graph=use_compat_graph
    )
