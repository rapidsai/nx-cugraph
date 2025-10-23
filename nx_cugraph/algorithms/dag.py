# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import index_dtype, networkx_algorithm

__all__ = [
    "descendants",
    "ancestors",
]


def _ancestors_and_descendants(G, source, *, is_ancestors):
    G = _to_graph(G)
    if source not in G:
        hash(source)  # To raise TypeError if appropriate
        raise nx.NetworkXError(
            f"The node {source} is not in the {G.__class__.__name__.lower()}."
        )
    src_index = source if G.key_to_id is None else G.key_to_id[source]
    distances, predecessors, node_ids = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(switch_indices=is_ancestors),
        sources=cp.array([src_index], dtype=index_dtype),
        direction_optimizing=False,
        depth_limit=-1,
        compute_predecessors=False,
        do_expensive_check=False,
    )
    mask = (distances != np.iinfo(distances.dtype).max) & (distances != 0)
    return G._nodearray_to_set(node_ids[mask])


@networkx_algorithm(version_added="24.02", _plc="bfs")
def descendants(G, source):
    return _ancestors_and_descendants(G, source, is_ancestors=False)


@networkx_algorithm(version_added="24.02", _plc="bfs")
def ancestors(G, source):
    return _ancestors_and_descendants(G, source, is_ancestors=True)
