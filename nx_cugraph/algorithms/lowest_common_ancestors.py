# Copyright (c) 2024, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cupy as cp
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_directed_graph
from nx_cugraph.utils import (
    _groupby,
    index_dtype,
    networkx_algorithm,
    not_implemented_for,
)

__all__ = ["lowest_common_ancestor"]


@not_implemented_for("undirected")
@networkx_algorithm(is_incomplete=True, version_added="24.12", _plc="bfs")
def lowest_common_ancestor(G, node1, node2, default=None):
    """May not always raise NetworkXError for graphs that are not DAGs."""
    G = _to_directed_graph(G)

    # if not nxcg.is_directed_acyclic_graph(G):  # TODO
    #     raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")

    if G._N == 0:
        raise nx.NetworkXPointlessConcept("LCA meaningless on null graphs.")
    if node1 not in G:
        raise nx.NodeNotFound(
            f"Node(s) { {node1} } from pair {(node1, node2)} not in G."
        )
    if node2 not in G:
        raise nx.NodeNotFound(
            f"Node(s) { {node2} } from pair {(node1, node2)} not in G."
        )

    # Ancestor BFS from node1
    node1_index = node1 if G.key_to_id is None else G.key_to_id[node1]
    node2_index = node2 if G.key_to_id is None else G.key_to_id[node2]
    if node1_index == node2_index:  # Handle trivial case
        return node1
    plc_graph = G._get_plc_graph(switch_indices=True)
    distances1, predecessors1, node_ids1 = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=plc_graph,
        sources=cp.array([node1_index], index_dtype),
        direction_optimizing=False,  # True for undirected only
        depth_limit=-1,
        compute_predecessors=False,
        do_expensive_check=False,
    )
    mask1 = distances1 != np.iinfo(distances1.dtype).max
    node_ids1 = node_ids1[mask1]

    # Ancestor BFS from node2
    distances2, predecessors2, node_ids2 = plc.bfs(
        handle=plc.ResourceHandle(),
        graph=plc_graph,
        sources=cp.array([node2_index], index_dtype),
        direction_optimizing=False,  # True for undirected only
        depth_limit=-1,
        compute_predecessors=False,
        do_expensive_check=False,
    )
    mask2 = distances2 != np.iinfo(distances2.dtype).max
    node_ids2 = node_ids2[mask2]

    # Find all common ancestors
    common_ids = cp.intersect1d(node_ids1, node_ids2, assume_unique=True)
    if common_ids.size == 0:
        return default
    if common_ids.size == 1:
        # Only one; it must be the lowest common ancestor
        node_index = common_ids[0].tolist()
        return node_index if G.key_to_id is None else G.id_to_key[node_index]

    # Find nodes from `common_ids` that have no predecessors from `common_ids`.
    # TODO: create utility functions for getting neighbors, predecessors,
    # and successors of nodes, which may simplify this code.
    mask = cp.isin(G.src_indices, common_ids) & (G.src_indices != G.dst_indices)
    groups = _groupby(G.src_indices[mask], G.dst_indices[mask])
    # Walk along successors until we reach a lowest common ancestor
    stack = sorted(groups, reverse=True)  # Sort for consistency
    seen = set()
    while stack:
        node_index = stack.pop()
        if node_index in seen:
            continue
        seen.add(node_index)
        successors = groups[node_index]
        lower_ancestors = successors[cp.isin(successors, common_ids)]
        if lower_ancestors.size == 0:
            return node_index if G.key_to_id is None else G.id_to_key[node_index]
        stack.extend(sorted(lower_ancestors.tolist(), reverse=True))

    raise nx.NetworkXError("LCA only defined on directed acyclic graphs.")
