# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import networkx as nx
import pylibcugraph as plc

from nx_cugraph.convert import _to_undirected_graph
from nx_cugraph.utils import index_dtype, networkx_algorithm, not_implemented_for

__all__ = [
    "jaccard_coefficient",
]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@networkx_algorithm(version_added="25.02", _plc="jaccard_coefficients")
def jaccard_coefficient(G, ebunch=None):
    G = _to_undirected_graph(G)

    # If ebunch is not specified, create pairs representing all non-edges.
    # This can be an extremely large set and is not realistic for large graphs,
    # but this is required for NX compatibility.
    if ebunch is None:
        A = cp.tri(G._N, G._N, dtype=bool)
        A[G.src_indices, G.dst_indices] = True
        u_indices, v_indices = cp.nonzero(~A)
        if u_indices.size == 0:
            return iter([])
        u_indices = u_indices.astype(index_dtype)
        v_indices = v_indices.astype(index_dtype)

    else:
        (u, v) = zip(*ebunch)
        try:
            # Convert the ebunch lists to cupy arrays for passing to PLC, possibly
            # mapping to integers if the Graph was renumbered.
            # Allow the Graph renumber lookup (if renumbering was done) to check
            # for invalid node IDs in ebunch.
            u_indices = G._list_to_nodearray(u)
            v_indices = G._list_to_nodearray(v)
        except (KeyError, ValueError) as n:
            raise nx.NodeNotFound(f"Node {n} not in G.")

    (u, v, p) = plc.jaccard_coefficients(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        first=u_indices,
        second=v_indices,
        use_weight=False,
        do_expensive_check=False,
    )

    u = G._nodearray_to_list(u)
    v = G._nodearray_to_list(v)
    p = p.tolist()

    return zip(u, v, p)
