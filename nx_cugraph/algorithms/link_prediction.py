# Copyright (c) 2025, NVIDIA CORPORATION.
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
        except KeyError as n:
            raise nx.NodeNotFound(f"Node {n} not in G.")

        # If G was not renumbered, then the ebunch nodes must be explicitly
        # checked. If not done, plc.jaccard_coefficients() will accept node IDs
        # not in the graph and return a coefficient of 0 for them, which is not
        # compatible with NX.
        if (not hasattr(G, "key_to_id") or G.key_to_id is None) and (
            (n := u_indices.max()) >= G._N
            or (n := v_indices.max()) >= G._N
            or (n := u_indices.min()) < 0
            or (n := v_indices.min()) < 0
        ):
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
