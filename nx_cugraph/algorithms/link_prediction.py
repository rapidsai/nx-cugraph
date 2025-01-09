import cupy as cp

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
import networkx as nx
import pylibcugraph as plc

from nx_cugraph.convert import _to_undirected_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = [
    "jaccard_coefficient",
]


@not_implemented_for("directed")
@not_implemented_for("multigraph")
@networkx_algorithm(version_added="25.02", _plc="jaccard_coefficients")
def jaccard_coefficient(G, ebunch=None):
    """
    Parameters
    ----------
    G : graph
        A NetworkX or nx-cugraph undirected graph.

    ebunch : iterable of node pairs, optional (default = None)
        Jaccard coefficient will be computed for each pair of nodes
        given in the iterable. The pairs must be given as 2-tuples
        (u, v) where u and v are nodes in the graph. If ebunch is None
        then all nonexistent edges in the graph will be used.

    Returns
    -------
    piter : iterator
        An iterator of 3-tuples in the form (u, v, p) where (u, v) is a
        pair of nodes and p is their Jaccard coefficient.

    Raises
    ------
    NetworkXNotImplemented
        If `G` is a `DiGraph`, a `Multigraph` or a `MultiDiGraph`.

    NodeNotFound
        If `ebunch` has a node that is not in `G`.
    """
    if ebunch is None:
        # FIXME: is there a more efficient way to do this (on GPU or
        # otherwise)?
        ebunch = list(nx.non_edges(G))
        if not ebunch:
            return iter([])

    G = _to_undirected_graph(G)

    (u, v) = zip(*ebunch)
    try:
        # Convert the ebunch lists to cupy arrays for passing to PLC, possibly
        # mapping to integers if the Graph was renumbered on creation.
        # Allow the Graph renumber lookup (if renumbering was done) to check
        # for invalid node IDs in ebunch.
        u = G._list_to_nodearray(u)
        v = G._list_to_nodearray(v)
    except KeyError as n:
        raise nx.NodeNotFound(f"Node {n} not in G.")
    else:
        # If G was not renumbered, then the ebunch nodes must be explicitly
        # checked.
        if not hasattr(G, "key_to_id") or G.key_to_id is None:
            ebunch_nodes = cp.unique(cp.concatenate([u, v]))
            graph_nodes = cp.unique(cp.concatenate([G.src_indices, G.dst_indices]))
            if invalid := cp.setdiff1d(ebunch_nodes, graph_nodes, assume_unique=True):
                raise nx.NodeNotFound(f"Node {invalid.tolist()[0]} not in G.")

    # Note that Jaccard similarity must run on a symmetric graph.
    # FIXME: PLC will symmetrize the graph if told to, but the symmetrize flag
    # to _get_plc_graph() does other things (cast to 64bit, etc.). Can we let
    # PLC do the symmetrization if the symmetrize flag is set instead?
    (u, v, p) = plc.jaccard_coefficients(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(symmetrize=None),
        first=u,
        second=v,
        use_weight=False,
        do_expensive_check=False,
    )

    u = G._nodearray_to_list(u)
    v = G._nodearray_to_list(v)
    p = p.tolist()

    # zip() returns a zip object, which is different than default NX but should
    # still be valid since it's an iterator of 3-tuples.
    return zip(u, v, p)


# @jaccard_coefficient._can_run is always True (default)

# @jaccard_coefficient._should_run is always True (default)
