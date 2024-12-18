# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
import pylibcugraph as plc
from networkx.utils import create_py_random_state

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import index_dtype, networkx_algorithm

__all__ = ["betweenness_centrality", "edge_betweenness_centrality"]


@networkx_algorithm(
    is_incomplete=True,  # weight not supported
    version_added="23.10",
    _plc="betweenness_centrality",
)
def betweenness_centrality(
    G, k=None, normalized=True, weight=None, endpoints=False, seed=None
):
    """`weight` parameter is not yet supported, and RNG with seed may be different."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of betweenness centrality not currently supported"
        )
    random_state = create_py_random_state(seed)
    G = _to_graph(G, weight)
    if k is not None and k < G._N:
        nodes = cp.array(random_state.sample(range(G._N), k), index_dtype)
    else:
        nodes = None
    node_ids, values = plc.betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=nodes,
        random_state=None,
        normalized=normalized,
        include_endpoints=endpoints,
        do_expensive_check=False,
    )
    return G._nodearrays_to_dict(node_ids, values)


@betweenness_centrality._can_run
def _(G, k=None, normalized=True, weight=None, endpoints=False, seed=None):
    return weight is None


@networkx_algorithm(
    is_incomplete=True,  # weight not supported
    version_added="23.10",
    _plc="edge_betweenness_centrality",
)
def edge_betweenness_centrality(G, k=None, normalized=True, weight=None, seed=None):
    """`weight` parameter is not yet supported, and RNG with seed may be different."""
    if weight is not None:
        raise NotImplementedError(
            "Weighted implementation of betweenness centrality not currently supported"
        )
    random_state = create_py_random_state(seed)
    G = _to_graph(G, weight)
    if k is not None and k < G._N:
        nodes = cp.array(random_state.sample(range(G._N), k), index_dtype)
    else:
        nodes = None
    src_ids, dst_ids, values, _edge_ids = plc.edge_betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=nodes,
        random_state=None,
        normalized=normalized,
        do_expensive_check=False,
    )
    if not G.is_directed():
        if nodes is not None:
            # Why is this necessary? What is going on?!
            # For undirected graphs, PLC only gives us data for one direction of the
            # edge (such as (0, 1), but not (1, 0)), but we don't know which one.
            dst_src = cp.hstack(
                (cp.vstack((dst_ids, src_ids)), cp.vstack((src_ids, dst_ids)))
            )
            indices = cp.lexsort(dst_src)
            dst_src = dst_src[:, indices][:, ::2]
            src_ids = dst_src[1]
            dst_ids = dst_src[0]
            values = values[indices % values.size].reshape(values.size, 2).sum(axis=-1)
        mask = src_ids <= dst_ids
        src_ids = src_ids[mask]
        dst_ids = dst_ids[mask]
        if nodes is not None:
            # Why is this necessary?
            values = (k / G._N) * values[mask]
        else:
            values = 2 * values[mask]
    elif nodes is not None:
        # Why is this necessary? Is plc or networkx "correct"?
        values *= k / G._N
    return G._edgearrays_to_dict(src_ids, dst_ids, values)


@edge_betweenness_centrality._can_run
def _(G, k=None, normalized=True, weight=None, seed=None):
    return weight is None
