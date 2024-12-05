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
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import networkx_algorithm

__all__ = ["betweenness_centrality"]


@networkx_algorithm(
    name="bipartite_betweenness_centrality",
    version_added="24.12",
    _plc="betweenness_centrality",
)
def betweenness_centrality(G, nodes):
    G = _to_graph(G)

    node_ids, values = plc.betweenness_centrality(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(),
        k=None,
        random_state=None,
        normalized=False,
        include_endpoints=False,
        do_expensive_check=False,
    )
    top_node_ids = G._nodekeys_to_nodearray(set(nodes))
    bottom_node_ids = cp.delete(cp.arange(G._N, dtype=top_node_ids.dtype), top_node_ids)
    n = top_node_ids.size
    m = bottom_node_ids.size
    s, t = divmod(n - 1, m)
    bet_max_top = (
        ((m**2) * ((s + 1) ** 2))
        + (m * (s + 1) * (2 * t - s - 1))
        - (t * ((2 * s) - t + 3))
    ) / 2.0
    p, r = divmod(m - 1, n)
    bet_max_bot = (
        ((n**2) * ((p + 1) ** 2))
        + (n * (p + 1) * (2 * r - p - 1))
        - (r * ((2 * p) - r + 3))
    ) / 2.0

    values = values[cp.argsort(node_ids)]

    values[top_node_ids] /= bet_max_top
    values[bottom_node_ids] /= bet_max_bot

    return G._nodearray_to_dict(values)
