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

import cugraph
import cupy as cp
import pylibcugraph as plc

from nx_cugraph.utils import _seed_to_int, networkx_algorithm

__all__ = [
    "forceatlas2_layout",
]


@networkx_algorithm(version_added="25.04", _plc="forceatlas2_layout")
def forceatlas2_layout(
    G,
    pos=None,
    *,
    max_iter=100,
    jitter_tolerance=1.0,
    scaling_ratio=2.0,
    gravity=1.0,
    distributed_action=False,
    strong_gravity=False,
    node_mass=None,
    node_size=None,
    weight=None,
    dissuade_hubs=False,
    linlog=False,
    seed=None,
    dim=2,
    store_pos_as=None,
):
    if (N := len(G)) == 0:
        return {}

    seed = _seed_to_int(seed)
    # parse optional pos positions
    if pos is None:
        # from nx.random_layout._process_params
        center = cp.zeros(dim)  # don't think we need center ever..?
        # broadcasting zeros onto an array is pointless in this alg
        # Then why does NetworkX do it?
        # pos = seed.rand(N, dim) + center
        pos_arr = seed.rand(N, dim) + center
    elif len(pos) == N:
        # same as nx just use cp?
        pos_arr = cp.array([pos[node].copy() for node in G])
    else:
        # set random node pos within the initial pos values
        pos_init = cp.array(list(pos.values()))
        max_pos = pos_init.max(axis=0)
        min_pos = pos_init.min(axis=0)
        dim = max_pos.size
        pos_arr = min_pos + seed.ran(N, dim) * (max_pos - min_pos)
        for idx, node in enumerate(G):
            if node in pos:
                pos_arr[idx] = pos[node].copy()

    mass = cp.zeros(N)
    size = cp.zeros(N)

    # Only adjust for size when the users specifies size other than default
    adjust_sizes = False
    if node_size is None:
        node_size = {}
    else:
        adjust_sizes = True

    if node_mass is None:
        node_mass = {}

    for idx, node in enumerate(G):
        mass[idx] = node_mass.get(node, G.degree(node) + 1)
        size[idx] = node_size.get(node, 1)

    # TODO: figure out if this is correct
    # missing args:
    # distributed_action
    # weight
    # dissuade_hubs
    # store_pos_as
    res = plc.force_atlas2(
        graph=G,
        max_iter=max_iter,
        jitter_tolerance=jitter_tolerance,
        lin_log_mode=linlog,
        strong_gravity_mode=strong_gravity,
        gravity=gravity,
        scaling_ratio=scaling_ratio,
    )
