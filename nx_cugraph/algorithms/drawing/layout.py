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
    """
    `seed`, `distributed_action`, `weight`, `store_pos_as`, `node_mass`,
    `node_size` parameter is currently ignored.
    Only `dim=2` is supported.
    """
    if (N := len(G)) == 0:
        return {}

    # pos -> n_start and y_start
    if isinstance(pos, dict):
        ...

    # FIXME: MISSING ARGS:
    
    # distributed_action : bool (default: False)
    #     Distributes the attraction force evenly among nodes.
    
    # weight?
    
    # store_pos_as : str, default None
    #     If non-None, the position of each node will be stored on the graph as
    #     an attribute with this string as its name, which can be accessed with
    #     ``G.nodes[...][store_pos_as]``. The function still returns the dictionary.
    
    # node_mass : dict or None, optional
    #     Maps nodes to their masses, influencing the attraction to other nodes.
    
    # node_size : dict or None, optional
    #     Maps nodes to their sizes, preventing crowding by creating a halo effect.

    (x_axis, y_axis) = plc.force_atlas2(
        graph=G,
        max_iter=max_iter,
        lin_log_mode=linlog,
        prevent_overlapping=dissuade_hubs, # this might not be right
        jitter_tolerance=jitter_tolerance,
        scaling_ratio=scaling_ratio,
        strong_gravity_mode=strong_gravity,
        gravity=gravity,
    )
