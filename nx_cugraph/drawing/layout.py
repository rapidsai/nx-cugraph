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

# ruff: noqa: D205

import cupy as cp
import networkx as nx
import pylibcugraph as plc

from nx_cugraph.utils import networkx_algorithm

__all__ = [
    "forceatlas2_layout",
]

outbound_attraction_distribution = {
    "outbound_attraction_distribution : bool, default ": (
        "Distributes attraction along outbound edges. "
        "Hubs attract less and thus are pushed to the borders."
    ),
}


@networkx_algorithm(
    extra_params=outbound_attraction_distribution,
    version_added="25.04",
    _plc="forceatlas2_layout",
)
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
    outbound_attraction_distribution=True,
):
    """
    `seed`, `distributed_action`, `weight`, `node_mass`, `node_size` parameter is
    currently ignored. Only `dim=2` is supported. `outbound_attraction_distribution` is
    a backend-only argument.
    """
    if len(G) == 0:
        return {}

    # Split dict into cupy arrays of XY coords for PLC
    if pos is not None:
        if not isinstance(pos, dict):
            raise TypeError(f"pos must be dict or None; got {type(pos)}")
        # NOTE currently only x & y (dim=2) coordinated are supported by PLC
        #   greater dimensions should be supported in the future to align with nx
        start_pos_arr = cp.array(list(pos.values()))
        x_start = start_pos_arr[:, 0]
        y_start = start_pos_arr[:, 1]

    vertices, x_axis, y_axis = plc.force_atlas2(
        plc.ResourceHandle(),
        graph=G,
        max_iter=max_iter,
        x_start=x_start,
        y_start=y_start,
        outbound_attraction_distribution=outbound_attraction_distribution,
        lin_log_mode=linlog,
        prevent_overlapping=dissuade_hubs,  # this might not be the right usage
        jitter_tolerance=jitter_tolerance,
        scaling_ratio=scaling_ratio,
        strong_gravity_mode=strong_gravity,
        gravity=gravity,
    )

    pos_arr = cp.vstack((x_axis, y_axis)).T
    pos = {int(vertices[i]): cp.asnumpy(pos_arr[i]) for i in range(vertices.shape[0])}

    if store_pos_as is not None:
        nx.set_node_attributes(G, pos, store_pos_as)

    return pos


@forceatlas2_layout._can_run
def _(
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
    outbound_attraction_distribution=True,
):
    if dim != 2:
        return f"dim={dim} not supported; only dim=2 is currently supported"
    return True
