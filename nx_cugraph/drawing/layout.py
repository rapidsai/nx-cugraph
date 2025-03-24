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

# this is to prevent ruff from complaining about newlines in docstrings
# the docstring should show up properly in the nx docs
# ruff: noqa: D205

import cupy as cp
import networkx as nx
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import (
    _dtype_param,
    _get_float_dtype,
    _seed_to_int,
    networkx_algorithm,
)

__all__ = [
    "forceatlas2_layout",
]


@networkx_algorithm(
    extra_params={
        "outbound_attraction_distribution : bool, default True": (
            "Distributes attraction along outbound edges. "
            "Hubs attract less and thus are pushed to the borders."
        ),
        **_dtype_param,
    },
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
    # nx_cugraph-only argument
    outbound_attraction_distribution=True,
    dtype=None,
):
    """
    `distributed_action`, `node_mass`, and `node_size` parameters are currently ignored.
    Only `dim=2` is supported.
    """
    if len(G) == 0:
        return {}

    if dim != 2:
        raise NotImplementedError(
            f"dim={dim} not supported; only dim=2 is currently supported"
        )

    # Split dict into cupy arrays of XY coords for PLC
    if pos is not None:
        if not isinstance(pos, dict):
            raise TypeError(f"pos must be dict or None; got {type(pos)}")

        # NOTE currently only x & y (dim=2) coordinated are supported by PLC
        #   greater dimensions should be supported in the future to align with nx
        start_pos_arr = cp.array(list(pos.values()))
        x_start = start_pos_arr[:, 0]
        y_start = start_pos_arr[:, 1]

    if weight is not None:
        # match the float dtype on input graph's edgelist, default to float32
        G = _to_graph(G, weight, 1, _get_float_dtype(dtype))
        dtype = _get_float_dtype(dtype, graph=G, weight=weight)
        G_plc = G._get_plc_graph(weight, 1, dtype)
    else:
        G = _to_graph(G)
        G_plc = G._get_plc_graph()

    seed = _seed_to_int(seed)

    vertices, x_axis, y_axis = plc.force_atlas2(
        plc.ResourceHandle(),
        random_state=seed,
        graph=G_plc,
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

    pos_arr = cp.column_stack((x_axis, y_axis))
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
