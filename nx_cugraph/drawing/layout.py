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
import numpy as np
import pylibcugraph as plc
from networkx.utils import create_random_state

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import (
    _dtype_param,
    _get_float_dtype,
    _seed_to_int,
    _update_cpu_gpu_graphs,
    index_dtype,
    networkx_algorithm,
)

__all__ = [
    "forceatlas2_layout",
]


@networkx_algorithm(
    extra_params=_dtype_param,
    is_incomplete=True,  # dim=2-only
    is_different=True,  # node_size handled differently, different RNG and results
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
    dtype=None,
):
    """Only `dim=2` is supported, and there may be minor numeric differences."""
    if len(G) == 0:
        return {}

    # Mutate original graph if store_pos_as is given.
    G_orig = G

    if dim != 2:
        raise NotImplementedError(
            f"dim={dim} not supported; only dim=2 is currently supported"
        )

    if weight is not None:
        # match the float dtype on input graph's edgelist, default to float32
        G = _to_graph(G, weight, 1, _get_float_dtype(dtype))
        dtype = _get_float_dtype(dtype, graph=G, weight=weight)
        G_plc = G._get_plc_graph(weight, 1, dtype)
    else:
        G = _to_graph(G)
        G_plc = G._get_plc_graph()

    # Split dict into cupy arrays of XY coords for PLC
    if pos is not None:
        # NOTE currently only x & y (dim=2) coordinated are supported by PLC
        #   greater dimensions should be supported in the future to align with nx
        start_pos_arr = G._dict_to_nodearray(
            pos, default=[np.nan] * dim, dtype=np.dtype((np.float32, dim))
        )

        # find, if there exists, the missing position values
        missing_vals = cp.isnan(start_pos_arr).all(axis=1)
        num_missing = int(cp.count_nonzero(missing_vals))

        # fill in with valid random range
        if num_missing:
            xy_min = cp.nanmin(start_pos_arr, axis=0)
            xy_max = cp.nanmax(start_pos_arr, axis=0)
            # random state from seed to fill missing coords is different from random
            # state used for PLC
            seed = create_random_state(seed)

            # fill missing gaps with valid random coords
            start_pos_arr[missing_vals] = xy_min + cp.asarray(
                seed.rand(num_missing, dim), dtype=np.float32
            ) * (xy_max - xy_min)

        start_vertices = cp.arange(G._N, dtype=index_dtype)
        x_start = start_pos_arr[:, 0]
        y_start = start_pos_arr[:, 1]
    else:
        start_vertices = None
        x_start = None
        y_start = None

    if node_size is not None:
        # nx:node_size and plc:vertex_radius are very similar, but are implemented
        # differently. For PLC, it's only used when `prevent_overlapping` is True,
        # but NetworkX doesn't have the equivalent of `prevent_overlapping`.
        # So, if we're given `node_size` argument, assume `prevent_overlapping=True`,
        # which should more closely match user expectations.
        vertex_radius_values = G._dict_to_nodearray(
            node_size, default=1.0, dtype=np.float32
        )
        vertex_radius_vertices = (
            start_vertices
            if start_vertices is not None
            else cp.arange(G._N, dtype=index_dtype)
        )
        prevent_overlapping = True
    else:
        vertex_radius_values = None
        vertex_radius_vertices = None
        prevent_overlapping = False

    if node_mass is not None:
        # Default mass is degree + 1
        vertex_mass_values = G._dict_to_nodearray(
            node_mass, default=np.nan, dtype=np.float32
        )
        isnan = cp.isnan(vertex_mass_values)
        if isnan.any():
            vertex_mass_values = cp.where(
                isnan, (G._degrees_array() + 1).astype(np.float32), vertex_mass_values
            )
        vertex_mass_vertices = (
            start_vertices
            if start_vertices is not None
            else cp.arange(G._N, dtype=index_dtype)
        )
    else:
        vertex_mass_values = None
        vertex_mass_vertices = None

    seed = _seed_to_int(seed)

    vertices, x_axis, y_axis = plc.force_atlas2(
        plc.ResourceHandle(),
        random_state=seed,
        graph=G_plc,
        max_iter=max_iter,
        start_vertices=start_vertices,
        x_start=x_start,
        y_start=y_start,
        outbound_attraction_distribution=distributed_action,
        lin_log_mode=linlog,
        prevent_overlapping=prevent_overlapping,
        vertex_radius_vertices=vertex_radius_vertices,
        vertex_radius_values=vertex_radius_values,
        overlap_scaling_ratio=100.0,
        edge_weight_influence=1,
        jitter_tolerance=jitter_tolerance,
        # We may want to expose barnes-hut--it's also surprising nx doesn't have it
        barnes_hut_optimize=False,
        barnes_hut_theta=0,
        scaling_ratio=scaling_ratio,
        strong_gravity_mode=strong_gravity,
        gravity=gravity,
        vertex_mobility_vertices=None,
        vertex_mobility_values=None,
        vertex_mass_vertices=vertex_mass_vertices,
        vertex_mass_values=vertex_mass_values,
        verbose=False,
        do_expensive_check=False,
    )

    pos_arr = cp.column_stack((x_axis, y_axis))
    pos = G._nodearrays_to_dict(
        node_ids=vertices, values=pos_arr, values_as_arrays=True
    )

    if store_pos_as is not None:

        def update_cpu(graph):
            nx.set_node_attributes(graph, pos, store_pos_as)

        update_pos_array = True

        def update_gpu(cuda_graph):
            # Ensure vertices are in order with their positions.
            # Use nonlocal variable to do this only once to ensure idempotency.
            nonlocal update_pos_array
            if update_pos_array:
                pos_arr[vertices] = pos_arr
                update_pos_array = False
            cuda_graph.node_values[store_pos_as] = pos_arr

        _update_cpu_gpu_graphs(G_orig, update_cpu=update_cpu, update_gpu=update_gpu)

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
    # nx_cugraph-only argument
    dtype=None,
):
    if dim != 2:
        return f"dim={dim} not supported; only dim=2 is currently supported"
    return True
