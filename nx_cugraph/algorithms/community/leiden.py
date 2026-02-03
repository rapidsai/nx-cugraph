# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_undirected_graph
from nx_cugraph.utils import (
    _dtype_param,
    _get_float_dtype,
    _groupby,
    _seed_to_int,
    networkx_algorithm,
    not_implemented_for,
)

__all__ = ["leiden_communities"]


@not_implemented_for("directed")
@networkx_algorithm(extra_params=_dtype_param, version_added="25.02", _plc="leiden")
def leiden_communities(
    G, weight="weight", resolution=1, max_level=None, seed=None, *, dtype=None
):
    # Warning: this API is experimental and may change. It is not yet in NetworkX.
    # See: https://github.com/networkx/networkx/pull/7743
    seed = _seed_to_int(seed)
    G = _to_undirected_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    if max_level is None or max_level < 0:
        max_level = 500
    node_ids, clusters, modularity = plc.leiden(
        resource_handle=plc.ResourceHandle(),
        random_state=seed,
        graph=G._get_plc_graph(weight, 1, dtype),
        max_level=max_level,
        resolution=resolution,
        theta=1,  # TODO: expose theta as a backend-only parameter once it's used
        do_expensive_check=False,
    )
    groups = _groupby(clusters, node_ids, groups_are_canonical=True)
    return [set(G._nodearray_to_list(ids)) for ids in groups.values()]
