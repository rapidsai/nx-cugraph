# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import (
    _dtype_param,
    _get_float_dtype,
    networkx_algorithm,
    not_implemented_for,
)

__all__ = ["eigenvector_centrality"]


@not_implemented_for("multigraph")
@networkx_algorithm(
    extra_params=_dtype_param,
    is_incomplete=True,  # nstart not supported
    version_added="23.12",
    _plc="eigenvector_centrality",
)
def eigenvector_centrality(
    G, max_iter=100, tol=1.0e-6, nstart=None, weight=None, *, dtype=None
):
    """`nstart` parameter is not used, but it is checked for validity."""
    G = _to_graph(G, weight, 1, np.float32)
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept(
            "cannot compute centrality for the null graph"
        )
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    if nstart is not None:
        # Check if given nstart is valid even though we don't use it
        nstart = G._dict_to_nodearray(nstart, dtype=dtype)
        if (nstart == 0).all():
            raise nx.NetworkXError("initial vector cannot have all zero values")
        if nstart.sum() == 0:
            raise ZeroDivisionError
        # nstart /= total  # Uncomment (and assign total) when nstart is used below
    try:
        node_ids, values = plc.eigenvector_centrality(
            resource_handle=plc.ResourceHandle(),
            graph=G._get_plc_graph(weight, 1, dtype, store_transposed=True),
            epsilon=tol,
            max_iterations=max_iter,
            do_expensive_check=False,
        )
    except RuntimeError as exc:
        # Errors from PLC are sometimes a little scary and not very helpful
        raise nx.PowerIterationFailedConvergence(max_iter) from exc
    return G._nodearrays_to_dict(node_ids, values)
