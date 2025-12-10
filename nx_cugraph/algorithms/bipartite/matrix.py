# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import networkx as nx
import numpy as np

from nx_cugraph.convert import _to_graph
from nx_cugraph.generators._utils import _create_using_class
from nx_cugraph.utils import index_dtype, networkx_algorithm

__all__ = ["biadjacency_matrix", "from_biadjacency_matrix"]


@networkx_algorithm(version_added="25.06")
def biadjacency_matrix(
    G, row_order, column_order=None, dtype=None, weight="weight", format="csr"
):
    import scipy as sp

    G = _to_graph(G, weight, 1, dtype)

    nrows = len(row_order)
    if nrows == 0:
        raise nx.NetworkXError("row_order is empty list")
    if len(row_order) != len(set(row_order)):
        raise nx.NetworkXError("Ambiguous ordering: `row_order` contained duplicates.")
    if column_order is None:
        column_order = list(set(G) - set(row_order))
    ncols = len(column_order)
    if len(column_order) != len(set(column_order)):
        raise nx.NetworkXError(
            "Ambiguous ordering: `column_order` contained duplicates."
        )

    if G.src_indices.size == 0:
        src_indices = dst_indices = edge_array = ()
    else:
        row_ids = G._nodekeys_to_nodearray(row_order)
        col_ids = G._nodekeys_to_nodearray(column_order)

        # Using `mapper` like this is a useful trick that may not be obvious.
        # This is also done in `to_scipy_sparse_array`.
        mapper = cp.empty(G._N, dtype=index_dtype)
        mapper[:] = -1  # Indicate nodes to exclude
        mapper[row_ids] = cp.arange(row_ids.size, dtype=index_dtype)
        src_indices = mapper[G.src_indices]

        mapper[:] = -1  # Indicate nodes to exclude
        mapper[col_ids] = cp.arange(col_ids.size, dtype=index_dtype)
        dst_indices = mapper[G.dst_indices]
        mask = (src_indices != -1) & (dst_indices != -1)
        src_indices = src_indices[mask]
        if src_indices.size == 0:
            src_indices = dst_indices = edge_array = ()
        else:
            dst_indices = dst_indices[mask]

            src_indices = cp.asnumpy(src_indices)
            dst_indices = cp.asnumpy(dst_indices)
            if weight in G.edge_values:
                edge_array = G.edge_values[weight]
                if weight in G.edge_masks:
                    edge_array = cp.where(G.edge_masks[weight], edge_array, 1)
                edge_array = edge_array[mask]
                edge_array = cp.asnumpy(edge_array)
            else:
                edge_array = np.repeat(1, src_indices.size)

    # PERF: convert to desired sparse format on GPU before copying to CPU
    A = sp.sparse.coo_array(
        (edge_array, (src_indices, dst_indices)), shape=(nrows, ncols), dtype=dtype
    )
    try:
        return A.asformat(format)
    except ValueError as exc:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {format}") from exc


@networkx_algorithm(version_added="25.06", fallback=True, create_using_arg=1)
def from_biadjacency_matrix(
    A, create_using=None, edge_attribute="weight", *, row_order=None, column_order=None
):
    if row_order is not None or column_order is not None:
        raise NotImplementedError(
            f"row_order={row_order} and column_order={column_order} not supported;"
            " only row_order=None and column_order=None are currently supported"
        )
    graph_class, inplace = _create_using_class(create_using)
    nrows, ncols = A.shape
    if A.format != "coo":
        A = A.tocoo()
    if A.dtype.kind in {"i", "u"} and graph_class.is_multigraph():
        src_indices = cp.array(np.repeat(A.row, A.data), index_dtype)
        dst_indices = cp.array(np.repeat(A.col, A.data), index_dtype)
        size = src_indices.size
        if not graph_class.is_directed():
            size *= 2
        weight = cp.empty(size, A.data.dtype)
        weight[:] = 1
    else:
        src_indices = cp.array(A.row, index_dtype)
        dst_indices = cp.array(A.col, index_dtype)
        weight = cp.array(A.data)

    dst_indices += nrows
    if not graph_class.is_directed():
        # Symmetrize if undirected
        src_indices, dst_indices = (
            cp.hstack([src_indices, dst_indices]),
            cp.hstack([dst_indices, src_indices]),
        )
        if weight.size != src_indices.size:
            weight = cp.hstack([weight, weight])

    G = graph_class.from_coo(
        nrows + ncols, src_indices, dst_indices, edge_values={"weight": weight}
    )
    node_data = cp.zeros(nrows + ncols, index_dtype)
    node_data[nrows:] = 1
    cudagraph = getattr(G, "_cudagraph", G)
    cudagraph.node_values["bipartite"] = node_data
    if inplace:
        return create_using._become(G)
    return G


@from_biadjacency_matrix._can_run
def _(
    A,
    create_using=None,
    edge_attribute="weight",
    *,
    row_order=None,
    column_order=None,
):
    if row_order is not None or column_order is not None:
        return (
            f"row_order={row_order} and column_order={column_order} not supported; "
            "only row_order=None and column_order=None are currently supported"
        )
    return True
