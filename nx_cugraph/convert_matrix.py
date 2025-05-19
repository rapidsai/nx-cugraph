# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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
import networkx as nx
import numpy as np

from nx_cugraph import _nxver

from .convert import _to_graph
from .generators._utils import _create_using_class
from .utils import (
    _cp_iscopied_asarray,
    index_dtype,
    networkx_algorithm,
)

__all__ = [
    "from_pandas_edgelist",
    "from_scipy_sparse_array",
    "to_scipy_sparse_array",
    "to_numpy_array",
    "from_numpy_array",
]


# Value columns with string dtype is not supported
@networkx_algorithm(
    is_incomplete=True, version_added="23.12", fallback=True, create_using_arg=4
)
def from_pandas_edgelist(
    df,
    source="source",
    target="target",
    edge_attr=None,
    create_using=None,
    edge_key=None,
):
    """cudf.DataFrame inputs also supported; value columns with str is unsuppported."""
    # This function never shares ownership of the underlying arrays of the DataFrame
    # columns. We will perform a copy if necessary even if given e.g. a cudf.DataFrame.
    graph_class, inplace = _create_using_class(create_using)
    # Try to be optimal whether using pandas, cudf, or cudf.pandas
    src_series = df[source]
    dst_series = df[target]
    try:
        # Optimistically try to use cupy, but fall back to numpy if necessary
        src_array = src_series.to_cupy()
        dst_array = dst_series.to_cupy()
    except (AttributeError, TypeError, ValueError, NotImplementedError):
        src_array = src_series.to_numpy()
        dst_array = dst_series.to_numpy()
    try:
        # Minimize unnecessary data copies by tracking whether we copy or not
        is_src_copied, src_array = _cp_iscopied_asarray(
            src_array, orig_object=src_series
        )
        is_dst_copied, dst_array = _cp_iscopied_asarray(
            dst_array, orig_object=dst_series
        )
        np_or_cp = cp
    except ValueError:
        is_src_copied = is_dst_copied = False
        src_array = np.asarray(src_array)
        dst_array = np.asarray(dst_array)
        np_or_cp = np
    # TODO: create renumbering helper function(s)
    # Renumber step 0: node keys
    nodes = np_or_cp.unique(np_or_cp.concatenate([src_array, dst_array]))
    N = nodes.size
    kwargs = {}
    if N > 0 and (
        nodes[0] != 0
        or nodes[N - 1] != N - 1
        or (
            nodes.dtype.kind not in {"i", "u"}
            and not (nodes == np_or_cp.arange(N, dtype=np.int64)).all()
        )
    ):
        # We need to renumber indices--np_or_cp.searchsorted to the rescue!
        kwargs["id_to_key"] = nodes.tolist()
        src_indices = cp.asarray(np_or_cp.searchsorted(nodes, src_array), index_dtype)
        dst_indices = cp.asarray(np_or_cp.searchsorted(nodes, dst_array), index_dtype)
    else:
        # Copy if necessary so we don't share ownership of input arrays.
        if is_src_copied:
            src_indices = src_array
        else:
            src_indices = cp.array(src_array)
        if is_dst_copied:
            dst_indices = dst_array
        else:
            dst_indices = cp.array(dst_array)

    if not graph_class.is_directed():
        # Symmetrize the edges
        mask = src_indices != dst_indices
        if mask.all():
            mask = None
        src_indices, dst_indices = (
            cp.hstack(
                (src_indices, dst_indices[mask] if mask is not None else dst_indices)
            ),
            cp.hstack(
                (dst_indices, src_indices[mask] if mask is not None else src_indices)
            ),
        )

    if edge_attr is not None:
        # Additional columns requested for edge data
        if edge_attr is True:
            attr_col_headings = df.columns.difference({source, target}).to_list()
        elif isinstance(edge_attr, (list, tuple)):
            attr_col_headings = edge_attr
        else:
            attr_col_headings = [edge_attr]
        if len(attr_col_headings) == 0:
            raise nx.NetworkXError(
                "Invalid edge_attr argument: No columns found with name: "
                f"{attr_col_headings}"
            )
        try:
            edge_values = {
                key: cp.array(val.to_numpy())
                for key, val in df[attr_col_headings].items()
            }
        except (KeyError, TypeError) as exc:
            raise nx.NetworkXError(f"Invalid edge_attr argument: {edge_attr}") from exc

        if not graph_class.is_directed():
            # Symmetrize the edges
            edge_values = {
                key: cp.hstack((val, val[mask] if mask is not None else val))
                for key, val in edge_values.items()
            }
        kwargs["edge_values"] = edge_values

    if (
        graph_class.is_multigraph()
        and edge_key is not None
        and (
            # In nx <= 3.3, `edge_key` was ignored if `edge_attr` is None
            edge_attr is not None
            or _nxver >= (3, 4)
        )
    ):
        try:
            edge_keys = df[edge_key].to_list()
        except (KeyError, TypeError) as exc:
            raise nx.NetworkXError(f"Invalid edge_key argument: {edge_key}") from exc
        if not graph_class.is_directed():
            # Symmetrize the edges; remember, `edge_keys` is a list!
            if mask is None:
                edge_keys *= 2
            else:
                edge_keys += [
                    key for keep, key in zip(mask.tolist(), edge_keys) if keep
                ]
        kwargs["edge_keys"] = edge_keys

    G = graph_class.from_coo(N, src_indices, dst_indices, **kwargs)
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(version_added="25.06")
def to_scipy_sparse_array(G, nodelist=None, dtype=None, weight="weight", format="csr"):
    # Future work: allow this to return a cupyx.scipy.sparse object.
    # This code is very well covered by networkx tests, and the logic
    # for raising errors closely matches networkx.
    import scipy as sp

    G = _to_graph(G, weight, 1, dtype)
    if G._N == 0:
        raise nx.NetworkXError("Graph has no nodes or edges")

    is_empty = G.src_indices.size == 0  # Use is_empty to avoid work
    if nodelist is None:
        # No reordering necessary
        nlen = G._N
        if is_empty:
            src_indices = dst_indices = edge_array = ()
        else:
            src_indices = G.src_indices
            dst_indices = G.dst_indices
            mask = None
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError("nodelist has no nodes")
        if nlen != len(set(G.nbunch_iter(nodelist))):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f"Node {n} in nodelist is not in G")
            raise nx.NetworkXError("nodelist contains duplicates.")
        if is_empty:
            src_indices = dst_indices = edge_array = ()
        else:
            src_indices, dst_indices, mask = G._subgraph_indices(nodelist)

            if src_indices.size == 0:
                is_empty = True
                src_indices = dst_indices = edge_array = ()

    if not is_empty:
        src_indices = cp.asnumpy(src_indices)
        dst_indices = cp.asnumpy(dst_indices)

        edge_array = G._subgraph_weights(mask, weight, 1)
        edge_array = cp.asnumpy(edge_array)

    # PERF: convert to desired sparse format on GPU before copying to CPU
    A = sp.sparse.coo_array(
        (edge_array, (src_indices, dst_indices)), shape=(nlen, nlen), dtype=dtype
    )
    try:
        return A.asformat(format)
    except ValueError as exc:
        raise nx.NetworkXError(f"Unknown sparse matrix format: {format}") from exc


@networkx_algorithm(version_added="23.12", fallback=True, create_using_arg=2)
def from_scipy_sparse_array(
    A, parallel_edges=False, create_using=None, edge_attribute="weight"
):
    graph_class, inplace = _create_using_class(create_using)
    m, n = A.shape
    if m != n:
        raise nx.NetworkXError(f"Adjacency matrix not square: nx,ny={A.shape}")
    if A.format != "coo":
        A = A.tocoo()
    if A.dtype.kind in {"i", "u"} and graph_class.is_multigraph() and parallel_edges:
        src_indices = cp.array(np.repeat(A.row, A.data), index_dtype)
        dst_indices = cp.array(np.repeat(A.col, A.data), index_dtype)
        weight = cp.empty(src_indices.size, A.data.dtype)
        weight[:] = 1
    else:
        src_indices = cp.array(A.row, index_dtype)
        dst_indices = cp.array(A.col, index_dtype)
        weight = cp.array(A.data)
    G = graph_class.from_coo(
        n, src_indices, dst_indices, edge_values={"weight": weight}
    )
    if inplace:
        return create_using._become(G)
    return G


@networkx_algorithm(
    extra_params={
        "use_numpy : bool, default False": (
            "When working with structured dtypes, might want to use numpy",
            "Referring to: ",
            "https://github.com/rapidsai/nx-cugraph/pull/127#discussion_r2078445779",
        ),
    },
    version_added="25.06",
    fallback=False,
)
def to_numpy_array(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
    # nx_cugraph-only argument
    use_numpy=False,
):
    """MultiGraphs are not yet supported"""
    if dtype is None:
        dtype = np.float64
    dtype = np.dtype(dtype)

    G = _to_graph(G, weight, 1, dtype)

    if nodelist is not None:
        N = len(nodelist)
        # use set to check for nodes not in the graph or duplicate nodes
        nodelist_as_set = set(nodelist)
        if nodelist_as_set - set(G):
            raise nx.NetworkXError(
                f"Nodes {nodelist_as_set - set(G)} in nodelist is not in G"
            )
        if len(nodelist_as_set) < N:
            raise nx.NetworkXError(f"Nodelist {nodelist} contains duplicates")
    else:
        N = G._N

    use_numpy = dtype.names is not None
    if not use_numpy:
        # try:
        A = cp.full((N, N), fill_value=nonedge, dtype=dtype, order=order)
        # except Exception:
        # use_numpy = True
    else:
        A = np.full((N, N), fill_value=nonedge, dtype=dtype, order=order)

    # Case: graph with no nodes
    if G._N == 0:
        return cp.asnumpy(A)

    # assume edge_attrs is None unless other weight value is specified ig?
    edge_attrs = None
    if A.dtype.names:
        if weight is None:
            edge_attrs = dtype.names
        else:
            raise ValueError(
                "Specifying `weight` not supported for structured dtypes\n."
                "To create adjacency matrices from structured dtypes,"
                "use `weight=None`."
            )

    src_indices, dst_indices, mask = G._subgraph_indices(nodelist)

    if edge_attrs:
        edge_array = np.empty(src_indices.size, dtype=dtype)
        for edge_attr in edge_attrs:
            if edge_attr in G.edge_values:
                e_array = G.edge_values[edge_attr]
                if edge_attr in G.edge_masks:
                    e_array = cp.where(G.edge_masks[edge_attr], e_array, 1)
            else:
                e_array = np.ones(G.src_indices.size, dtype=dtype)
            edge_array[edge_attr] = cp.asnumpy(e_array)
    else:
        edge_array = G._subgraph_weights(mask, weight, 1)

    if use_numpy:
        src_indices = cp.asnumpy(src_indices)
        dst_indices = cp.asnumpy(dst_indices)

    A[src_indices, dst_indices] = edge_array

    return cp.asnumpy(A)


@to_numpy_array._can_run
def _(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    # do not handle multigraph yet
    return not G.is_multigraph()


# @networkx_algorithm(version_added="25.06", fallback=True)
def from_numpy_array(
    A, parallel_edges=False, create_using=None, edge_attr="weight", *, nodelist=None
):
    pass
