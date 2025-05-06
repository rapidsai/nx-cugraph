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
    _get_float_dtype,
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
            node_ids = G._nodekeys_to_nodearray(nodelist)

            # Subgraph
            if nlen < G._N:
                # TODO: create utility funcs for renumbering/reordering node_ids.
                # Using `mapper` like this is a useful trick that may not be obvious.
                mapper = cp.empty(G._N, dtype=index_dtype)
                mapper[:] = -1  # Indicate nodes to exclude
                mapper[node_ids] = cp.arange(node_ids.size, dtype=index_dtype)
                src_indices = mapper[G.src_indices]
                dst_indices = mapper[G.dst_indices]
                mask = (src_indices != -1) & (dst_indices != -1)
                src_indices = src_indices[mask]
                if src_indices.size == 0:
                    is_empty = True
                    src_indices = dst_indices = edge_array = ()
                else:
                    dst_indices = dst_indices[mask]

            # All nodes, reordered
            else:
                # TODO: create utility funcs for renumbering/reordering node_ids.
                # Using `mapper` like this is a useful trick that may not be obvious.
                mapper = cp.empty(G._N, dtype=index_dtype)
                mapper[node_ids] = cp.arange(node_ids.size, dtype=index_dtype)
                src_indices = mapper[G.src_indices]
                dst_indices = mapper[G.dst_indices]

    if not is_empty:
        src_indices = cp.asnumpy(src_indices)
        dst_indices = cp.asnumpy(dst_indices)

        if weight in G.edge_values:
            edge_array = G.edge_values[weight]
            if weight in G.edge_masks:
                edge_array = cp.where(G.edge_masks[weight], edge_array, 1)
            if nlen < G._N:
                edge_array = edge_array[mask]
            edge_array = cp.asnumpy(edge_array)
        else:
            edge_array = np.repeat(1, src_indices.size)

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


@networkx_algorithm(version_added="25.06", fallback=True)
def to_numpy_array(
    G,
    nodelist=None,
    dtype=None,
    order=None,
    multigraph_weight=sum,
    weight="weight",
    nonedge=0.0,
):
    """MultiGraphs are not yet supported"""
    # print(" ==> hello!! dispatched to nxcg!")
    G = _to_graph(G, weight, 1, _get_float_dtype(dtype))

    if nodelist is None:
        nodelist = list(G)

    N = len(nodelist)

    # use set to check for nodes not in the graph or duplicate nodes
    nodelist_as_set = set(nodelist)
    if nodelist_as_set - set(G):
        raise nx.NetworkXError(
            f"Nodes {nodelist_as_set - set(G)} in nodelist is not in G"
        )
    if len(nodelist_as_set) < N:
        raise nx.NetworkXError(f"Nodelist {nodelist} contains duplicates")

    # Construct array of fill_value matching resulting shape
    A = cp.full((N, N), fill_value=nonedge, dtype=dtype, order=order)

    # Case: empty nodelist or graph without any edges
    if N == 0 or G.number_of_edges() == 0:
        return A

    # If dtype is structured and weight is None, use dtype field names as
    # edge attribs
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

    # Map nodes to row/col in matrix
    # idx = dict(zip(nodelist, range(N)))
    if len(nodelist) < len(G):
        G = G.subgraph(nodelist)

    # Collect all edge weights and reduce with `multigraph_weights`
    # TODO handle this case
    if G.is_multigraph():
        if edge_attrs:
            raise nx.NetworkXError(
                "Structured arrays are not supported for MultiGraphs"
            )
    else:
        # Special case: TODO: figure out
        if edge_attrs:
            pass

        wts = G.edge_values["weight"]
        A[G.src_indices, G.dst_indices] = wts

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


@networkx_algorithm(version_added="25.06", fallback=True)
def from_numpy_array(
    A, parallel_edges=False, create_using=None, edge_attr="weight", *, nodelist=None
):
    pass
