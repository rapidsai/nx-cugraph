# Copyright (c) 2024, NVIDIA CORPORATION.
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
import networkx as nx
import numpy as np
import pylibcugraph as plc

from nx_cugraph.convert import _to_graph
from nx_cugraph.utils import _dtype_param, _get_float_dtype, networkx_algorithm

from .unweighted import PathMapping, ReversePathMapping, _bfs

__all__ = [
    "dijkstra_path",
    "dijkstra_path_length",
    "single_source_dijkstra",
    "single_source_dijkstra_path",
    "single_source_dijkstra_path_length",
    "all_pairs_dijkstra",
    "all_pairs_dijkstra_path",
    "all_pairs_dijkstra_path_length",
    "bellman_ford_path",
    "bellman_ford_path_length",
    "single_source_bellman_ford",
    "single_source_bellman_ford_path",
    "single_source_bellman_ford_path_length",
    "all_pairs_bellman_ford_path",
    "all_pairs_bellman_ford_path_length",
]


def _add_doc(func):
    func.__doc__ = (
        "Negative cycles are not yet supported. ``NotImplementedError`` will be raised "
        "if there are negative edge weights. We plan to support negative edge weights "
        "soon. Also, callable ``weight`` argument is not supported."
    )
    return func


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def dijkstra_path(G, source, target, weight="weight", *, dtype=None):
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    return _sssp(G, source, weight, target, return_type="path", dtype=dtype)


@dijkstra_path._can_run
def _(G, source, target, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
@_add_doc
def bellman_ford_path(G, source, target, weight="weight", *, dtype=None):
    return dijkstra_path(G, source, target, weight=weight, dtype=dtype)


@bellman_ford_path._can_run
def _(G, source, target, weight="weight", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def dijkstra_path_length(G, source, target, weight="weight", *, dtype=None):
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    return _sssp(G, source, weight, target, return_type="length", dtype=dtype)


@dijkstra_path._can_run
def _(G, source, target, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
@_add_doc
def bellman_ford_path_length(G, source, target, weight="weight", *, dtype=None):
    return dijkstra_path_length(G, source, target, weight=weight, dtype=dtype)


@bellman_ford_path_length._can_run
def _(G, source, target, weight="weight", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def single_source_dijkstra_path(G, source, cutoff=None, weight="weight", *, dtype=None):
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    return _sssp(G, source, weight, return_type="path", dtype=dtype, cutoff=cutoff)


@single_source_dijkstra_path._can_run
def _(G, source, cutoff=None, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
@_add_doc
def single_source_bellman_ford_path(G, source, weight="weight", *, dtype=None):
    return single_source_dijkstra_path(G, source, weight=weight, dtype=dtype)


@single_source_bellman_ford_path._can_run
def _(G, source, weight="weight", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def single_source_dijkstra_path_length(
    G, source, cutoff=None, weight="weight", *, dtype=None
):
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    return _sssp(G, source, weight, return_type="length", dtype=dtype, cutoff=cutoff)


@single_source_dijkstra_path_length._can_run
def _(G, source, cutoff=None, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
@_add_doc
def single_source_bellman_ford_path_length(G, source, weight="weight", *, dtype=None):
    return single_source_dijkstra_path_length(G, source, weight=weight, dtype=dtype)


@single_source_bellman_ford_path_length._can_run
def _(G, source, weight="weight", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def single_source_dijkstra(
    G, source, target=None, cutoff=None, weight="weight", *, dtype=None
):
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    return _sssp(
        G, source, weight, target, return_type="length-path", dtype=dtype, cutoff=cutoff
    )


@single_source_dijkstra._can_run
def _(G, source, target=None, cutoff=None, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
@_add_doc
def single_source_bellman_ford(G, source, target=None, weight="weight", *, dtype=None):
    return single_source_dijkstra(G, source, target=target, weight=weight, dtype=dtype)


@single_source_bellman_ford._can_run
def _(G, source, target=None, weight="weight", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def all_pairs_dijkstra(G, cutoff=None, weight="weight", *, dtype=None):
    # TODO PERF: batched bfs to compute many at once
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    for n in G:
        yield (
            n,
            _sssp(G, n, weight, return_type="length-path", dtype=dtype, cutoff=cutoff),
        )


@all_pairs_dijkstra._can_run
def _(G, cutoff=None, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def all_pairs_dijkstra_path_length(G, cutoff=None, weight="weight", *, dtype=None):
    # TODO PERF: batched bfs to compute many at once
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    for n in G:
        yield (n, _sssp(G, n, weight, return_type="length", dtype=dtype, cutoff=cutoff))


@all_pairs_dijkstra_path_length._can_run
def _(G, cutoff=None, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
@_add_doc
def all_pairs_bellman_ford_path_length(G, weight="weight", *, dtype=None):
    return all_pairs_dijkstra_path_length(G, weight=weight, dtype=None)


@all_pairs_bellman_ford_path_length._can_run
def _(G, weight="weight", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


@networkx_algorithm(extra_params=_dtype_param, version_added="24.08", _plc="sssp")
def all_pairs_dijkstra_path(G, cutoff=None, weight="weight", *, dtype=None):
    # TODO PERF: batched bfs to compute many at once
    G = _to_graph(G, weight, 1, np.float32)
    dtype = _get_float_dtype(dtype, graph=G, weight=weight)
    for n in G:
        yield (n, _sssp(G, n, weight, return_type="path", dtype=dtype, cutoff=cutoff))


@all_pairs_dijkstra_path._can_run
def _(G, cutoff=None, weight="weight", *, dtype=None):
    return not callable(weight)


@networkx_algorithm(extra_params=_dtype_param, version_added="24.04", _plc="sssp")
@_add_doc
def all_pairs_bellman_ford_path(G, weight="weight", *, dtype=None):
    return all_pairs_dijkstra_path(G, weight=weight, dtype=None)


@all_pairs_bellman_ford_path._can_run
def _(G, weight="weight", *, dtype=None):
    return (
        weight is None
        or not callable(weight)
        and not nx.is_negatively_weighted(G, weight=weight)
    )


def _sssp(
    G,
    source,
    weight,
    target=None,
    *,
    return_type,
    dtype,
    reverse_path=False,
    cutoff=None,
):
    """SSSP for weighted shortest paths.

    Parameters
    ----------
    return_type : {"length", "path", "length-path"}

    """
    # DRY: _bfs in unweighted.py has similar code
    if source not in G:
        raise nx.NodeNotFound(f"Node {source} not found in graph")
    if target is not None:
        if source == target:
            if return_type == "path":
                return [source]
            if return_type == "length":
                return 0
            # return_type == "length-path"
            return 0, [source]
        if target not in G or G.src_indices.size == 0:
            raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
    elif G.src_indices.size == 0:
        if return_type == "path":
            return {source: [source]}
        if return_type == "length":
            return {source: 0}
        # return_type == "length-path"
        return {source: 0}, {source: [source]}

    if callable(weight):
        raise NotImplementedError("callable `weight` argument is not supported")

    if weight not in G.edge_values:
        # No edge values, so use BFS instead
        return _bfs(G, source, cutoff, "Source", return_type=return_type, target=target)

    # Check for negative values since we don't support negative cycles
    edge_vals = G.edge_values[weight]
    if weight in G.edge_masks:
        edge_vals = edge_vals[G.edge_masks[weight]]
    if (edge_vals < 0).any():
        raise NotImplementedError("Negative edge weights not yet supported")
    edge_val = edge_vals[0]
    if (edge_vals == edge_val).all() and (
        edge_vals.size == G.src_indices.size or edge_val == 1
    ):
        # Edge values are all the same, so use scaled BFS instead
        return _bfs(
            G,
            source,
            None if cutoff is None else cutoff / edge_val,
            "Source",
            return_type=return_type,
            target=target,
            scale=edge_val,
            reverse_path=reverse_path,
        )

    src_index = source if G.key_to_id is None else G.key_to_id[source]
    if cutoff is None:
        cutoff = np.inf
    else:
        cutoff = np.nextafter(cutoff, np.inf, dtype=np.float64)

    node_ids, distances, predecessors = plc.sssp(
        resource_handle=plc.ResourceHandle(),
        graph=G._get_plc_graph(weight, 1, dtype),
        source=src_index,
        cutoff=cutoff,
        compute_predecessors=True,  # TODO: False is not yet supported
        # compute_predecessors=return_type != "length",
        do_expensive_check=False,
    )
    mask = distances != np.finfo(distances.dtype).max
    node_ids = node_ids[mask]
    if return_type != "path":
        lengths = G._nodearrays_to_dict(node_ids, distances[mask])
        if target is not None:
            if target not in lengths:
                raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
            lengths = lengths[target]
    if return_type != "length":
        if target is not None:
            d = dict(zip(node_ids.tolist(), predecessors[mask].tolist()))
            dst_index = target if G.key_to_id is None else G.key_to_id[target]
            if dst_index not in d:
                raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")
            cur = dst_index
            paths = [dst_index]
            while cur != src_index:
                cur = d[cur]
                paths.append(cur)
            if (id_to_key := G.id_to_key) is not None:
                if reverse_path:
                    paths = [id_to_key[cur] for cur in paths]
                else:
                    paths = [id_to_key[cur] for cur in reversed(paths)]
            elif not reverse_path:
                paths.reverse()
        else:
            # Computing paths to all nodes can be expensive, so let's delay
            # computation until needed using `PathMapping`.
            key_iter = node_ids.tolist()
            pred_iter = predecessors[mask].tolist()
            if G.key_to_id is not None:
                key_iter = G._nodeiter_to_iter(key_iter)
                pred_iter = G._nodeiter_to_iter(pred_iter)
            key_to_pred = dict(zip(key_iter, pred_iter))
            key_to_pred[source] = None
            if reverse_path:
                paths = ReversePathMapping({source: [source]}, key_to_pred)
            else:
                paths = PathMapping({source: [source]}, key_to_pred)
    if return_type == "path":
        return paths
    if return_type == "length":
        return lengths
    # return_type == "length-path"
    return lengths, paths
