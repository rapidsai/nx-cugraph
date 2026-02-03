# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from nx_cugraph.convert import _to_directed_graph, _to_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = ["degree_centrality", "in_degree_centrality", "out_degree_centrality"]


@networkx_algorithm(version_added="23.12")
def degree_centrality(G):
    G = _to_graph(G)
    if len(G) <= 1:
        return dict.fromkeys(G, 1)
    deg = G._degrees_array()
    centrality = deg * (1 / (len(G) - 1))
    return G._nodearray_to_dict(centrality)


@degree_centrality._should_run
def _(G):
    return "Fast algorithm; not worth converting."


@not_implemented_for("undirected")
@networkx_algorithm(version_added="23.12")
def in_degree_centrality(G):
    G = _to_directed_graph(G)
    if len(G) <= 1:
        return dict.fromkeys(G, 1)
    deg = G._in_degrees_array()
    centrality = deg * (1 / (len(G) - 1))
    return G._nodearray_to_dict(centrality)


@in_degree_centrality._should_run
def _(G):
    return "Fast algorithm; not worth converting."


@not_implemented_for("undirected")
@networkx_algorithm(version_added="23.12")
def out_degree_centrality(G):
    G = _to_directed_graph(G)
    if len(G) <= 1:
        return dict.fromkeys(G, 1)
    deg = G._out_degrees_array()
    centrality = deg * (1 / (len(G) - 1))
    return G._nodearray_to_dict(centrality)


@out_degree_centrality._should_run
def _(G):
    return "Fast algorithm; not worth converting."
