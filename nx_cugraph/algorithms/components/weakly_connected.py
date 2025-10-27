# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from nx_cugraph.convert import _to_directed_graph
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

from .connected import (
    _connected_components,
    _is_connected,
    _number_connected_components,
)

__all__ = [
    "number_weakly_connected_components",
    "weakly_connected_components",
    "is_weakly_connected",
]


@not_implemented_for("undirected")
@networkx_algorithm(version_added="24.02", _plc="weakly_connected_components")
def weakly_connected_components(G):
    G = _to_directed_graph(G)
    return _connected_components(G, symmetrize="union")


@not_implemented_for("undirected")
@networkx_algorithm(version_added="24.02", _plc="weakly_connected_components")
def number_weakly_connected_components(G):
    G = _to_directed_graph(G)
    return _number_connected_components(G, symmetrize="union")


@not_implemented_for("undirected")
@networkx_algorithm(version_added="24.02", _plc="weakly_connected_components")
def is_weakly_connected(G):
    G = _to_directed_graph(G)
    return _is_connected(G, symmetrize="union")
