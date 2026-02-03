# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import networkx as nx

import nx_cugraph as nxcg

from ..utils import networkx_algorithm
from .digraph import CudaDiGraph, DiGraph
from .graph import Graph, _GraphCache
from .multigraph import CudaMultiGraph, MultiGraph

__all__ = ["CudaMultiDiGraph", "MultiDiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.MultiDiGraph)
gpu_cpu_api = nxcg.utils.decorators._gpu_cpu_api(nx.MultiDiGraph, __name__)


class MultiDiGraph(nx.MultiDiGraph, MultiGraph, DiGraph):
    name = Graph.name
    _node = Graph._node
    _adj = DiGraph._adj
    _succ = DiGraph._succ
    _pred = DiGraph._pred

    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return True

    @classmethod
    @networkx_api
    def is_multigraph(cls) -> bool:
        return True

    @classmethod
    def to_cudagraph_class(cls) -> type[CudaMultiDiGraph]:
        return CudaMultiDiGraph

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiDiGraph]:
        return nx.MultiDiGraph

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.__networkx_cache__ = _GraphCache(self)

    ##########################
    # Networkx graph methods #
    ##########################

    # Dispatch to nx.MultiDiGraph or CudaMultiDiGraph
    __contains__ = gpu_cpu_api("__contains__")
    __len__ = gpu_cpu_api("__len__")
    __iter__ = gpu_cpu_api("__iter__")
    clear = DiGraph.clear
    clear_edges = DiGraph.clear_edges
    get_edge_data = gpu_cpu_api("get_edge_data", edge_data=True)
    has_edge = gpu_cpu_api("has_edge")
    neighbors = gpu_cpu_api("neighbors")
    has_node = gpu_cpu_api("has_node")
    nbunch_iter = gpu_cpu_api("nbunch_iter")
    number_of_edges = MultiGraph.number_of_edges
    number_of_nodes = gpu_cpu_api("number_of_nodes")
    order = gpu_cpu_api("order")
    successors = gpu_cpu_api("successors")


class CudaMultiDiGraph(CudaMultiGraph, CudaDiGraph):
    is_directed = classmethod(MultiDiGraph.is_directed.__func__)
    is_multigraph = classmethod(MultiDiGraph.is_multigraph.__func__)
    to_cudagraph_class = classmethod(MultiDiGraph.to_cudagraph_class.__func__)
    to_networkx_class = classmethod(MultiDiGraph.to_networkx_class.__func__)

    @classmethod
    def _to_compat_graph_class(cls) -> type[MultiDiGraph]:
        return MultiDiGraph

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def to_undirected(self, reciprocal=False, as_view=False):
        raise NotImplementedError


@networkx_algorithm(name="multidigraph__new__", version_added="26.04")
def __new__(cls, *args, **kwargs):
    if nx.config.backends.cugraph.use_compat_graphs:
        return object.__new__(MultiDiGraph)
    return CudaMultiDiGraph(*args, **kwargs)


@__new__._can_run
def _(cls, *args, **kwargs):
    if cls is not nx.MultiDiGraph:
        return "Unknown subclasses of nx.MultiDiGraph are not supported."
    return True
