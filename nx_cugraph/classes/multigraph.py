# Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, ClassVar

import cupy as cp
import networkx as nx
import numpy as np

import nx_cugraph as nxcg

from ..utils import index_dtype
from .graph import CudaGraph, Graph, _GraphCache

if TYPE_CHECKING:
    from nx_cugraph.typing import (
        AttrKey,
        EdgeKey,
        EdgeValue,
        IndexValue,
        NodeKey,
        NodeValue,
        any_ndarray,
    )
__all__ = ["MultiGraph", "CudaMultiGraph"]

networkx_api = nxcg.utils.decorators.networkx_class(nx.MultiGraph)
gpu_cpu_api = nxcg.utils.decorators._gpu_cpu_api(nx.MultiGraph, __name__)


class MultiGraph(nx.MultiGraph, Graph):
    name = Graph.name
    _node = Graph._node
    _adj = Graph._adj

    @classmethod
    @networkx_api
    def is_directed(cls) -> bool:
        return False

    @classmethod
    @networkx_api
    def is_multigraph(cls) -> bool:
        return True

    @classmethod
    def to_cudagraph_class(cls) -> type[CudaMultiGraph]:
        return CudaMultiGraph

    @classmethod
    @networkx_api
    def to_directed_class(cls) -> type[nxcg.MultiDiGraph]:
        return nxcg.MultiDiGraph

    @classmethod
    def to_networkx_class(cls) -> type[nx.MultiGraph]:
        return nx.MultiGraph

    @classmethod
    @networkx_api
    def to_undirected_class(cls) -> type[MultiGraph]:
        return MultiGraph

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        super().__init__(incoming_graph_data, multigraph_input, **attr)
        self.__networkx_cache__ = _GraphCache(self)

    ####################
    # Creation methods #
    ####################

    @classmethod
    def from_coo(
        cls,
        N: int,
        src_indices: cp.ndarray[IndexValue],
        dst_indices: cp.ndarray[IndexValue],
        edge_indices: cp.ndarray[IndexValue] | None = None,
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        edge_keys: list[EdgeKey] | None = None,
        use_compat_graph: bool | None = None,
        **attr,
    ) -> MultiGraph | CudaMultiGraph:
        new_graph = super(cls.to_undirected_class(), cls).from_coo(
            N,
            src_indices,
            dst_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            use_compat_graph=False,
            **attr,
        )
        new_graph.edge_indices = edge_indices
        new_graph.edge_keys = edge_keys
        # Easy and fast sanity checks
        if (
            new_graph.edge_keys is not None
            and len(new_graph.edge_keys) != src_indices.size
        ):
            raise ValueError
        if use_compat_graph or use_compat_graph is None and issubclass(cls, Graph):
            new_graph = new_graph._to_compat_graph()
        return new_graph

    @classmethod
    def from_csr(
        cls,
        indptr: cp.ndarray[IndexValue],
        dst_indices: cp.ndarray[IndexValue],
        edge_indices: cp.ndarray[IndexValue] | None = None,
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        edge_keys: list[EdgeKey] | None = None,
        use_compat_graph: bool | None = None,
        **attr,
    ) -> MultiGraph | CudaMultiGraph:
        N = indptr.size - 1
        src_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(np.arange(N, dtype=index_dtype), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
            edge_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            edge_keys=edge_keys,
            use_compat_graph=use_compat_graph,
            **attr,
        )

    @classmethod
    def from_csc(
        cls,
        indptr: cp.ndarray[IndexValue],
        src_indices: cp.ndarray[IndexValue],
        edge_indices: cp.ndarray[IndexValue] | None = None,
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        edge_keys: list[EdgeKey] | None = None,
        use_compat_graph: bool | None = None,
        **attr,
    ) -> MultiGraph | CudaMultiGraph:
        N = indptr.size - 1
        dst_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(np.arange(N, dtype=index_dtype), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
            edge_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            edge_keys=edge_keys,
            use_compat_graph=use_compat_graph,
            **attr,
        )

    @classmethod
    def from_dcsr(
        cls,
        N: int,
        compressed_srcs: cp.ndarray[IndexValue],
        indptr: cp.ndarray[IndexValue],
        dst_indices: cp.ndarray[IndexValue],
        edge_indices: cp.ndarray[IndexValue] | None = None,
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        edge_keys: list[EdgeKey] | None = None,
        use_compat_graph: bool | None = None,
        **attr,
    ) -> MultiGraph | CudaMultiGraph:
        src_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(compressed_srcs.get(), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
            edge_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            edge_keys=edge_keys,
            use_compat_graph=use_compat_graph,
            **attr,
        )

    @classmethod
    def from_dcsc(
        cls,
        N: int,
        compressed_dsts: cp.ndarray[IndexValue],
        indptr: cp.ndarray[IndexValue],
        src_indices: cp.ndarray[IndexValue],
        edge_indices: cp.ndarray[IndexValue] | None = None,
        edge_values: dict[AttrKey, cp.ndarray[EdgeValue]] | None = None,
        edge_masks: dict[AttrKey, cp.ndarray[bool]] | None = None,
        node_values: dict[AttrKey, any_ndarray[NodeValue]] | None = None,
        node_masks: dict[AttrKey, any_ndarray[bool]] | None = None,
        *,
        key_to_id: dict[NodeKey, IndexValue] | None = None,
        id_to_key: list[NodeKey] | None = None,
        edge_keys: list[EdgeKey] | None = None,
        use_compat_graph: bool | None = None,
        **attr,
    ) -> MultiGraph | CudaGraph:
        dst_indices = cp.array(
            # cp.repeat is slow to use here, so use numpy instead
            np.repeat(compressed_dsts.get(), cp.diff(indptr).get())
        )
        return cls.from_coo(
            N,
            src_indices,
            dst_indices,
            edge_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            edge_keys=edge_keys,
            use_compat_graph=use_compat_graph,
            **attr,
        )

    ##########################
    # Networkx graph methods #
    ##########################

    # Dispatch to nx.MultiGraph or CudaMultiGraph
    __contains__ = gpu_cpu_api("__contains__")
    __len__ = gpu_cpu_api("__len__")
    __iter__ = gpu_cpu_api("__iter__")
    clear = Graph.clear
    clear_edges = Graph.clear_edges
    get_edge_data = gpu_cpu_api("get_edge_data", edge_data=True)
    has_edge = gpu_cpu_api("has_edge")
    neighbors = gpu_cpu_api("neighbors")
    has_node = gpu_cpu_api("has_node")
    nbunch_iter = gpu_cpu_api("nbunch_iter")

    @networkx_api
    def number_of_edges(
        self, u: NodeKey | None = None, v: NodeKey | None = None
    ) -> int:
        if u is not None or v is not None:
            # NotImplemented by CudaGraph
            nx_class = self.to_networkx_class()
            return nx_class.number_of_edges(self, u, v)
        return self._number_of_edges(u, v)

    _number_of_edges = gpu_cpu_api("number_of_edges")
    number_of_nodes = gpu_cpu_api("number_of_nodes")
    order = gpu_cpu_api("order")


class CudaMultiGraph(CudaGraph):
    # networkx properties
    edge_key_dict_factory: ClassVar[type] = dict

    # Not networkx properties

    # In a MultiGraph, each edge has a unique `(src, dst, key)` key.
    # By default, `key` is 0 if possible, else 1, else 2, etc.
    # This key can be any hashable Python object in NetworkX.
    # We don't use a dict for our data structure here, because
    # that would require a `(src, dst, key)` key.
    # Instead, we keep `edge_keys` and/or `edge_indices`.
    # `edge_keys` is the list of Python objects for each edge.
    # `edge_indices` is for the common case of default multiedge keys,
    # in which case we can store it as a cupy array.
    # `edge_indices` is generally preferred. It is possible to provide
    # both where edge_indices is the default and edge_keys is anything.
    # It is also possible for them both to be None, which means the
    # default edge indices has not yet been calculated.
    edge_indices: cp.ndarray[IndexValue] | None
    edge_keys: list[EdgeKey] | None

    ####################
    # Creation methods #
    ####################

    from_coo = classmethod(MultiGraph.from_coo.__func__)
    from_csr = classmethod(MultiGraph.from_csr.__func__)
    from_csc = classmethod(MultiGraph.from_csc.__func__)
    from_dcsr = classmethod(MultiGraph.from_dcsr.__func__)
    from_dcsc = classmethod(MultiGraph.from_dcsc.__func__)

    def __new__(
        cls, incoming_graph_data=None, multigraph_input=None, **attr
    ) -> CudaMultiGraph:
        if isinstance(incoming_graph_data, dict) and multigraph_input is not False:
            new_graph = nxcg.from_networkx(
                nx.MultiGraph(incoming_graph_data, multigraph_input=multigraph_input),
                preserve_all_attrs=True,
            )
        else:
            new_graph = super().__new__(cls, incoming_graph_data)
        new_graph.graph.update(attr)
        return new_graph

    #################
    # Class methods #
    #################

    is_directed = classmethod(MultiGraph.is_directed.__func__)
    is_multigraph = classmethod(MultiGraph.is_multigraph.__func__)
    to_cudagraph_class = classmethod(MultiGraph.to_cudagraph_class.__func__)
    to_networkx_class = classmethod(MultiGraph.to_networkx_class.__func__)

    @classmethod
    @networkx_api
    def to_directed_class(cls) -> type[nxcg.CudaMultiDiGraph]:
        return nxcg.CudaMultiDiGraph

    @classmethod
    @networkx_api
    def to_undirected_class(cls) -> type[CudaMultiGraph]:
        return CudaMultiGraph

    @classmethod
    def _to_compat_graph_class(cls) -> type[MultiGraph]:
        return MultiGraph

    ##########################
    # NetworkX graph methods #
    ##########################

    @networkx_api
    def clear(self) -> None:
        super().clear()
        self.edge_indices = None
        self.edge_keys = None

    @networkx_api
    def clear_edges(self) -> None:
        super().clear_edges()
        self.edge_indices = None
        self.edge_keys = None

    @networkx_api
    def copy(self, as_view: bool = False) -> CudaMultiGraph:
        # Does shallow copy in networkx
        return self._copy(as_view, self.__class__)

    @networkx_api
    def get_edge_data(
        self,
        u: NodeKey,
        v: NodeKey,
        key: EdgeKey | None = None,
        default: EdgeValue | None = None,
    ):
        if self.key_to_id is not None:
            try:
                u = self.key_to_id[u]
                v = self.key_to_id[v]
            except KeyError:
                return default
        else:
            try:
                if u < 0 or v < 0 or u >= self._N or v >= self._N:
                    return default
            except TypeError:
                return default
        mask = (self.src_indices == u) & (self.dst_indices == v)
        if not mask.any():
            return default
        if self.edge_keys is None and key is not None:
            if self.edge_indices is None:
                self._calculate_edge_indices()
            try:
                mask = mask & (self.edge_indices == key)
            except TypeError:
                return default
        indices = cp.nonzero(mask)[0]
        if indices.size == 0:
            return default
        edge_keys = self.edge_keys
        if key is not None and edge_keys is not None:
            mask[[i for i in indices.tolist() if edge_keys[i] != key]] = False
            indices = cp.nonzero(mask)[0]
            if indices.size == 0:
                return default
        if key is not None:
            [index] = indices.tolist()
            return {
                k: v[index].tolist()
                for k, v in self.edge_values.items()
                if k not in self.edge_masks or self.edge_masks[k][index]
            }
        return {
            edge_keys[index] if edge_keys is not None else index: {
                k: v[index].tolist()
                for k, v in self.edge_values.items()
                if k not in self.edge_masks or self.edge_masks[k][index]
            }
            for index in indices.tolist()
        }

    @networkx_api
    def has_edge(self, u: NodeKey, v: NodeKey, key: EdgeKey | None = None) -> bool:
        if self.key_to_id is not None:
            try:
                u = self.key_to_id[u]
                v = self.key_to_id[v]
            except KeyError:
                return False
        mask = (self.src_indices == u) & (self.dst_indices == v)
        if key is None or (self.edge_indices is None and self.edge_keys is None):
            return bool(mask.any())
        if self.edge_keys is None:
            try:
                return bool((mask & (self.edge_indices == key)).any())
            except TypeError:
                return False
        indices = cp.nonzero(mask)[0]
        if indices.size == 0:
            return False
        edge_keys = self.edge_keys
        return any(edge_keys[i] == key for i in indices.tolist())

    @networkx_api
    def to_directed(self, as_view: bool = False) -> nxcg.CudaMultiDiGraph:
        return self._copy(as_view, self.to_directed_class())

    @networkx_api
    def to_undirected(self, as_view: bool = False) -> CudaMultiGraph:
        # Does deep copy in networkx
        return self._copy(as_view, self.to_undirected_class())

    ###################
    # Private methods #
    ###################

    def _copy(self, as_view: bool, cls: type[CudaGraph], reverse: bool = False):
        # DRY warning: see also CudaGraph._copy
        src_indices = self.src_indices
        dst_indices = self.dst_indices
        edge_indices = self.edge_indices
        edge_values = self.edge_values
        edge_masks = self.edge_masks
        node_values = self.node_values
        node_masks = self.node_masks
        key_to_id = self.key_to_id
        id_to_key = None if key_to_id is None else self._id_to_key
        edge_keys = self.edge_keys
        if self.__networkx_cache__ is None:
            __networkx_cache__ = None
        elif not reverse and cls is self.__class__:
            __networkx_cache__ = self.__networkx_cache__
        else:
            __networkx_cache__ = {}
        if not as_view:
            src_indices = src_indices.copy()
            dst_indices = dst_indices.copy()
            edge_indices = edge_indices.copy()
            edge_values = {key: val.copy() for key, val in edge_values.items()}
            edge_masks = {key: val.copy() for key, val in edge_masks.items()}
            node_values = {key: val.copy() for key, val in node_values.items()}
            node_masks = {key: val.copy() for key, val in node_masks.items()}
            if key_to_id is not None:
                key_to_id = key_to_id.copy()
                if id_to_key is not None:
                    id_to_key = id_to_key.copy()
            if edge_keys is not None:
                edge_keys = edge_keys.copy()
            if __networkx_cache__ is not None:
                __networkx_cache__ = __networkx_cache__.copy()
        if reverse:
            src_indices, dst_indices = dst_indices, src_indices
        rv = cls.from_coo(
            self._N,
            src_indices,
            dst_indices,
            edge_indices,
            edge_values,
            edge_masks,
            node_values,
            node_masks,
            key_to_id=key_to_id,
            id_to_key=id_to_key,
            edge_keys=edge_keys,
            use_compat_graph=False,
        )
        if as_view:
            rv.graph = self.graph
        else:
            rv.graph.update(deepcopy(self.graph))
        rv.__networkx_cache__ = __networkx_cache__
        return rv

    def _sort_edge_indices(self, primary="src"):
        # DRY warning: see also CudaGraph._sort_edge_indices
        if self.edge_indices is None and self.edge_keys is None:
            return super()._sort_edge_indices(primary=primary)
        if primary == "src":
            if self.edge_indices is None:
                stacked = (self.dst_indices, self.src_indices)
            else:
                stacked = (self.edge_indices, self.dst_indices, self.src_indices)
        elif primary == "dst":
            if self.edge_indices is None:
                stacked = (self.src_indices, self.dst_indices)
            else:
                stacked = (self.edge_indices, self.dst_indices, self.src_indices)
        else:
            raise ValueError(
                f'Bad `primary` argument; expected "src" or "dst", got {primary!r}'
            )
        indices = cp.lexsort(cp.vstack(stacked))
        if (cp.diff(indices) > 0).all():
            # Already sorted
            return
        self.src_indices = self.src_indices[indices]
        self.dst_indices = self.dst_indices[indices]
        self.edge_values.update(
            {key: val[indices] for key, val in self.edge_values.items()}
        )
        self.edge_masks.update(
            {key: val[indices] for key, val in self.edge_masks.items()}
        )
        if self.edge_indices is not None:
            self.edge_indices = self.edge_indices[indices]
        if self.edge_keys is not None:
            edge_keys = self.edge_keys
            self.edge_keys = [edge_keys[i] for i in indices.tolist()]
