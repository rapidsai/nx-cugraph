# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import pytest

import nx_cugraph as nxcg
from nx_cugraph import _nxver


def test_class_to_class():
    """Basic sanity checks to ensure metadata relating graph classes are accurate."""
    for prefix in ["", "Cuda"]:
        for suffix in ["Graph", "DiGraph", "MultiGraph", "MultiDiGraph"]:
            cls_name = f"{prefix}{suffix}"
            cls = getattr(nxcg, cls_name)
            assert cls.__name__ == cls_name
            G = cls()
            assert cls is G.__class__
            # cudagraph
            val = cls.to_cudagraph_class()
            val2 = G.to_cudagraph_class()
            assert val is val2
            assert val.__name__ == f"Cuda{suffix}"
            assert val.__module__.startswith("nx_cugraph")
            assert cls.is_directed() == G.is_directed() == val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            # networkx
            val = cls.to_networkx_class()
            val2 = G.to_networkx_class()
            assert val is val2
            assert val.__name__ == suffix
            assert val.__module__.startswith("networkx")
            val = val()
            assert cls.is_directed() == G.is_directed() == val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            # directed
            val = cls.to_directed_class()
            val2 = G.to_directed_class()
            assert val is val2
            assert val.__module__.startswith("nx_cugraph")
            assert val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            if "Di" in suffix:
                assert val is cls
            else:
                assert "Di" in val.__name__
                assert prefix in val.__name__
                assert cls.to_undirected_class() is cls
            # undirected
            val = cls.to_undirected_class()
            val2 = G.to_undirected_class()
            assert val is val2
            assert val.__module__.startswith("nx_cugraph")
            assert not val.is_directed()
            assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()
            if "Di" not in suffix:
                assert val is cls
            else:
                assert "Di" not in val.__name__
                assert prefix in val.__name__
                assert cls.to_directed_class() is cls
            # "zero"
            if prefix == "Cuda":
                val = cls._to_compat_graph_class()
                val2 = G._to_compat_graph_class()
                assert val is val2
                assert val.__name__ == suffix
                assert val.__module__.startswith("nx_cugraph")
                assert val.to_cudagraph_class() is cls
                assert cls.is_directed() == G.is_directed() == val.is_directed()
                assert cls.is_multigraph() == G.is_multigraph() == val.is_multigraph()


@pytest.mark.parametrize(
    "nxcg_class", [nxcg.Graph, nxcg.DiGraph, nxcg.MultiGraph, nxcg.MultiDiGraph]
)
def test_dispatch_graph_classes(nxcg_class):
    if _nxver < (3, 5):
        pytest.skip(reason="Dispatching graph classes requires nx >=3.5")
    nx_class = nxcg_class.to_networkx_class()
    assert nx_class is not nxcg_class

    class NxGraphSubclass(nx_class):
        pass

    class NxcgGraphSubclass(nxcg_class):
        pass

    with nx.config.backend_priority(classes=[]):
        G = nx_class()
        assert type(G) is nx_class
        G = nx_class(backend="cugraph")
        assert type(G) is nxcg_class
        G = NxGraphSubclass()
        assert type(G) is NxGraphSubclass
        with pytest.raises(NotImplementedError, match="not implemented by 'cugraph'"):
            # can_run is False for unknown subclasses
            NxGraphSubclass(backend="cugraph")
        G = NxcgGraphSubclass()
        assert type(G) is NxcgGraphSubclass
        with pytest.raises(NotImplementedError, match="not implemented by 'cugraph'"):
            NxcgGraphSubclass(backend="cugraph")

    with nx.config.backend_priority(classes=["cugraph"]):
        G = nx_class()
        assert type(G) is nxcg_class
        G = nx_class(backend="networkx")
        assert type(G) is nx_class

        # can_run is False for unknown subclasses
        G = NxGraphSubclass()
        assert type(G) is NxGraphSubclass
        G = NxGraphSubclass(backend="networkx")
        assert type(G) is NxGraphSubclass

        G = NxcgGraphSubclass()
        assert type(G) is NxcgGraphSubclass
        G = NxcgGraphSubclass(backend="networkx")
        assert type(G) is NxcgGraphSubclass  # Perhaps odd, but the correct behavior
