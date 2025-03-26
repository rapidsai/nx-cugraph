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

from nx_cugraph import _nxver


def test_bc():
    G = nx.karate_club_graph()
    bc_nx = nx.betweenness_centrality(G)
    bc_cg = nx.betweenness_centrality(G, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg


@pytest.mark.skipif(_nxver < (3, 5),
                    reason="Test only supported for BC normalization used in NX 3.5+")
def test_bc_rng():
    G = nx.karate_club_graph()
    bc_nx = nx.betweenness_centrality(G, k=4, seed=42)
    bc_cg = nx.betweenness_centrality(G, k=4, seed=42, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg


def test_edge_bc():
    G = nx.karate_club_graph()
    bc_nx = nx.edge_betweenness_centrality(G)
    bc_cg = nx.edge_betweenness_centrality(G, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg

    G = nx.path_graph(5, create_using=nx.Graph)
    bc_nx = nx.edge_betweenness_centrality(G)
    bc_cg = nx.edge_betweenness_centrality(G, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg

    G = nx.path_graph(10, create_using=nx.DiGraph)
    bc_nx = nx.edge_betweenness_centrality(G)
    bc_cg = nx.edge_betweenness_centrality(G, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg


def test_edge_bc_rng():
    G = nx.karate_club_graph()
    bc_nx = nx.edge_betweenness_centrality(G, k=4, seed=7)
    bc_cg = nx.edge_betweenness_centrality(G, k=4, seed=7, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg

    G = nx.path_graph(5, create_using=nx.Graph)
    bc_nx = nx.edge_betweenness_centrality(G, k=2, seed=7)
    bc_cg = nx.edge_betweenness_centrality(G, k=2, seed=7, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg

    G = nx.path_graph(10, create_using=nx.DiGraph)
    bc_nx = nx.edge_betweenness_centrality(G, k=4, seed=8)
    bc_cg = nx.edge_betweenness_centrality(G, k=4, seed=8, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg
