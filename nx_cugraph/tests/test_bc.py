# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import pytest

from nx_cugraph import _nxver


def test_bc():
    G = nx.karate_club_graph()
    bc_nx = nx.betweenness_centrality(G)
    bc_cg = nx.betweenness_centrality(G, backend="cugraph")
    assert pytest.approx(bc_nx) == bc_cg


@pytest.mark.skipif(
    _nxver < (3, 5), reason="Test only supported for BC normalization used in NX 3.5+"
)
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
