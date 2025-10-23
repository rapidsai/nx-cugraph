# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import pytest

import nx_cugraph as nxcg

from .testing_utils import assert_graphs_equal


@pytest.mark.parametrize(
    "create_using", [nx.Graph, nx.MultiGraph, nxcg.Graph, nxcg.MultiGraph]
)
def test_biadjacency_matrix_undirected(create_using):
    backend = getattr(create_using, "__networkx_backend__", "networkx")
    G = nx.bipartite.complete_bipartite_graph(
        3, 4, create_using=create_using, backend=backend
    )
    A = nx.bipartite.biadjacency_matrix(G, list(range(3)), list(range(3, 7)))
    H = nx.bipartite.from_biadjacency_matrix(
        A, create_using=create_using, backend=backend
    )
    assert type(G) is type(H)
    assert G.number_of_nodes() == H.number_of_nodes()
    assert G.number_of_edges() == H.number_of_edges()
    if backend == "cugraph":
        assert not G._is_on_cpu
        assert G._is_on_gpu
        assert not H._is_on_cpu
        assert H._is_on_gpu
        # Test that edges were properly symmetrized
        assert G._cudagraph.src_indices.size == H._cudagraph.src_indices.size
    assert G.nodes() == H.nodes()
    assert sorted(G.edges()) == sorted(H.edges())


@pytest.mark.parametrize(
    "create_using", [nx.DiGraph, nx.MultiDiGraph, nxcg.DiGraph, nxcg.MultiDiGraph]
)
def test_biadjacency_matrix_directed(create_using):
    backend = getattr(create_using, "__networkx_backend__", "networkx")
    graph_class = {
        nx.DiGraph: nx.Graph,
        nxcg.DiGraph: nxcg.Graph,
        nx.MultiDiGraph: nx.MultiGraph,
        nxcg.MultiDiGraph: nxcg.MultiGraph,
    }[create_using]
    G = nx.bipartite.complete_bipartite_graph(
        3, 4, create_using=graph_class, backend=backend
    ).to_directed()
    A = nx.bipartite.biadjacency_matrix(G, list(range(3)), list(range(3, 7)))
    H = nx.bipartite.from_biadjacency_matrix(
        A, create_using=create_using, backend=backend
    )
    assert type(G) is type(H)
    assert G.number_of_nodes() == H.number_of_nodes()
    assert G.number_of_edges() == 2 * H.number_of_edges()
    assert G.nodes() == H.nodes()
    H.add_edges_from(H.reverse().edges())
    assert sorted(G.edges()) == sorted(H.edges())
    if backend == "cugraph":
        Gnx = nx.bipartite.complete_bipartite_graph(
            3, 4, create_using=graph_class.to_networkx_class()
        ).to_directed()
        assert_graphs_equal(Gnx, G._cudagraph)
        Anx = nx.bipartite.biadjacency_matrix(Gnx, list(range(3)), list(range(3, 7)))
        assert (Anx != A).nnz == 0
        Hnx = nx.bipartite.from_biadjacency_matrix(
            Anx, create_using=create_using.to_networkx_class()
        )
        Hnx.add_edges_from(Hnx.reverse().edges())
        assert_graphs_equal(Hnx, H._cudagraph)


@pytest.mark.parametrize("create_using", [nx.Graph, nxcg.Graph])
def test_biadjacency_matrix_empty(create_using):
    backend = getattr(create_using, "__networkx_backend__", "networkx")
    G = nx.empty_graph(5, create_using=create_using, backend=backend)
    A = nx.bipartite.biadjacency_matrix(G, [0, 1])
    assert A.nnz == 0
    H = nx.bipartite.from_biadjacency_matrix(
        A, create_using=create_using, backend=backend
    )
    assert type(G) is type(H)
    assert G.number_of_nodes() == H.number_of_nodes() == 5
    assert G.number_of_edges() == H.number_of_edges() == 0
