# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx


def test_selfloops():
    G = nx.complete_graph(5)
    H = nx.complete_graph(5)
    H.add_edge(0, 0)
    H.add_edge(1, 1)
    H.add_edge(2, 2)
    # triangles
    expected = nx.triangles(G)
    assert expected == nx.triangles(H)
    assert expected == nx.triangles(G, backend="cugraph")
    assert expected == nx.triangles(H, backend="cugraph")
    # average_clustering
    expected = nx.average_clustering(G)
    assert expected == nx.average_clustering(H)
    assert expected == nx.average_clustering(G, backend="cugraph")
    assert expected == nx.average_clustering(H, backend="cugraph")
    # clustering
    expected = nx.clustering(G)
    assert expected == nx.clustering(H)
    assert expected == nx.clustering(G, backend="cugraph")
    assert expected == nx.clustering(H, backend="cugraph")
    # transitivity
    expected = nx.transitivity(G)
    assert expected == nx.transitivity(H)
    assert expected == nx.transitivity(G, backend="cugraph")
    assert expected == nx.transitivity(H, backend="cugraph")
