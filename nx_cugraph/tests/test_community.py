# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx

import nx_cugraph as nxcg


def test_louvain_isolated_nodes():

    def check(left, right):
        assert len(left) == len(right)
        assert set(map(frozenset, left)) == set(map(frozenset, right))

    # Empty graph (no nodes)
    G = nx.Graph()
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
    # Graph with no edges
    G.add_nodes_from(range(5))
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
    # Graph with isolated nodes
    G.add_edge(1, 2)
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
    # Another one
    G.add_edge(4, 4)
    nx_result = nx.community.louvain_communities(G)
    cg_result = nxcg.community.louvain_communities(G)
    check(nx_result, cg_result)
