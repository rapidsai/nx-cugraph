# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import math

import networkx as nx

import nx_cugraph as nxcg

if __name__ == "__main__":
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3)])

    nx_result = nx.betweenness_centrality(G)
    # nx_cugraph is intended to be called via the NetworkX dispatcher, like
    # this:
    #    nxcu_result = nx.betweenness_centrality(G, backend="cugraph")
    #
    # but here it is being called directly since the NetworkX version that
    # supports the "backend" kwarg may not be available in the testing env.
    nxcu_result = nxcg.betweenness_centrality(G)

    nx_nodes, nxcu_nodes = nx_result.keys(), nxcu_result.keys()
    assert nxcu_nodes == nx_nodes
    for node_id in nx_nodes:
        nx_bc, nxcu_bc = nx_result[node_id], nxcu_result[node_id]
        assert math.isclose(
            nx_bc, nxcu_bc, rel_tol=1e-6
        ), f"bc for {node_id=} exceeds tolerance: {nx_bc=}, {nxcu_bc=}"
