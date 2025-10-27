# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx

import nx_cugraph as nxcg


def assert_graphs_equal(Gnx, Gcg):
    assert isinstance(Gnx, nx.Graph)
    assert isinstance(Gcg, nxcg.CudaGraph)
    assert (a := Gnx.number_of_nodes()) == (b := Gcg.number_of_nodes()), (a, b)
    assert (a := Gnx.number_of_edges()) == (b := Gcg.number_of_edges()), (a, b)
    assert (a := Gnx.is_directed()) == (b := Gcg.is_directed()), (a, b)
    assert (a := Gnx.is_multigraph()) == (b := Gcg.is_multigraph()), (a, b)
    G = nxcg.to_networkx(Gcg)
    rv = nx.utils.graphs_equal(G, Gnx)
    if not rv:
        print("GRAPHS ARE NOT EQUAL!")
        assert sorted(G) == sorted(Gnx)
        assert sorted(G._adj) == sorted(Gnx._adj)
        assert sorted(G._node) == sorted(Gnx._node)
        for k in sorted(G._adj):
            print(k, sorted(G._adj[k]), sorted(Gnx._adj[k]))
        if len(G) > 0:
            print(nx.to_scipy_sparse_array(G).todense())
        else:
            print(G)
        if len(Gnx) > 0:
            print(nx.to_scipy_sparse_array(Gnx).todense())
        else:
            print(Gnx)
        print(f"{G.graph=}")
        print(f"{Gnx.graph=}")
    assert rv
