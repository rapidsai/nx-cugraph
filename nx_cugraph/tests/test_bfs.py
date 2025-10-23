# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx


def test_generic_bfs_edges():
    # generic_bfs_edges currently isn't exercised by networkx tests
    Gnx = nx.karate_club_graph()
    Gcg = nx.karate_club_graph(backend="cugraph")
    for depth_limit in (0, 1, 2):
        for source in Gnx:
            # Some ordering is arbitrary, so I think there's a chance
            # this test may fail if networkx or nx-cugraph changes.
            nx_result = nx.generic_bfs_edges(Gnx, source, depth_limit=depth_limit)
            cg_result = nx.generic_bfs_edges(Gcg, source, depth_limit=depth_limit)
            assert sorted(nx_result) == sorted(cg_result), (source, depth_limit)
