# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import networkx as nx

import nx_cugraph as nxcg
from nx_cugraph import _nxver


def test_leiden_karate():
    # Basic smoke test; if something here changes, we want to know!
    G = nxcg.karate_club_graph()
    leiden = nxcg.community.leiden_communities(G, seed=123, metric="modularity")
    louvain = nxcg.community.louvain_communities(G, seed=123)
    assert leiden == louvain


def test_leiden_can_run_networkx_37():
    if _nxver < (3, 7):
        return
    reason = nxcg.interface.BackendInterface.leiden_communities.can_run(
        nx.path_graph(2)
    )
    assert isinstance(reason, str)
