# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import nx_cugraph as nxcg


def test_leiden_karate():
    # Basic smoke test; if something here changes, we want to know!
    G = nxcg.karate_club_graph()
    leiden = nxcg.community.leiden_communities(G, seed=123, metric="modularity")
    louvain = nxcg.community.louvain_communities(G, seed=123)
    assert leiden == louvain
