# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import pytest

import nx_cugraph as nxcg


@pytest.mark.parametrize(
    "get_graph", [nx.florentine_families_graph, nx.les_miserables_graph]
)
def test_k_truss(get_graph):
    Gnx = get_graph()
    Gcg = nxcg.from_networkx(Gnx, preserve_all_attrs=True)
    for k in range(6):
        Hnx = nx.k_truss(Gnx, k)
        Hcg = nxcg.k_truss(Gcg, k)
        assert nx.utils.graphs_equal(Hnx, nxcg.to_networkx(Hcg))
        if Hnx.number_of_edges() == 0:
            break
