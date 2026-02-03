# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import networkx as nx
import pytest

import nx_cugraph as nxcg

# The tests in this file cover use cases unique to nx-cugraph.  If the coverage
# here is not unique to nx-cugraph, consider moving those tests to the NetworkX
# project.


@pytest.mark.parametrize("get_graph", [nx.karate_club_graph, nx.les_miserables_graph])
def test_forceatlas2(get_graph):
    Gnx = get_graph()
    Gcg = nxcg.from_networkx(Gnx, preserve_all_attrs=True)

    res_nx = nx.forceatlas2_layout(Gnx)
    res_cg = nxcg.forceatlas2_layout(Gcg)

    assert len(res_nx) == len(res_cg)
    assert res_nx.keys() == res_cg.keys()
