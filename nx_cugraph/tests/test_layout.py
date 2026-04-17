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


def test_forceatlas2_extra_params():
    Gnx = nx.karate_club_graph()
    Gcg = nxcg.from_networkx(Gnx, preserve_all_attrs=True)

    node_mobility = {n: 1.0 for n in Gcg}

    res_cg = nxcg.forceatlas2_layout(
        Gcg,
        barnes_hut_optimize=True,
        barnes_hut_theta=0.5,
        node_mobility=node_mobility,
    )

    assert len(res_cg) == len(Gcg)
    assert res_cg.keys() == nx.nodes(Gcg)



def test_forceatlas2_max_iter_0():
    G = nx.cycle_graph(5)
    Gcg = nxcg.from_networkx(G, preserve_all_attrs=True)

    initial_positions = {
        0: (0.0, 1.0),
        1: (2.0, 3.0),
        2: (4.0, 5.0),
        3: (6.0, 7.0),
        4: (8.0, 9.0)
    }

    final_positions = nxcg.forceatlas2_layout(
        Gcg,
        pos=initial_positions,
        max_iter=0,
    )

    for node, pos in initial_positions.items():
        assert final_positions[node][0] == pos[0]
        assert final_positions[node][1] == pos[1]