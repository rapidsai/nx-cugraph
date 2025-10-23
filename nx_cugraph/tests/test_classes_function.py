# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Test functions from nx_cugraph/classes/function.py"""
import networkx as nx

import nx_cugraph as nxcg


def test_is_negatively_weighted():
    Gnx = nx.MultiGraph()
    Gnx.add_edge(0, 1, 2, weight=-3)
    Gnx.add_edge(2, 3, foo=3)
    Gcg = nxcg.from_networkx(Gnx, preserve_edge_attrs=True)
    assert nx.is_negatively_weighted(Gnx)
    assert nxcg.is_negatively_weighted(Gnx)
    assert nxcg.is_negatively_weighted(Gcg)
    assert not nx.is_negatively_weighted(Gnx, weight="foo")
    assert not nxcg.is_negatively_weighted(Gcg, weight="foo")
    assert not nx.is_negatively_weighted(Gnx, weight="bar")
    assert not nxcg.is_negatively_weighted(Gcg, weight="bar")
    assert nx.is_negatively_weighted(Gnx, (0, 1, 2))
    assert nxcg.is_negatively_weighted(Gcg, (0, 1, 2))
    assert nx.is_negatively_weighted(Gnx, (0, 1)) == nxcg.is_negatively_weighted(
        Gcg, (0, 1)
    )
