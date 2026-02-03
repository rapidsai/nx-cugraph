# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import pandas as pd
import pytest


def test_pagerank_multigraph():
    """
    Ensures correct pagerank for Graphs and MultiGraphs when using from_pandas_edgelist.

    PageRank for MultiGraph should give different result compared to Graph; when using
    a Graph, the duplicate edges should be dropped.
    """
    df = pd.DataFrame(
        {"source": [0, 1, 1, 1, 1, 1, 1, 2], "target": [1, 2, 2, 2, 2, 2, 2, 3]}
    )
    expected_pr_for_G = nx.pagerank(nx.from_pandas_edgelist(df))
    expected_pr_for_MultiG = nx.pagerank(
        nx.from_pandas_edgelist(df, create_using=nx.MultiGraph)
    )

    G = nx.from_pandas_edgelist(df, backend="cugraph")
    actual_pr_for_G = nx.pagerank(G, backend="cugraph")

    MultiG = nx.from_pandas_edgelist(df, create_using=nx.MultiGraph, backend="cugraph")
    actual_pr_for_MultiG = nx.pagerank(MultiG, backend="cugraph")

    assert actual_pr_for_G == pytest.approx(expected_pr_for_G)
    assert actual_pr_for_MultiG == pytest.approx(expected_pr_for_MultiG)
