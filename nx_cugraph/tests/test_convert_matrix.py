# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import networkx as nx
import pandas as pd
import pytest

import nx_cugraph as nxcg
from nx_cugraph.utils import _cp_iscopied_asarray

try:
    import cudf
except ModuleNotFoundError:
    cudf = None


DATA = [
    {"source": [0, 1], "target": [1, 2]},  # nodes are 0, 1, 2
    {"source": [0, 1], "target": [1, 3]},  # nodes are 0, 1, 3 (need renumbered!)
    {"source": ["a", "b"], "target": ["b", "c"]},  # nodes are 'a', 'b', 'c'
]
CREATE_USING = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]


@pytest.mark.skipif("not cudf")
@pytest.mark.parametrize("data", DATA)
@pytest.mark.parametrize("create_using", CREATE_USING)
def test_from_cudf_edgelist(data, create_using):
    df = cudf.DataFrame(data)
    nxcg.from_pandas_edgelist(df, create_using=create_using)  # Basic smoke test
    source = df["source"]
    if source.dtype == int:
        is_copied, src_array = _cp_iscopied_asarray(source)
        assert is_copied is False
        is_copied, src_array = _cp_iscopied_asarray(source.to_cupy())
        assert is_copied is False
        is_copied, src_array = _cp_iscopied_asarray(source, orig_object=source)
        assert is_copied is False
        is_copied, src_array = _cp_iscopied_asarray(
            source.to_cupy(), orig_object=source
        )
        assert is_copied is False
        # to numpy
        is_copied, src_array = _cp_iscopied_asarray(source.to_numpy())
        assert is_copied is True
        is_copied, src_array = _cp_iscopied_asarray(
            source.to_numpy(), orig_object=source
        )
        assert is_copied is True
    else:
        with pytest.raises(TypeError):
            _cp_iscopied_asarray(source)
        with pytest.raises(TypeError):
            _cp_iscopied_asarray(source.to_cupy())
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _cp_iscopied_asarray(source.to_numpy())
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _cp_iscopied_asarray(source.to_numpy(), orig_object=source)


@pytest.mark.parametrize("data", DATA)
@pytest.mark.parametrize("create_using", CREATE_USING)
def test_from_pandas_edgelist(data, create_using):
    df = pd.DataFrame(data)
    nxcg.from_pandas_edgelist(df, create_using=create_using)  # Basic smoke test
    source = df["source"]
    if source.dtype == int:
        is_copied, src_array = _cp_iscopied_asarray(source)
        assert is_copied is True
        is_copied, src_array = _cp_iscopied_asarray(source, orig_object=source)
        assert is_copied is True
        is_copied, src_array = _cp_iscopied_asarray(source.to_numpy())
        assert is_copied is True
        is_copied, src_array = _cp_iscopied_asarray(
            source.to_numpy(), orig_object=source
        )
        assert is_copied is True
