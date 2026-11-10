# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import warnings
from functools import cache

import pytest

import _nx_cugraph


def test_normal_backend_info(monkeypatch):
    monkeypatch.delenv("_NETWORKX_BUILDING_DOCS_", raising=False)
    monkeypatch.delenv("NX_CUGRAPH_AUTOCONFIG", raising=False)
    monkeypatch.delenv("NX_CUGRAPH_USE_COMPAT_GRAPHS", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        info = _nx_cugraph.get_info()
    assert not caught
    assert info["functions"].keys() == _nx_cugraph._info["functions"]
    metadata = info["functions"]["eigenvector_centrality"]
    assert metadata["extra_docstring"] == metadata["additional_docs"]
    assert metadata["extra_parameters"] == metadata["additional_parameters"]
    assert info["default_config"] == {"use_compat_graphs": True}


@pytest.mark.parametrize("value", ["True", ""])
def test_docs_backend_info(monkeypatch, value):
    monkeypatch.delenv("_NETWORKX_BUILDING_DOCS_", raising=False)
    monkeypatch.delenv("NX_CUGRAPH_AUTOCONFIG", raising=False)
    normal_info = _nx_cugraph.get_info()
    monkeypatch.setenv("_NETWORKX_BUILDING_DOCS_", value)
    assert _nx_cugraph.get_info() == {**normal_info, "functions": {}}
    monkeypatch.delenv("_NETWORKX_BUILDING_DOCS_")
    assert _nx_cugraph.get_info() == normal_info


def test_autoconfig_warns_once(monkeypatch):
    monkeypatch.setattr(
        _nx_cugraph, "_warn_deprecated", cache(_nx_cugraph._warn_deprecated.__wrapped__)
    )
    monkeypatch.setenv("NX_CUGRAPH_AUTOCONFIG", "true")
    env_vars = (
        "NETWORKX_BACKEND_PRIORITY",
        "NETWORKX_BACKEND_PRIORITY_ALGOS",
        "NETWORKX_AUTOMATIC_BACKENDS",
        "NETWORKX_BACKEND_PRIORITY_GENERATORS",
        "NETWORKX_BACKEND_PRIORITY_CLASSES",
        "NETWORKX_FALLBACK_TO_NX",
        "NETWORKX_CACHE_CONVERTED_GRAPHS",
    )
    for name in env_vars:
        monkeypatch.delenv(name, raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _nx_cugraph.get_info()
        _nx_cugraph.get_info()
        _nx_cugraph._warn_deprecated()
    assert len(caught) == 1
    assert caught[0].category is FutureWarning
    # rapids-pre-commit-hooks: disable-next-line[verify-hardcoded-version]
    assert "26.12 will be its last release" in str(caught[0].message)
    assert "cuGraph" in str(caught[0].message)
