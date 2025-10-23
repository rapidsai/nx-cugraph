# SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import _nx_cugraph
import nx_cugraph


def test_version_constants_are_populated():
    # __git_commit__ will only be non-empty in a built distribution
    assert isinstance(nx_cugraph.__git_commit__, str)

    # __version__ should always be non-empty
    assert isinstance(nx_cugraph.__version__, str)
    assert len(nx_cugraph.__version__) > 0


def test_nx_ver():
    assert _nx_cugraph._check_networkx_version() == nx_cugraph._nxver
    assert _nx_cugraph._check_networkx_version("3.4") == (3, 4)
    assert _nx_cugraph._check_networkx_version("3.4.2") == (3, 4, 2)
    assert _nx_cugraph._check_networkx_version("3.4rc0") == (3, 4)
    assert _nx_cugraph._check_networkx_version("3.4.2rc1") == (3, 4, 2)
    assert _nx_cugraph._check_networkx_version("3.5rc0.dev0") == (3, 5)
    assert _nx_cugraph._check_networkx_version("3.5.1rc0.dev0") == (3, 5, 1)
    assert _nx_cugraph._check_networkx_version("3.5.dev0") == (3, 5)
    assert _nx_cugraph._check_networkx_version("3.5.1.dev0") == (3, 5, 1)
    with pytest.raises(ValueError, match="not enough values to unpack"):
        _nx_cugraph._check_networkx_version("3")
    with pytest.raises(RuntimeWarning, match="does not work with networkx version"):
        _nx_cugraph._check_networkx_version("3.4bad")
    with pytest.warns(
        UserWarning, match="only known to work with networkx versions 3.x"
    ):
        assert _nx_cugraph._check_networkx_version("2.2") == (2, 2)
    with pytest.warns(
        UserWarning, match="only known to work with networkx versions 3.x"
    ):
        assert _nx_cugraph._check_networkx_version("4.2") == (4, 2)
