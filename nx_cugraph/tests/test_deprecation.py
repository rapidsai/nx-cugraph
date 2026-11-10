# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("autoconfig", ["false", "true"])
def test_import_warns_once(autoconfig):
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-W",
            "always::FutureWarning",
            "-c",
            "import networkx; import nx_cugraph; import nx_cugraph; "
            "nx_cugraph.Graph.is_directed(); nx_cugraph.Graph.is_directed()",
        ],
        env={**os.environ, "NX_CUGRAPH_AUTOCONFIG": autoconfig},
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stderr.count("nx-cugraph is deprecated.") == 1


def test_networkx_docs_omit_backend():
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-c",
            "import networkx as nx; "
            "assert nx.utils.backends.backend_info['cugraph']['functions'] == {}; "
            "assert 'cugraph' not in nx.triangles.__doc__",
        ],
        env={
            **os.environ,
            "_NETWORKX_BUILDING_DOCS_": "True",
            "NX_CUGRAPH_AUTOCONFIG": "false",
        },
        check=True,
    )
