# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import networkx as nx

from nx_cugraph import _nxver


def install():
    """Enable NetworkX acceleration with nx-cugraph."""

    # See also handling of NX_CUGRAPH_AUTOCONFIG env var in _nx_cugraph/__init__.py
    def update_priority(priority):
        while priority and "cugraph" in priority:
            priority.remove("cugraph")
        priority.insert(0, "cugraph")

    if _nxver < (3, 3):
        # No config available; modify internal state.
        update_priority(nx.utils._dispatch._automatic_backends)
        nx.utils._dispatch._fallback_to_nx = True
        return
        # Or we could raise...
        # raise RuntimeError(
        #     f"NetworkX version {_nxver} not supported by `nx_cugraph.accel.install`; "
        #     "NetworkX version 3.3 or greater is required for this functionality."
        # )

    cfg = nx.config
    if isinstance(cfg.backend_priority, list):
        update_priority(cfg.backend_priority)
    else:
        update_priority(cfg.backend_priority.algos)
        update_priority(cfg.backend_priority.generators)
    if "fallback_to_nx" in cfg:
        cfg.fallback_to_nx = True
    if "cache_converted_graphs" in cfg:
        cfg.cache_converted_graphs = True
    nx.config.backends.cugraph.use_compat_graphs = True
