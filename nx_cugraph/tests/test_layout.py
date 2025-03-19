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
    # for each pos element in res_nx assert that it's equal to res_cg
