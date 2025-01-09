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
from collections.abc import Iterable

import networkx as nx
import pytest

# The tests in this file cover use cases unique to nx-cugraph.  If the coverage
# here is not unique to nx-cugraph, consider moving those tests to the NetworkX
# project.


def test_no_nonexistent_edges_no_ebunch():
    """Test no ebunch and G is fully connected

    Ensure function returns iter([]) or equivalent due to no nonexistent edges.
    """
    G = nx.complete_graph(5)
    result = nx.jaccard_coefficient(G)
    assert isinstance(result, Iterable)
    assert pytest.raises(StopIteration, next, result)
