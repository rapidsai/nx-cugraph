# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

import nx_cugraph as nxcg


def test_leiden_karate():
    # Basic smoke test; if something here changes, we want to know!
    G = nxcg.karate_club_graph()
    leiden = nxcg.community.leiden_communities(G, seed=123)
    louvain = nxcg.community.louvain_communities(G, seed=123)
    assert leiden == louvain
