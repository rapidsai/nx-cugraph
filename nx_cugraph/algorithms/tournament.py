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

from nx_cugraph.linalg import adjacency_matrix
from nx_cugraph.utils import networkx_algorithm, not_implemented_for

__all__ = []


@not_implemented_for("undirected")
@not_implemented_for("multigraph")
@networkx_algorithm(version_added="25.06")
def tournament_matrix(G):
    A = adjacency_matrix(G)
    # TODO: do this on the GPU (but cupyx.scipy.sparse is pretty limited atm)
    # Once we are able to perform sparse array operations on the GPU, we can
    # optimize this function and implement many others that return sparse arrays!
    return A - A.T
