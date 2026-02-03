# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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
