# <div align="left"><img src="img/rapids_logo.png" width="90px"/>&nbsp;nx-cugraph - GPU Backend for NetworkX</div>

## Description
[nx-cugraph](https://rapids.ai/nx-cugraph) is a [backend to NetworkX](https://networkx.org/documentation/stable/backends.html) to run algorithms with zero code change GPU acceleration.

## 🔍 Try it in Google Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rapidsai/nx-cugraph/blob/HEAD/notebooks/demo/nx_cugraph_demo_2506.ipynb)

---

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Enabling nx-cugraph](#enabling-nx-cugraph)
- [Supported Algorithms](#supported-algorithms)


## System Requirements

 * **GPU:** NVIDIA Volta architecture or later, with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
   * Pascal GPU support was [removed in 24.02](https://docs.rapids.ai/notices/rsn0034/). Compute capability 7.0+ is required for RAPIDS 24.02 and later.
 * **CUDA Version:** 11.4 - 11.8 or 12.0 - 12.5
 * **Python Version:** 3.10, 3.11, or 3.12
 * **NetworkX Version:** minimum 3.2 (version 3.4 or higher recommended)

Note: nx-cugraph is supported only on Linux, and with Python versions 3.10 and later.

See [RAPIDS System Requirements](https://docs.rapids.ai/install#system-req) for detailed information on OS and Versions.


## Installation

nx-cugraph can be installed using either conda or pip with the following commands.

### conda

nx-cugraph can be installed with conda (via [Miniforge](https://github.com/conda-forge/miniforge)) from the `rapidsai` channel.
```
conda install -c rapidsai -c conda-forge -c nvidia nx-cugraph
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly/nx-cugraph) built from the HEAD of our latest development branch.
```
conda install -c rapidsai-nightly -c conda-forge -c nvidia nx-cugraph
```

### pip

nx-cugraph can be installed via `pip` from the NVIDIA Python Package Index.

#### For CUDA 11.x:

Latest nightly version
```
python -m pip install nx-cugraph-cu11 --extra-index-url https://pypi.anaconda.org/rapidsai-wheels-nightly/simple
```

Latest stable version
```
python -m pip install nx-cugraph-cu11 --extra-index-url https://pypi.nvidia.com
```
Notes:
 * The pip example above installs for CUDA 11. To install for CUDA 12, replace `-cu11` with `-cu12`
 * Try out the [RAPIDS Install Selector Tool](https://docs.rapids.ai/install/#install-rapids) to install other RAPIDS packages.

## Enabling nx-cugraph

NetworkX will use nx-cugraph as the graph analytics backend if any of the
following are used:

### `NX_CUGRAPH_AUTOCONFIG` environment variable.
By setting `NX_CUGRAPH_AUTOCONFIG=True`, NetworkX will automatically dispatch algorithm calls to nx-cugraph (if the backend is supported). This allows users to GPU accelerate their code with zero code change.

Read more on [Networkx Backends and How They Work](https://networkx.org/documentation/stable/reference/backends.html).

Example:
```
bash> NX_CUGRAPH_AUTOCONFIG=True python my_networkx_script.py
```

### `backend=` keyword argument
To explicitly specify a particular backend for an API, use the `backend=`
keyword argument. This argument takes precedence over the
`NX_CUGRAPH_AUTOCONFIG` environment variable. This requires anyone
running code that uses the `backend=` keyword argument to have the specified
backend installed.

Example:
```
nx.betweenness_centrality(cit_patents_graph, k=k, backend="cugraph")
```

### Type-based dispatching

NetworkX also supports automatically dispatching to backends associated with
specific graph types. Like the `backend=` keyword argument example above, this
requires the user to write code for a specific backend, and therefore requires
the backend to be installed, but has the advantage of ensuring a particular
behavior without the potential for runtime conversions.

To use type-based dispatching with nx-cugraph, the user must import the backend
directly in their code to access the utilities provided to create a Graph
instance specifically for the nx-cugraph backend.

Example:
```
import networkx as nx
import nx_cugraph as nxcg

G = nx.Graph()
...
nxcg_G = nxcg.from_networkx(G)             # conversion happens once here
nx.betweenness_centrality(nxcg_G, k=1000)  # nxcg Graph type causes cugraph backend
                                           # to be used, no conversion necessary
```

## Supported Algorithms

The nx-cugraph backend to NetworkX connects
[pylibcugraph](https://github.com/rapidsai/cugraph/blob/-/readme_pages/pylibcugraph.md) (cuGraph's low-level python
interface to its CUDA-based graph analytics library) and
[CuPy](https://cupy.dev/) (a GPU-accelerated array library) to NetworkX's
familiar and easy-to-use API.

Below is the list of algorithms that are currently supported in nx-cugraph.

### [Algorithms](https://networkx.org/documentation/latest/reference/algorithms/index.html)

<pre>
<a href="https://networkx.org/documentation/stable/reference/algorithms/bipartite.html#module-networkx.algorithms.bipartite">bipartite</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/bipartite.html#module-networkx.algorithms.bipartite.centrality">centrality</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.centrality.betweenness_centrality.html#networkx.algorithms.bipartite.centrality.betweenness_centrality">betweenness_centrality</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/bipartite.html#module-networkx.algorithms.bipartite.generators">generators</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.generators.complete_bipartite_graph.html#networkx.algorithms.bipartite.generators.complete_bipartite_graph">complete_bipartite_graph</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/bipartite.html#module-networkx.algorithms.bipartite.matrix">matrix</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.matrix.biadjacency_matrix.html#networkx.algorithms.bipartite.matrix.biadjacency_matrix">biadjacency_matrix</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.bipartite.matrix.from_biadjacency_matrix.html#networkx.algorithms.bipartite.matrix.from_biadjacency_matrix">from_biadjacency_matrix</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/centrality.html#module-networkx.algorithms.centrality">centrality</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/centrality.html#networkx-algorithms-centrality-betweenness">betweenness</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.betweenness_centrality.html#networkx.algorithms.centrality.betweenness_centrality">betweenness_centrality</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.edge_betweenness_centrality.html#networkx.algorithms.centrality.edge_betweenness_centrality">edge_betweenness_centrality</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/centrality.html#networkx-algorithms-centrality-degree-alg">degree_alg</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.degree_centrality.html#networkx.algorithms.centrality.degree_centrality">degree_centrality</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.in_degree_centrality.html#networkx.algorithms.centrality.in_degree_centrality">in_degree_centrality</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.out_degree_centrality.html#networkx.algorithms.centrality.out_degree_centrality">out_degree_centrality</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/centrality.html#networkx-algorithms-centrality-eigenvector">eigenvector</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.eigenvector_centrality.html#networkx.algorithms.centrality.eigenvector_centrality">eigenvector_centrality</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/centrality.html#networkx-algorithms-centrality-katz">katz</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.centrality.katz_centrality.html#networkx.algorithms.centrality.katz_centrality">katz_centrality</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/clustering.html#module-networkx.algorithms.cluster">cluster</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.average_clustering.html#networkx.algorithms.cluster.average_clustering">average_clustering</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.clustering.html#networkx.algorithms.cluster.clustering">clustering</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.transitivity.html#networkx.algorithms.cluster.transitivity">transitivity</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cluster.triangles.html#networkx.algorithms.cluster.triangles">triangles</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/community.html#module-networkx.algorithms.community">community</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/community.html#module-networkx.algorithms.community.leiden">leiden</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.leiden.leiden_communities.html#networkx.algorithms.community.leiden.leiden_communities">leiden_communities</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/community.html#module-networkx.algorithms.community.louvain">louvain</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html#networkx.algorithms.community.louvain.louvain_communities">louvain_communities</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/component.html#module-networkx.algorithms.components">components</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/component.html#networkx-algorithms-components-connected">connected</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.connected_components.html#networkx.algorithms.components.connected_components">connected_components</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.is_connected.html#networkx.algorithms.components.is_connected">is_connected</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.node_connected_component.html#networkx.algorithms.components.node_connected_component">node_connected_component</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.number_connected_components.html#networkx.algorithms.components.number_connected_components">number_connected_components</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/component.html#networkx-algorithms-components-weakly-connected">weakly_connected</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.is_weakly_connected.html#networkx.algorithms.components.is_weakly_connected">is_weakly_connected</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.number_weakly_connected_components.html#networkx.algorithms.components.number_weakly_connected_components">number_weakly_connected_components</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.components.weakly_connected_components.html#networkx.algorithms.components.weakly_connected_components">weakly_connected_components</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/core.html#module-networkx.algorithms.core">core</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.core_number.html#networkx.algorithms.core.core_number">core_number</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.k_truss.html#networkx.algorithms.core.k_truss">k_truss</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/dag.html#module-networkx.algorithms.dag">dag</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.ancestors.html#networkx.algorithms.dag.ancestors">ancestors</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.descendants.html#networkx.algorithms.dag.descendants">descendants</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/isolates.html#module-networkx.algorithms.isolate">isolate</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isolate.is_isolate.html#networkx.algorithms.isolate.is_isolate">is_isolate</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isolate.isolates.html#networkx.algorithms.isolate.isolates">isolates</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.isolate.number_of_isolates.html#networkx.algorithms.isolate.number_of_isolates">number_of_isolates</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/link_analysis.html">link_analysis</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/link_analysis.html#module-networkx.algorithms.link_analysis.hits_alg">hits_alg</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.hits_alg.hits.html#networkx.algorithms.link_analysis.hits_alg.hits">hits</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/link_analysis.html#module-networkx.algorithms.link_analysis.pagerank_alg">pagerank_alg</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_analysis.pagerank_alg.pagerank.html#networkx.algorithms.link_analysis.pagerank_alg.pagerank">pagerank</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html#module-networkx.algorithms.link_prediction">link_prediction</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.link_prediction.jaccard_coefficient.html#networkx.algorithms.link_prediction.jaccard_coefficient">jaccard_coefficient</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/lowest_common_ancestors.html#module-networkx.algorithms.lowest_common_ancestors">lowest_common_ancestors</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.lowest_common_ancestors.lowest_common_ancestor.html#networkx.algorithms.lowest_common_ancestors.lowest_common_ancestor">lowest_common_ancestor</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/operators.html">operators</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/operators.html#module-networkx.algorithms.operators.unary">unary</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.unary.complement.html#networkx.algorithms.operators.unary.complement">complement</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.operators.unary.reverse.html#networkx.algorithms.operators.unary.reverse">reverse</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/reciprocity.html#module-networkx.algorithms.reciprocity">reciprocity</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.reciprocity.overall_reciprocity.html#networkx.algorithms.reciprocity.overall_reciprocity">overall_reciprocity</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.reciprocity.reciprocity.html#networkx.algorithms.reciprocity.reciprocity">reciprocity</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html">shortest_paths</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html#module-networkx.algorithms.shortest_paths.generic">generic</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.has_path.html#networkx.algorithms.shortest_paths.generic.has_path">has_path</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path.html#networkx.algorithms.shortest_paths.generic.shortest_path">shortest_path</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.generic.shortest_path_length.html#networkx.algorithms.shortest_paths.generic.shortest_path_length">shortest_path_length</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html#module-networkx.algorithms.shortest_paths.unweighted">unweighted</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path.html#networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path">all_pairs_shortest_path</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length.html#networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path_length">all_pairs_shortest_path_length</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.bidirectional_shortest_path.html#networkx.algorithms.shortest_paths.unweighted.bidirectional_shortest_path">bidirectional_shortest_path</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path.html#networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path">single_source_shortest_path</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path_length.html#networkx.algorithms.shortest_paths.unweighted.single_source_shortest_path_length">single_source_shortest_path_length</a>
 │   ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.single_target_shortest_path.html#networkx.algorithms.shortest_paths.unweighted.single_target_shortest_path">single_target_shortest_path</a>
 │   └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.single_target_shortest_path_length.html#networkx.algorithms.shortest_paths.unweighted.single_target_shortest_path_length">single_target_shortest_path_length</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html#module-networkx.algorithms.shortest_paths.weighted">weighted</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path.html#networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path">all_pairs_bellman_ford_path</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path_length.html#networkx.algorithms.shortest_paths.weighted.all_pairs_bellman_ford_path_length">all_pairs_bellman_ford_path_length</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra">all_pairs_dijkstra</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path">all_pairs_dijkstra_path</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra_path_length">all_pairs_dijkstra_path_length</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.bellman_ford_path.html#networkx.algorithms.shortest_paths.weighted.bellman_ford_path">bellman_ford_path</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.bellman_ford_path_length.html#networkx.algorithms.shortest_paths.weighted.bellman_ford_path_length">bellman_ford_path_length</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path.html#networkx.algorithms.shortest_paths.weighted.dijkstra_path">dijkstra_path</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.dijkstra_path_length.html#networkx.algorithms.shortest_paths.weighted.dijkstra_path_length">dijkstra_path_length</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford.html#networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford">single_source_bellman_ford</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford_path.html#networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford_path">single_source_bellman_ford_path</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford_path_length.html#networkx.algorithms.shortest_paths.weighted.single_source_bellman_ford_path_length">single_source_bellman_ford_path_length</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_dijkstra.html#networkx.algorithms.shortest_paths.weighted.single_source_dijkstra">single_source_dijkstra</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path.html#networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path">single_source_dijkstra_path</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length.html#networkx.algorithms.shortest_paths.weighted.single_source_dijkstra_path_length">single_source_dijkstra_path_length</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/tournament.html#module-networkx.algorithms.tournament">tournament</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tournament.tournament_matrix.html#networkx.algorithms.tournament.tournament_matrix">tournament_matrix</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/traversal.html">traversal</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/traversal.html#module-networkx.algorithms.traversal.breadth_first_search">breadth_first_search</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_edges.html#networkx.algorithms.traversal.breadth_first_search.bfs_edges">bfs_edges</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_layers.html#networkx.algorithms.traversal.breadth_first_search.bfs_layers">bfs_layers</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_predecessors.html#networkx.algorithms.traversal.breadth_first_search.bfs_predecessors">bfs_predecessors</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_successors.html#networkx.algorithms.traversal.breadth_first_search.bfs_successors">bfs_successors</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.bfs_tree.html#networkx.algorithms.traversal.breadth_first_search.bfs_tree">bfs_tree</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.descendants_at_distance.html#networkx.algorithms.traversal.breadth_first_search.descendants_at_distance">descendants_at_distance</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.traversal.breadth_first_search.generic_bfs_edges.html#networkx.algorithms.traversal.breadth_first_search.generic_bfs_edges">generic_bfs_edges</a>
<a href="https://networkx.org/documentation/stable/reference/algorithms/tree.html">tree</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/tree.html#module-networkx.algorithms.tree.recognition">recognition</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.recognition.is_arborescence.html#networkx.algorithms.tree.recognition.is_arborescence">is_arborescence</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.recognition.is_branching.html#networkx.algorithms.tree.recognition.is_branching">is_branching</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.recognition.is_forest.html#networkx.algorithms.tree.recognition.is_forest">is_forest</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.tree.recognition.is_tree.html#networkx.algorithms.tree.recognition.is_tree">is_tree</a>
</pre>

### [Generators](https://networkx.org/documentation/latest/reference/generators.html)

<pre>
<a href="https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.classic">classic</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.barbell_graph.html#networkx.generators.classic.barbell_graph">barbell_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.circular_ladder_graph.html#networkx.generators.classic.circular_ladder_graph">circular_ladder_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.complete_graph.html#networkx.generators.classic.complete_graph">complete_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.complete_multipartite_graph.html#networkx.generators.classic.complete_multipartite_graph">complete_multipartite_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.cycle_graph.html#networkx.generators.classic.cycle_graph">cycle_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.empty_graph.html#networkx.generators.classic.empty_graph">empty_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.ladder_graph.html#networkx.generators.classic.ladder_graph">ladder_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.lollipop_graph.html#networkx.generators.classic.lollipop_graph">lollipop_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.null_graph.html#networkx.generators.classic.null_graph">null_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.path_graph.html#networkx.generators.classic.path_graph">path_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.star_graph.html#networkx.generators.classic.star_graph">star_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.tadpole_graph.html#networkx.generators.classic.tadpole_graph">tadpole_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.trivial_graph.html#networkx.generators.classic.trivial_graph">trivial_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.turan_graph.html#networkx.generators.classic.turan_graph">turan_graph</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.classic.wheel_graph.html#networkx.generators.classic.wheel_graph">wheel_graph</a>
<a href="https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.community">community</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.caveman_graph.html#networkx.generators.community.caveman_graph">caveman_graph</a>
<a href="https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.ego">ego</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.ego.ego_graph.html#networkx.generators.ego.ego_graph">ego_graph</a>
<a href="https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.small">small</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.bull_graph.html#networkx.generators.small.bull_graph">bull_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.chvatal_graph.html#networkx.generators.small.chvatal_graph">chvatal_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.cubical_graph.html#networkx.generators.small.cubical_graph">cubical_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.desargues_graph.html#networkx.generators.small.desargues_graph">desargues_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.diamond_graph.html#networkx.generators.small.diamond_graph">diamond_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.dodecahedral_graph.html#networkx.generators.small.dodecahedral_graph">dodecahedral_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.frucht_graph.html#networkx.generators.small.frucht_graph">frucht_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.heawood_graph.html#networkx.generators.small.heawood_graph">heawood_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.house_graph.html#networkx.generators.small.house_graph">house_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.house_x_graph.html#networkx.generators.small.house_x_graph">house_x_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.icosahedral_graph.html#networkx.generators.small.icosahedral_graph">icosahedral_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.krackhardt_kite_graph.html#networkx.generators.small.krackhardt_kite_graph">krackhardt_kite_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.moebius_kantor_graph.html#networkx.generators.small.moebius_kantor_graph">moebius_kantor_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.octahedral_graph.html#networkx.generators.small.octahedral_graph">octahedral_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.pappus_graph.html#networkx.generators.small.pappus_graph">pappus_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.petersen_graph.html#networkx.generators.small.petersen_graph">petersen_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.sedgewick_maze_graph.html#networkx.generators.small.sedgewick_maze_graph">sedgewick_maze_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.tetrahedral_graph.html#networkx.generators.small.tetrahedral_graph">tetrahedral_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.truncated_cube_graph.html#networkx.generators.small.truncated_cube_graph">truncated_cube_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.truncated_tetrahedron_graph.html#networkx.generators.small.truncated_tetrahedron_graph">truncated_tetrahedron_graph</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.small.tutte_graph.html#networkx.generators.small.tutte_graph">tutte_graph</a>
<a href="https://networkx.org/documentation/stable/reference/generators.html#module-networkx.generators.social">social</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.davis_southern_women_graph.html#networkx.generators.social.davis_southern_women_graph">davis_southern_women_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.florentine_families_graph.html#networkx.generators.social.florentine_families_graph">florentine_families_graph</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.karate_club_graph.html#networkx.generators.social.karate_club_graph">karate_club_graph</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.generators.social.les_miserables_graph.html#networkx.generators.social.les_miserables_graph">les_miserables_graph</a>
</pre>

### Other

<pre>
<a href="https://networkx.org/documentation/stable/reference/classes/index.html">classes</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/functions.html#module-networkx.classes.function">function</a>
     ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.is_negatively_weighted.html#networkx.classes.function.is_negatively_weighted">is_negatively_weighted</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.classes.function.number_of_selfloops.html#networkx.classes.function.number_of_selfloops">number_of_selfloops</a>
<a href="https://networkx.org/documentation/stable/reference/convert.html#module-networkx.convert">convert</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.convert.from_dict_of_lists.html#networkx.convert.from_dict_of_lists">from_dict_of_lists</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.convert.to_dict_of_lists.html#networkx.convert.to_dict_of_lists">to_dict_of_lists</a>
<a href="https://networkx.org/documentation/stable/reference/convert.html#module-networkx.convert_matrix">convert_matrix</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_pandas_edgelist.html#networkx.convert_matrix.from_pandas_edgelist">from_pandas_edgelist</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.from_scipy_sparse_array.html#networkx.convert_matrix.from_scipy_sparse_array">from_scipy_sparse_array</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.to_numpy_array.html#networkx.convert_matrix.to_numpy_array">to_numpy_array</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.convert_matrix.to_scipy_sparse_array.html#networkx.convert_matrix.to_scipy_sparse_array">to_scipy_sparse_array</a>
<a href="https://networkx.org/documentation/stable/reference/drawing.html">drawing</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/drawing.html#module-networkx.drawing.layout">layout</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.forceatlas2_layout.html#networkx.drawing.layout.forceatlas2_layout">forceatlas2_layout</a>
<a href="https://networkx.org/documentation/stable/reference/linalg.html">linalg</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/linalg.html#module-networkx.linalg.graphmatrix">graphmatrix</a>
     └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.linalg.graphmatrix.adjacency_matrix.html#networkx.linalg.graphmatrix.adjacency_matrix">adjacency_matrix</a>
<a href="https://networkx.org/documentation/stable/reference/relabel.html#module-networkx.relabel">relabel</a>
 ├─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.relabel.convert_node_labels_to_integers.html#networkx.relabel.convert_node_labels_to_integers">convert_node_labels_to_integers</a>
 └─ <a href="https://networkx.org/documentation/stable/reference/generated/networkx.relabel.relabel_nodes.html#networkx.relabel.relabel_nodes">relabel_nodes</a>
</pre>

To request nx-cugraph backend support for a NetworkX API that is not listed
above, [file an issue](https://github.com/rapidsai/nx-cugraph/issues/new/choose).

## Contributing

If you would like to contribute to nx-cugraph, refer to the [Contributing Guide](./CONTRIBUTING.md)
