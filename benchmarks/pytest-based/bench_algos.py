# Copyright (c) 2023-2025, NVIDIA CORPORATION.
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


from collections.abc import Mapping

import networkx as nx
import pandas as pd
import pytest
from cugraph import datasets

import nx_cugraph as nxcg

################################################################################
# Fixtures and params

# See https://pytest-benchmark.readthedocs.io/en/latest/glossary.html for how
# these variables are used.
rounds = 1
iterations = 1
warmup_rounds = 1

dataset_param_values = [
    # name: karate, nodes: 34, edges: 156
    pytest.param(datasets.karate, marks=[pytest.mark.small, pytest.mark.undirected]),
    # name: netscience, nodes: 1461, edges: 5484
    pytest.param(datasets.netscience, marks=[pytest.mark.small, pytest.mark.directed]),
    # name: email-Eu-core, nodes: 1005, edges: 25571
    pytest.param(
        datasets.email_Eu_core, marks=[pytest.mark.small, pytest.mark.directed]
    ),
    # name: amazon0302, nodes: 262111, edges: 1234877
    pytest.param(datasets.amazon0302, marks=[pytest.mark.medium, pytest.mark.directed]),
    # name: cit-Patents, nodes: 3774768, edges: 16518948
    pytest.param(
        datasets.cit_patents, marks=[pytest.mark.medium, pytest.mark.directed]
    ),
    # name: hollywood, nodes: 1139905, edges: 57515616
    pytest.param(
        datasets.hollywood, marks=[pytest.mark.medium, pytest.mark.undirected]
    ),
    # name: soc-LiveJournal1, nodes: 4847571, edges: 68993773
    pytest.param(
        datasets.soc_livejournal, marks=[pytest.mark.medium, pytest.mark.directed]
    ),
    # name: europe_osm, nodes: 50912018, edges: 54054660
    pytest.param(
        datasets.europe_osm, marks=[pytest.mark.large, pytest.mark.undirected]
    ),
]

backend_param_values = ["cugraph", "cugraph-preconverted", None]


def setup_module(module):
    """
    Trivial conversion call to force various one-time CUDA initialization
    operations to happen outside of benchmarks.
    """
    G = nx.karate_club_graph()
    nxcg.from_networkx(G)


# Test IDs are generated using the lambda assigned to the ids arg to provide an
# easier-to-read name. This is especially helpful for Dataset objs (see
# https://docs.pytest.org/en/stable/reference/reference.html#pytest-fixture)
@pytest.fixture(
    scope="module", params=dataset_param_values, ids=lambda ds: f"ds={ds!s}"
)
def graph_obj(request):
    """
    Returns a NX Graph or DiGraph obj from the dataset instance parameter.
    """
    dataset = request.param
    return nx_graph_from_dataset(dataset)


@pytest.fixture(
    scope="module",
    params=backend_param_values,
    ids=lambda backend: f"backend={backend}",
)
def backend(request):
    """
    Returns the backend name to use. This is done as a fixture for consistency
    and simplicity when creating benchmarks (no need to mark the benchmark as
    parametrized).
    """
    return request.param


################################################################################
# Helpers
def nx_graph_from_dataset(dataset_obj):
    """
    Read the dataset specified by the dataset_obj and create and return a
    nx.Graph or nx.DiGraph instance based on the dataset is_directed metadata.
    """
    create_using = nx.DiGraph if dataset_obj.metadata["is_directed"] else nx.Graph
    names = dataset_obj.metadata["col_names"]
    dtypes = dataset_obj.metadata["col_types"]
    if isinstance(dataset_obj.metadata["header"], int):
        header = dataset_obj.metadata["header"]
    else:
        header = None

    pandas_edgelist = pd.read_csv(
        dataset_obj.get_path(),
        delimiter=dataset_obj.metadata["delim"],
        names=names,
        dtype=dict(zip(names, dtypes)),
        header=header,
    )
    G = nx.from_pandas_edgelist(
        pandas_edgelist, source=names[0], target=names[1], create_using=create_using
    )
    return G


def get_backend_wrapper(backend_name):
    """
    Returns a callable that wraps an algo function in order to set the
    "backend" kwarg on it.
    """

    # FIXME: consider a pytest param to run force_unlazy_eval=False for
    # benchmarks that set it to True in order to measure the time spent for any
    # deferred computation/conversions.
    def wrap_callable_for_dispatch(func, force_unlazy_eval=False):
        # force_unlazy_eval=True forces iterators and other containers to
        # generate a complete set of results in order to include any deferred
        # compute or conversion in the benchmark
        def wrapper(*args, **kwargs):
            kwargs["backend"] = backend_name
            retval = func(*args, **kwargs)
            if force_unlazy_eval:
                if isinstance(retval, Mapping):
                    retval = dict(retval)
                else:
                    retval = list(retval)
            return retval

        return wrapper

    return wrap_callable_for_dispatch


@pytest.fixture(
    scope="module",
    params=backend_param_values,
    ids=lambda backend: f"backend={backend}",
)
def backend_wrapper(request):
    """
    Returns a callable that takes a function algo and wraps it in another
    function that calls the algo using the appropriate backend.

    For example: if the backend to test is "cugraph", this will return a
    function that calls nx.pagerank(..., backend='cugraph')
    """
    backend_name = request.param
    actual_backend_name = backend_name

    # Special case: cugraph-preconverted may be specified as a backend but this
    # name is reserved to indicate a cugraph backend is to be used with a
    # preconverted graph obj (rather than having the backend do the
    # conversion).
    if backend_name == "cugraph-preconverted":
        actual_backend_name = "cugraph"

    wrapper = get_backend_wrapper(actual_backend_name)

    wrapper.backend_name = backend_name
    return wrapper


def get_graph_obj_for_benchmark(graph_obj, backend_wrapper):
    """
    Given a Graph object and a backend name, return a converted Graph or the
    original Graph object based on the backend to use.

    This is needed because some backend names are actually used as descriptions
    for combinations of backends and converted/non-converted graphs.  For
    example, a benchmark may specify the "cugraph-preconverted" backend, which
    is not an installed backend but instead refers to the "cugraph" backend
    passed a NX Graph that has been converted to a nx-cugraph Graph object.
    """
    G = graph_obj
    if backend_wrapper.backend_name == "cugraph-preconverted":
        G = nxcg.from_networkx(
            G,
            preserve_all_attrs=True,
            use_compat_graph=True,
        )
    return G


def get_highest_degree_node(graph_obj):
    degrees = graph_obj.degree()  # list of tuples of (node, degree)
    return max(degrees, key=lambda t: t[1])[0]


def build_personalization_dict(pagerank_dict):
    """
    Returns a dictionary that can be used as the personalization value for a
    call to nx.pagerank(). The pagerank_dict passed in is used as the initial
    source of values for each node, and this function simply treats the list of
    dict values as two halves (halves A and B) and swaps them so (most if not
    all) nodes/keys are assigned a different value from the dictionary.
    """
    num_half = len(pagerank_dict) // 2
    A_half_items = list(pagerank_dict.items())[:num_half]
    B_half_items = list(pagerank_dict.items())[num_half:]

    # Support an odd number of items by initializing with B_half_items, which
    # will always be one bigger if the number of items is odd. This will leave
    # the one remainder (in the case of an odd number) unchanged.
    pers_dict = dict(B_half_items)
    pers_dict.update({A_half_items[i][0]: B_half_items[i][1] for i in range(num_half)})
    pers_dict.update({B_half_items[i][0]: A_half_items[i][1] for i in range(num_half)})

    return pers_dict


# Used to return a function that calls the original function inside a try-except block
# which is useful because it allows us to save pytest-benchmark numbers if failure is
# the correct behavior for certain graphs
def possible_to_fail(exception, function):
    def nested_func(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except exception:
            print(f"{function.__name__} raised {exception}")

    return nested_func


################################################################################
# Benchmarks
def bench_from_networkx(benchmark, graph_obj):
    benchmark(nxcg.from_networkx, graph_obj)


# normalized_param_values = [True, False]
normalized_param_values = [True]
k_param_values = [10, 100, 1000]


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
@pytest.mark.parametrize("k", k_param_values, ids=lambda k: f"{k=}")
def bench_betweenness_centrality(benchmark, graph_obj, backend_wrapper, normalized, k):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)

    if k > G.number_of_nodes():
        pytest.skip(reason=f"{k=} > {G.number_of_nodes()=}")

    result = benchmark.pedantic(
        target=backend_wrapper(nx.betweenness_centrality),
        args=(G,),
        kwargs=dict(
            weight=None,
            normalized=normalized,
            k=k,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
@pytest.mark.parametrize("k", k_param_values, ids=lambda k: f"{k=}")
def bench_edge_betweenness_centrality(
    benchmark, graph_obj, backend_wrapper, normalized, k
):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)

    if k > G.number_of_nodes():
        pytest.skip(reason=f"{k=} > {G.number_of_nodes()=}")

    result = benchmark.pedantic(
        target=backend_wrapper(nx.edge_betweenness_centrality),
        args=(G,),
        kwargs=dict(
            weight=None,
            normalized=normalized,
            k=k,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_louvain_communities(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.community.louvain_communities),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


@pytest.mark.skipif("not hasattr(nx.community, 'leiden_communities')")
def bench_leiden_communities(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported
    if G.is_directed():
        G = G.to_undirected()
    if G.__networkx_backend__ not in nx.community.leiden_communities.backends:
        pytest.skip(
            reason=f"leiden_communities not implemented by {G.__networkx_backend__!r}"
        )
        return
    result = benchmark.pedantic(
        target=backend_wrapper(nx.community.leiden_communities),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_degree_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.degree_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_eigenvector_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.eigenvector_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
def bench_hits(benchmark, graph_obj, backend_wrapper, normalized):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.hits),
        args=(G,),
        kwargs=dict(
            normalized=normalized,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is tuple
    assert len(result) == 2
    assert type(result[0]) is dict
    assert type(result[1]) is dict


def bench_in_degree_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.in_degree_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


@pytest.mark.parametrize(
    "normalized", normalized_param_values, ids=lambda norm: f"{norm=}"
)
def bench_katz_centrality(benchmark, graph_obj, backend_wrapper, normalized):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        # calling katz_centrality this way because the algorithm may fail to
        # converge for some graphs, which is expected
        target=possible_to_fail(
            nx.PowerIterationFailedConvergence, backend_wrapper(nx.katz_centrality)
        ),
        args=(G,),
        kwargs=dict(
            normalized=normalized,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict or result is None


def bench_k_truss(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.k_truss),
        args=(G,),
        kwargs=dict(
            k=2,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    # Check that this at least appears to be some kind of NX-like Graph
    assert hasattr(result, "has_node")


def bench_out_degree_centrality(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.out_degree_centrality),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_pagerank(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.pagerank),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_pagerank_personalized(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)

    # FIXME: This will run for every combination of inputs, even if the
    # graph/dataset does not change. Ideally this is run once per
    # graph/dataset.
    pagerank_dict = nx.pagerank(G)
    personalization_dict = build_personalization_dict(pagerank_dict)

    result = benchmark.pedantic(
        target=backend_wrapper(nx.pagerank),
        args=(G,),
        kwargs={"personalization": personalization_dict},
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_shortest_path(benchmark, graph_obj, backend_wrapper):
    """
    This passes in the source node with the highest degree, but no target.
    """
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)

    result = benchmark.pedantic(
        target=backend_wrapper(nx.shortest_path, force_unlazy_eval=True),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert isinstance(result, Mapping)  # dict in nx, but we duck-type


def bench_single_source_shortest_path_length(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)

    result = benchmark.pedantic(
        target=backend_wrapper(nx.single_source_shortest_path_length),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_single_target_shortest_path_length(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(
            nx.single_target_shortest_path_length, force_unlazy_eval=True
        ),
        args=(G,),
        kwargs=dict(
            target=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    # force_unlazy_eval=True forces iterators and other containers to generate
    # a complete set of results (in order to include any deferred compute or
    # conversion in the benchmark), but is not needed for this algo in NX 3.3+
    # since it returns a dict instead of an iterator. Forcing eval does not
    # change the benchmark timing.
    assert type(result) is list


def bench_ancestors(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.ancestors),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is set


def bench_average_clustering(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported by nx-cugraph
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.average_clustering),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is float


def bench_generic_bfs_edges(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.generic_bfs_edges, force_unlazy_eval=True),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_bfs_edges(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.bfs_edges, force_unlazy_eval=True),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_bfs_layers(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.bfs_layers, force_unlazy_eval=True),
        args=(G,),
        kwargs=dict(
            sources=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_bfs_predecessors(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.bfs_predecessors, force_unlazy_eval=True),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_bfs_successors(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.bfs_successors, force_unlazy_eval=True),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_bfs_tree(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.bfs_tree),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    # Check that this at least appears to be some kind of NX-like Graph
    assert hasattr(result, "has_node")


def bench_clustering(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported by nx-cugraph
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.clustering),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_core_number(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported by nx-cugraph
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.core_number),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_descendants(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.descendants),
        args=(G,),
        kwargs=dict(
            source=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is set


def bench_descendants_at_distance(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.descendants_at_distance),
        args=(G,),
        kwargs=dict(
            source=node,
            distance=1,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is set


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_bipartite(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.is_bipartite),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is bool


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_strongly_connected(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.is_strongly_connected),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is bool


def bench_is_weakly_connected(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.is_weakly_connected),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is bool


@pytest.mark.skip(reason="benchmark not implemented")
def bench_number_strongly_connected_components(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.number_strongly_connected_components),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is int


def bench_number_weakly_connected_components(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.number_weakly_connected_components),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is int


def bench_overall_reciprocity(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.overall_reciprocity),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is float


def bench_reciprocity(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.reciprocity),
        args=(G,),
        kwargs=dict(
            nodes=node,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is float


@pytest.mark.skip(reason="benchmark not implemented")
def bench_strongly_connected_components(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(
            nx.strongly_connected_components, force_unlazy_eval=True
        ),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_transitivity(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported by nx-cugraph
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.transitivity),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is float


def bench_triangles(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    # DiGraphs are not supported
    if G.is_directed():
        G = G.to_undirected()
    result = benchmark.pedantic(
        target=backend_wrapper(nx.triangles),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_weakly_connected_components(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.weakly_connected_components, force_unlazy_eval=True),
        args=(G,),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


def bench_ego_graph(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    node = get_highest_degree_node(graph_obj)
    result = benchmark.pedantic(
        target=backend_wrapper(nx.ego_graph),
        args=(G,),
        kwargs=dict(
            n=node,
            radius=100,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is type(G)


def bench_bipartite_BC_n1000_m3000_k100000(benchmark, backend_wrapper):
    # Example how to run:
    # $ pytest -sv -k "bench_bipartite_BC" \
    #   --benchmark-json="logs/None__bipartite_BC__None.json" \
    #   bench_algos.py
    n = 1000
    m = 3000
    k = 100000
    graph_obj = nx.bipartite.generators.gnmk_random_graph(n, m, k)
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)
    nodes = list(range(n))
    result = benchmark.pedantic(
        target=backend_wrapper(nx.bipartite.betweenness_centrality),
        args=(G, nodes),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is dict


def bench_jaccard(benchmark, graph_obj, backend_wrapper):
    G = get_graph_obj_for_benchmark(graph_obj, backend_wrapper)

    # ebunch is a list of node pairs to limit the jaccard run.
    nodes = list(G.nodes)
    start = nodes[0]
    ebunch = [(start, n) for n in nodes[1:]]
    start = nodes[1]
    ebunch += [(start, n) for n in nodes[2:]]
    start = nodes[2]
    ebunch += [(start, n) for n in nodes[3:]]

    # DiGraphs are not supported
    if G.is_directed():
        G = G.to_undirected()

    result = benchmark.pedantic(
        target=backend_wrapper(nx.jaccard_coefficient, force_unlazy_eval=True),
        args=(G,),
        kwargs=dict(
            ebunch=ebunch,
        ),
        rounds=rounds,
        iterations=iterations,
        warmup_rounds=warmup_rounds,
    )
    assert type(result) is list


@pytest.mark.skip(reason="benchmark not implemented")
def bench_complete_bipartite_graph(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_connected_components(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_connected(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_node_connected_component(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_number_connected_components(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_isolate(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_isolates(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_number_of_isolates(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_complement(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_reverse(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_arborescence(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_branching(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_forest(benchmark, graph_obj, backend_wrapper):
    pass


@pytest.mark.skip(reason="benchmark not implemented")
def bench_is_tree(benchmark, graph_obj, backend_wrapper):
    pass
