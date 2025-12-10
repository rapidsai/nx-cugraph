# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re

import networkx as nx

import nx_cugraph as nxcg
from nx_cugraph import _nxver


class BackendInterface:
    # Required conversions
    @staticmethod
    def convert_from_nx(graph, *args, **kwargs):
        return nxcg.from_networkx(
            graph,
            *args,
            use_compat_graph=_nxver < (3, 3)
            or nx.config.backends.cugraph.use_compat_graphs,
            **kwargs,
        )

    @staticmethod
    def convert_to_nx(obj, *, name: str | None = None):
        if isinstance(obj, nxcg.Graph | nxcg.CudaGraph):
            return nxcg.to_networkx(obj)
        return obj

    @staticmethod
    def on_start_tests(items):
        """Modify pytest items after tests have been collected.

        This is called during ``pytest_collection_modifyitems`` phase of pytest.
        We use this to set `xfail` on tests we expect to fail. See:

        https://docs.pytest.org/en/stable/reference/reference.html#std-hook-pytest_collection_modifyitems
        """
        try:
            import pytest
        except ModuleNotFoundError:
            return

        def key(testpath):
            # Allow testpath to be either a regex (useful for matching parameterized
            # tests) or a formatted string to match specific tests.
            if isinstance(testpath, re.Pattern):
                return testpath
            filename, path = testpath.split(":")
            *names, testname = path.split(".")
            if names:
                [classname] = names
                return (testname, frozenset({classname, filename}))
            return (testname, frozenset({filename}))

        use_compat_graph = (
            _nxver < (3, 3) or nx.config.backends.cugraph.use_compat_graphs
        )
        fallback = use_compat_graph or nx.utils.backends._dispatchable._fallback_to_nx

        # Reasons for xfailing
        # For nx version <= 3.2
        nx_cugraph_in_test_setup = (
            "nx-cugraph Graph is incompatible in test setup in nx versions < 3.3"
        )
        different_iteration_order = "Different graph data iteration order"
        # For nx version <= 3.4
        bc_normalization_fixed = "BC normalization fixed in 3.5"
        # For nx version >= 3.6
        bc_results_different = (
            "Betweenness centrality results computed differently in 3.6"
        )
        # For nx version == 3.6.1
        undirected_edges_source_target_order_differs = (
            "Source/target edge order may differ for undirected graph "
            "and fail overly-strict comparison"
        )
        # For all versions
        louvain_different = "Louvain may be different due to RNG"
        sssp_path_different = "sssp may choose a different valid path"
        tuple_elements_preferred = "elements are tuples instead of lists"
        no_mixed_dtypes_for_nodes = (
            # This one is tricky b/c we don't raise; all dtypes are treated as str
            "mixed dtypes (str, int, float) for single node property not supported"
        )
        # These shouldn't fail if using Graph or falling back to networkx
        no_string_dtype = "string edge values not currently supported"
        no_object_dtype_for_edges = (
            "Edges don't support object dtype (lists, strings, etc.)"
        )
        xfail = {
            # This is removed while strongly_connected_components() is not
            # dispatchable. See algorithms/components/strongly_connected.py for
            # details.
            #
            # key(
            #     "test_strongly_connected.py:"
            #     "TestStronglyConnected.test_condensation_mapping_and_members"
            # ): "Strongly connected groups in different iteration order",
            key(
                "test_cycles.py:TestMinimumCycleBasis.test_unweighted_diamond"
            ): sssp_path_different,
            key(
                "test_cycles.py:TestMinimumCycleBasis.test_weighted_diamond"
            ): sssp_path_different,
            key(
                "test_cycles.py:TestMinimumCycleBasis.test_petersen_graph"
            ): sssp_path_different,
            key(
                "test_cycles.py:TestMinimumCycleBasis."
                "test_gh6787_and_edge_attribute_names"
            ): sssp_path_different,
            key(
                "test_relabel.py:"
                "test_relabel_preserve_node_order_partial_mapping_with_copy_false"
            ): "Node order is preserved when relabeling with partial mapping",
            key(
                "test_gml.py:"
                "TestPropertyLists.test_reading_graph_with_single_element_list_property"
            ): tuple_elements_preferred,
        }
        if not fallback:
            xfail.update(
                {
                    key(
                        "test_graph_hashing.py:test_isomorphic_edge_attr"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_graph_hashing.py:test_isomorphic_edge_attr_and_node_attr"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_graph_hashing.py:test_isomorphic_edge_attr_subgraph_hash"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_graph_hashing.py:"
                        "test_isomorphic_edge_attr_and_node_attr_subgraph_hash"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_summarization.py:TestSNAPNoEdgeTypes.test_summary_graph"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_summarization.py:TestSNAPUndirected.test_summary_graph"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_summarization.py:TestSNAPDirected.test_summary_graph"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_gexf.py:TestGEXF.test_relabel"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_gml.py:TestGraph.test_parse_gml_cytoscape_bug"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_gml.py:TestGraph.test_parse_gml"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_gml.py:TestGraph.test_read_gml"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_gml.py:TestGraph.test_data_types"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_gml.py:"
                        "TestPropertyLists.test_reading_graph_with_list_property"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_relabel.py:"
                        "TestRelabel.test_relabel_multidigraph_inout_merge_nodes"
                    ): no_string_dtype,
                    key(
                        "test_relabel.py:"
                        "TestRelabel.test_relabel_multigraph_merge_inplace"
                    ): no_string_dtype,
                    key(
                        "test_relabel.py:"
                        "TestRelabel.test_relabel_multidigraph_merge_inplace"
                    ): no_string_dtype,
                    key(
                        "test_relabel.py:"
                        "TestRelabel.test_relabel_multidigraph_inout_copy"
                    ): no_string_dtype,
                    key(
                        "test_relabel.py:TestRelabel.test_relabel_multigraph_merge_copy"
                    ): no_string_dtype,
                    key(
                        "test_relabel.py:"
                        "TestRelabel.test_relabel_multidigraph_merge_copy"
                    ): no_string_dtype,
                    key(
                        "test_relabel.py:"
                        "TestRelabel.test_relabel_multigraph_nonnumeric_key"
                    ): no_string_dtype,
                    key(
                        "test_contraction.py:test_multigraph_path"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_contraction.py:test_directed_multigraph_path"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_contraction.py:test_multigraph_blockmodel"
                    ): no_object_dtype_for_edges,
                    key(
                        "test_summarization.py:"
                        "TestSNAPUndirectedMulti.test_summary_graph"
                    ): no_string_dtype,
                    key(
                        "test_summarization.py:TestSNAPDirectedMulti.test_summary_graph"
                    ): no_string_dtype,
                }
            )
        else:
            xfail.update(
                {
                    key(
                        "test_gml.py:"
                        "TestPropertyLists.test_reading_graph_with_list_property"
                    ): no_mixed_dtypes_for_nodes,
                }
            )

        if _nxver < (3, 3):
            xfail.update(
                {
                    # NetworkX versions prior to 3.2.1 have tests written to
                    # expect sp.sparse.linalg.ArpackNoConvergence exceptions
                    # raised on no convergence in HITS. Newer versions since
                    # the merge of
                    # https://github.com/networkx/networkx/pull/7084 expect
                    # nx.PowerIterationFailedConvergence, which is what
                    # nx_cugraph.hits raises, so we mark them as xfail for
                    # previous versions of NX.
                    key(
                        "test_hits.py:TestHITS.test_hits_not_convergent"
                    ): "nx_cugraph.hits raises updated exceptions not caught in "
                    "these tests",
                    # NetworkX versions 3.2 and older contain tests that fail
                    # with pytest>=8. Assume pytest>=8 and mark xfail.
                    key(
                        "test_strongly_connected.py:"
                        "TestStronglyConnected.test_connected_raise"
                    ): "test is incompatible with pytest>=8",
                    # NetworkX 3.3 introduced logic around functions that return graphs
                    key(
                        "test_vf2pp_helpers.py:TestGraphTinoutUpdating.test_updating"
                    ): nx_cugraph_in_test_setup,
                    key(
                        "test_vf2pp_helpers.py:TestGraphTinoutUpdating.test_restoring"
                    ): nx_cugraph_in_test_setup,
                    key(
                        "test_vf2pp_helpers.py:TestDiGraphTinoutUpdating.test_updating"
                    ): nx_cugraph_in_test_setup,
                    key(
                        "test_vf2pp_helpers.py:TestDiGraphTinoutUpdating.test_restoring"
                    ): nx_cugraph_in_test_setup,
                    # Different iteration
                    key(
                        "test_cycles.py:TestMinimumCycleBasis."
                        "test_gh6787_and_edge_attribute_names"
                    ): different_iteration_order,
                    key(
                        "test_euler.py:TestEulerianCircuit."
                        "test_eulerian_circuit_cycle"
                    ): different_iteration_order,
                    key(
                        "test_gml.py:TestGraph.test_special_float_label"
                    ): different_iteration_order,
                }
            )
        else:
            xfail.update(
                {
                    key("test_louvain.py:test_max_level"): louvain_different,
                }
            )
        if _nxver < (3, 5):
            xfail.update(
                {
                    key(
                        "test_betweenness_centrality.py:"
                        "TestBetweennessCentrality.test_sample_from_P3"
                    ): bc_normalization_fixed,
                }
            )
        if _nxver >= (3, 6):
            xfail.update(
                {
                    key(
                        "test_betweenness_centrality.py:"
                        "TestEdgeBetweennessCentrality.test_edge_betweenness_k"
                    ): bc_results_different,
                    key(
                        "test_betweenness_centrality_subset.py:"
                        "test_equivalence_non_subset"
                    ): bc_results_different,
                }
            )
        xfail.update(
            {
                key("test_louvain.py:test_karate_club_partition"): louvain_different,
                key("test_louvain.py:test_none_weight_param"): louvain_different,
                key("test_louvain.py:test_multigraph"): louvain_different,
                # See networkx#6630
                key(
                    "test_louvain.py:test_undirected_selfloops"
                ): "self-loops not handled in Louvain",
            }
        )
        if not fallback:
            xfail.update(
                {
                    key(
                        "test_convert_pandas.py:TestConvertPandas."
                        "test_from_edgelist_multi_attr_incl_target"
                    ): no_string_dtype,
                    key(
                        "test_convert_pandas.py:TestConvertPandas."
                        "test_from_edgelist_multidigraph_and_edge_attr"
                    ): no_string_dtype,
                    key(
                        "test_convert_pandas.py:TestConvertPandas."
                        "test_from_edgelist_int_attr_name"
                    ): no_string_dtype,
                }
            )

        if _nxver == (3, 4, 2):
            xfail[key("test_pylab.py:test_return_types")] = "Ephemeral NetworkX bug"

        too_slow = "Too slow to run"
        fails_strict_compare_with_nan_inf = "Fails strict compare with nan/inf"
        undirected_graph_edges_out_of_order = (
            "Undirected graph edges may be out of order"
        )
        skip = {
            key("test_tree_isomorphism.py:test_positive"): too_slow,
            key("test_tree_isomorphism.py:test_negative"): too_slow,
            # These repeatedly call `bfs_layers`, which converts the graph every call
            key(
                "test_vf2pp.py:TestGraphISOVF2pp.test_custom_graph2_different_labels"
            ): too_slow,
            key(
                "test_vf2pp.py:TestGraphISOVF2pp.test_custom_graph3_same_labels"
            ): too_slow,
            key(
                "test_vf2pp.py:TestGraphISOVF2pp.test_custom_graph3_different_labels"
            ): too_slow,
            key(
                "test_vf2pp.py:TestGraphISOVF2pp.test_custom_graph4_same_labels"
            ): too_slow,
            key(
                "test_vf2pp.py:TestGraphISOVF2pp."
                "test_disconnected_graph_all_same_labels"
            ): too_slow,
            key(
                "test_vf2pp.py:TestGraphISOVF2pp."
                "test_disconnected_graph_all_different_labels"
            ): too_slow,
            key(
                "test_vf2pp.py:TestGraphISOVF2pp."
                "test_disconnected_graph_some_same_labels"
            ): too_slow,
            key(
                "test_vf2pp.py:TestMultiGraphISOVF2pp."
                "test_custom_multigraph3_same_labels"
            ): too_slow,
            key(
                "test_vf2pp_helpers.py:TestNodeOrdering."
                "test_matching_order_all_branches"
            ): too_slow,
            key(
                "test_gml.py:TestGraph.test_special_float_label"
            ): fails_strict_compare_with_nan_inf,
            key(
                "test_relabel.py:TestRelabel.test_relabel_toposort"
            ): undirected_graph_edges_out_of_order,
        }
        # These were added in NetworkX 3.5
        for i in range(8, 12):
            skip[
                key(
                    "test_tree_isomorphism.py:"
                    f"test_tree_isomorphism_all_non_isomorphic_pairs[{i}]"
                )
            ] = too_slow
        for i in range(9, 15):
            skip[
                key(
                    "test_tree_isomorphism.py:"
                    f"test_tree_isomorphic_all_non_isomorphic_trees_relabeled[{i}]"
                )
            ] = too_slow

        # Needed for new tests added in networkx 3.6.1, should not be needed in later
        # versions.
        if _nxver == (3, 6, 1):
            skip.update(
                {
                    key(
                        re.compile(
                            r"test_matrix\.py:TestBiadjacencyMatrix\.test_from_biadjacency_nodelist\[.*\]"  # noqa: E501
                        )
                    ): undirected_edges_source_target_order_differs,
                }
            )

        if os.environ.get("PYTEST_NO_SKIP", ""):
            skip.clear()

        for item in items:
            kset = set(item.keywords)
            item_test_path = f"{item.path.name}:{item.getmodpath()}"
            # Xfail tests
            for test_name, reason in xfail.items():
                if isinstance(test_name, re.Pattern):
                    if test_name.match(item_test_path):
                        item.add_marker(pytest.mark.xfail(reason=reason))
                else:
                    (test_name, keywords) = test_name
                    if item.name == test_name and keywords.issubset(kset):
                        item.add_marker(pytest.mark.xfail(reason=reason))
            # Skip tests
            for test_name, reason in skip.items():
                if isinstance(test_name, re.Pattern):
                    if test_name.match(item_test_path):
                        item.add_marker(pytest.mark.skip(reason=reason))
                else:
                    (test_name, keywords) = test_name
                    if item.name == test_name and keywords.issubset(kset):
                        item.add_marker(pytest.mark.skip(reason=reason))

    @classmethod
    def can_run(cls, name, args, kwargs):
        """Can this backend run the specified algorithms with the given arguments?"""
        return getattr(cls, name).can_run(*args, **kwargs)

    @classmethod
    def should_run(cls, name, args, kwargs):
        """Should this backend run the specified algorithms with the given arguments?"""
        return getattr(cls, name).should_run(*args, **kwargs)
