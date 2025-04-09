# nx-cugraph 25.04.00 (9 Apr 2025)

## 🚨 Breaking Changes

- Add ForceAtlas2 Support ([#99](https://github.com/rapidsai/nx-cugraph/pull/99)) [@nv-rliu](https://github.com/nv-rliu)

## 🐛 Bug Fixes

- Fix `(Multi)DiGraph.__networkx_cache__` initialization ([#91](https://github.com/rapidsai/nx-cugraph/pull/91)) [@eriknw](https://github.com/eriknw)

## 📖 Documentation

- [DOC] Add a Contributing Guide ([#102](https://github.com/rapidsai/nx-cugraph/pull/102)) [@nv-rliu](https://github.com/nv-rliu)
- Update README ([#100](https://github.com/rapidsai/nx-cugraph/pull/100)) [@nv-rliu](https://github.com/nv-rliu)

## 🚀 New Features

- Add ForceAtlas2 Support ([#99](https://github.com/rapidsai/nx-cugraph/pull/99)) [@nv-rliu](https://github.com/nv-rliu)

## 🛠️ Improvements

- Update docs for BC ([#107](https://github.com/rapidsai/nx-cugraph/pull/107)) [@eriknw](https://github.com/eriknw)
- Add Benchmarks for ForceAtlas2 ([#97](https://github.com/rapidsai/nx-cugraph/pull/97)) [@nv-rliu](https://github.com/nv-rliu)
- Use conda-build instead of conda-mambabuild ([#95](https://github.com/rapidsai/nx-cugraph/pull/95)) [@bdice](https://github.com/bdice)
- add cugraph-notebook-codeowners to CODEOWNERS ([#94](https://github.com/rapidsai/nx-cugraph/pull/94)) [@AyodeAwe](https://github.com/AyodeAwe)
- remove docs dependencies ([#93](https://github.com/rapidsai/nx-cugraph/pull/93)) [@jameslamb](https://github.com/jameslamb)
- Create Conda CI test env in one step ([#90](https://github.com/rapidsai/nx-cugraph/pull/90)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Update pre-commit hooks ([#88](https://github.com/rapidsai/nx-cugraph/pull/88)) [@eriknw](https://github.com/eriknw)
- Use shared-workflows branch-25.04 ([#85](https://github.com/rapidsai/nx-cugraph/pull/85)) [@bdice](https://github.com/bdice)
- Add `shellcheck` to pre-commit and fix warnings ([#84](https://github.com/rapidsai/nx-cugraph/pull/84)) [@gforsyth](https://github.com/gforsyth)
- Add build_type input field for `test.yaml` ([#82](https://github.com/rapidsai/nx-cugraph/pull/82)) [@gforsyth](https://github.com/gforsyth)
- Forward-merge branch-25.02 to branch-25.04 ([#76](https://github.com/rapidsai/nx-cugraph/pull/76)) [@bdice](https://github.com/bdice)
- include all pyproject dependencies in &#39;requirements&#39; output ([#75](https://github.com/rapidsai/nx-cugraph/pull/75)) [@jameslamb](https://github.com/jameslamb)
- Migrate to NVKS for amd64 CI runners ([#72](https://github.com/rapidsai/nx-cugraph/pull/72)) [@bdice](https://github.com/bdice)

# nx-cugraph 25.02.00 (13 Feb 2025)

## 🐛 Bug Fixes

- Update BC to use random nodes to match nx given `k` ([#59](https://github.com/rapidsai/nx-cugraph/pull/59)) [@eriknw](https://github.com/eriknw)
- Updates to Benchmark Script ([#51](https://github.com/rapidsai/nx-cugraph/pull/51)) [@nv-rliu](https://github.com/nv-rliu)

## 📖 Documentation

- Adds notebook for Jaccard Similarity ([#71](https://github.com/rapidsai/nx-cugraph/pull/71)) [@rlratzel](https://github.com/rlratzel)
- Fix links in README ([#54](https://github.com/rapidsai/nx-cugraph/pull/54)) [@eriknw](https://github.com/eriknw)

## 🛠️ Improvements

- Use `rapids-pip-retry` in CI jobs that might need retries ([#80](https://github.com/rapidsai/nx-cugraph/pull/80)) [@gforsyth](https://github.com/gforsyth)
- Revert CUDA 12.8 shared workflow branch changes ([#74](https://github.com/rapidsai/nx-cugraph/pull/74)) [@vyasr](https://github.com/vyasr)
- Add benchmark for `leiden_communities` ([#67](https://github.com/rapidsai/nx-cugraph/pull/67)) [@eriknw](https://github.com/eriknw)
- Build and test with CUDA 12.8.0 ([#66](https://github.com/rapidsai/nx-cugraph/pull/66)) [@bdice](https://github.com/bdice)
- update pip devcontainers to UCX 1.18, other small packaging changes ([#65](https://github.com/rapidsai/nx-cugraph/pull/65)) [@jameslamb](https://github.com/jameslamb)
- Adds support for `jaccard_coefficient` ([#62](https://github.com/rapidsai/nx-cugraph/pull/62)) [@rlratzel](https://github.com/rlratzel)
- Add `amazon0302` Dataset to nx-cugraph Bench Algos ([#61](https://github.com/rapidsai/nx-cugraph/pull/61)) [@nv-rliu](https://github.com/nv-rliu)
- Remove sphinx pinning ([#60](https://github.com/rapidsai/nx-cugraph/pull/60)) [@vyasr](https://github.com/vyasr)
- Add `can_run` that checks `create_using` argument ([#56](https://github.com/rapidsai/nx-cugraph/pull/56)) [@eriknw](https://github.com/eriknw)
- prefer system install of UCX in devcontainers, update outdated RAPIDS references ([#53](https://github.com/rapidsai/nx-cugraph/pull/53)) [@jameslamb](https://github.com/jameslamb)
- Require approval to run CI on draft PRs ([#52](https://github.com/rapidsai/nx-cugraph/pull/52)) [@bdice](https://github.com/bdice)
- Add experimental version of `leiden_communities` ([#50](https://github.com/rapidsai/nx-cugraph/pull/50)) [@eriknw](https://github.com/eriknw)
- Add `lowest_common_ancestor` algorithm ([#35](https://github.com/rapidsai/nx-cugraph/pull/35)) [@eriknw](https://github.com/eriknw)

# nx-cugraph 24.12.00 (11 Dec 2024)

## 🚨 Breaking Changes

- Add `nx-cugraph` Package Publishing ([#16](https://github.com/rapidsai/nx-cugraph/pull/16)) [@nv-rliu](https://github.com/nv-rliu)
- Merge fast-forwarded files from cugraph into nx-cugraph ([#13](https://github.com/rapidsai/nx-cugraph/pull/13)) [@nv-rliu](https://github.com/nv-rliu)
- Update `.pre-commit-config.yaml` and Implement Suggestions ([#12](https://github.com/rapidsai/nx-cugraph/pull/12)) [@nv-rliu](https://github.com/nv-rliu)
- [CI] Adding CI Workflows: checks, changed-files, builds ([#6](https://github.com/rapidsai/nx-cugraph/pull/6)) [@nv-rliu](https://github.com/nv-rliu)
- Setting Up New Repo, Adding Files, etc. ([#5](https://github.com/rapidsai/nx-cugraph/pull/5)) [@nv-rliu](https://github.com/nv-rliu)

## 🐛 Bug Fixes

- Add sphinx-lint pre-commit (and some docs fixes) ([#29](https://github.com/rapidsai/nx-cugraph/pull/29)) [@eriknw](https://github.com/eriknw)
- Update and test `_nx_cugraph._check_networkx_version` ([#24](https://github.com/rapidsai/nx-cugraph/pull/24)) [@eriknw](https://github.com/eriknw)
- Remove automatic &quot;Python&quot; labeler ([#23](https://github.com/rapidsai/nx-cugraph/pull/23)) [@eriknw](https://github.com/eriknw)

## 📖 Documentation

- Remove `docs/` directory ([#37](https://github.com/rapidsai/nx-cugraph/pull/37)) [@eriknw](https://github.com/eriknw)

## 🛠️ Improvements

- Small Updates to Benchmarks Directory ([#48](https://github.com/rapidsai/nx-cugraph/pull/48)) [@nv-rliu](https://github.com/nv-rliu)
- Includes all deferred conversion costs in benchmarks ([#34](https://github.com/rapidsai/nx-cugraph/pull/34)) [@rlratzel](https://github.com/rlratzel)
- Add Bipartite Betweenness Centrality ([#32](https://github.com/rapidsai/nx-cugraph/pull/32)) [@nv-rliu](https://github.com/nv-rliu)
- Change `degree_type` of `core_number` to `&quot;outgoing&quot;` ([#28](https://github.com/rapidsai/nx-cugraph/pull/28)) [@eriknw](https://github.com/eriknw)
- Drop support for NetworkX 3.0 and 3.1 ([#27](https://github.com/rapidsai/nx-cugraph/pull/27)) [@eriknw](https://github.com/eriknw)
- remove versioning workaround for nightlies ([#26](https://github.com/rapidsai/nx-cugraph/pull/26)) [@jameslamb](https://github.com/jameslamb)
- add devcontainers ([#25](https://github.com/rapidsai/nx-cugraph/pull/25)) [@jameslamb](https://github.com/jameslamb)
- Add pre-commit hook to disallow improper comparison to `_nxver` ([#22](https://github.com/rapidsai/nx-cugraph/pull/22)) [@eriknw](https://github.com/eriknw)
- Add notebooks/demo/accelerating_networkx.ipynb ([#21](https://github.com/rapidsai/nx-cugraph/pull/21)) [@eriknw](https://github.com/eriknw)
- enforce wheel size limits, README formatting in CI ([#19](https://github.com/rapidsai/nx-cugraph/pull/19)) [@jameslamb](https://github.com/jameslamb)
- Faster `shortest_path` ([#18](https://github.com/rapidsai/nx-cugraph/pull/18)) [@eriknw](https://github.com/eriknw)
- nx-cugraph: dispatch graph method to gpu or cpu ([#17](https://github.com/rapidsai/nx-cugraph/pull/17)) [@eriknw](https://github.com/eriknw)
- Add `nx-cugraph` Package Publishing ([#16](https://github.com/rapidsai/nx-cugraph/pull/16)) [@nv-rliu](https://github.com/nv-rliu)
- add CI workflows running tests ([#15](https://github.com/rapidsai/nx-cugraph/pull/15)) [@jameslamb](https://github.com/jameslamb)
- remove more cugraph-only details, other miscellaneous build/packaging changes ([#14](https://github.com/rapidsai/nx-cugraph/pull/14)) [@jameslamb](https://github.com/jameslamb)
- Merge fast-forwarded files from cugraph into nx-cugraph ([#13](https://github.com/rapidsai/nx-cugraph/pull/13)) [@nv-rliu](https://github.com/nv-rliu)
- Update `.pre-commit-config.yaml` and Implement Suggestions ([#12](https://github.com/rapidsai/nx-cugraph/pull/12)) [@nv-rliu](https://github.com/nv-rliu)
- Adding a `dependencies.yaml` file ([#9](https://github.com/rapidsai/nx-cugraph/pull/9)) [@nv-rliu](https://github.com/nv-rliu)
- [CI] Adding CI Workflows: checks, changed-files, builds ([#6](https://github.com/rapidsai/nx-cugraph/pull/6)) [@nv-rliu](https://github.com/nv-rliu)
- Setting Up New Repo, Adding Files, etc. ([#5](https://github.com/rapidsai/nx-cugraph/pull/5)) [@nv-rliu](https://github.com/nv-rliu)
