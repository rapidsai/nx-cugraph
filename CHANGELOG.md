# nx-cugraph 25.10.00 (8 Oct 2025)

## 🛠️ Improvements

- Update `forceatlas2_layout` to use updated `plc.force_atlas2` ([#191](https://github.com/rapidsai/nx-cugraph/pull/191)) [@eriknw](https://github.com/eriknw)
- Configure repo for automatic release notes generation ([#186](https://github.com/rapidsai/nx-cugraph/pull/186)) [@AyodeAwe](https://github.com/AyodeAwe)
- Adds Leiden notebook used in upcoming tech blog ([#185](https://github.com/rapidsai/nx-cugraph/pull/185)) [@rlratzel](https://github.com/rlratzel)
- Use branch-25.10 again ([#184](https://github.com/rapidsai/nx-cugraph/pull/184)) [@jameslamb](https://github.com/jameslamb)
- Update rapids-dependency-file-generator ([#183](https://github.com/rapidsai/nx-cugraph/pull/183)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Build and test with CUDA 13.0.0 ([#182](https://github.com/rapidsai/nx-cugraph/pull/182)) [@jameslamb](https://github.com/jameslamb)
- Use build cluster in devcontainers ([#181](https://github.com/rapidsai/nx-cugraph/pull/181)) [@trxcllnt](https://github.com/trxcllnt)
- Update rapids-build-backend to 0.4.1 ([#178](https://github.com/rapidsai/nx-cugraph/pull/178)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- ci(labeler): update labeler action to [@v5 ([#175](https://github.com/rapidsai/nx-cugraph/pull/175)) @gforsyth](https://github.com/v5 ([#175](https://github.com/rapidsai/nx-cugraph/pull/175)) @gforsyth)
- Allow latest OS in devcontainers ([#171](https://github.com/rapidsai/nx-cugraph/pull/171)) [@bdice](https://github.com/bdice)

# nx-cugraph 25.08.00 (6 Aug 2025)

## 🚨 Breaking Changes

- Remove CUDA 11 from dependencies.yaml ([#151](https://github.com/rapidsai/nx-cugraph/pull/151)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- stop uploading packages to downloads.rapids.ai ([#145](https://github.com/rapidsai/nx-cugraph/pull/145)) [@jameslamb](https://github.com/jameslamb)

## 🐛 Bug Fixes

- Catch `CUDARuntimeError` in CuPy GPU check ([#166](https://github.com/rapidsai/nx-cugraph/pull/166)) [@jakirkham](https://github.com/jakirkham)
- Temporarily skip certain tests to unblock CI ([#163](https://github.com/rapidsai/nx-cugraph/pull/163)) [@nv-rliu](https://github.com/nv-rliu)
- Address Test Failures for NetworkX 3.5 ([#160](https://github.com/rapidsai/nx-cugraph/pull/160)) [@nv-rliu](https://github.com/nv-rliu)
- Updates can_run function signature for forceatlas2_layout ([#158](https://github.com/rapidsai/nx-cugraph/pull/158)) [@rlratzel](https://github.com/rlratzel)

## 📖 Documentation

- add docs on CI workflow inputs ([#164](https://github.com/rapidsai/nx-cugraph/pull/164)) [@jameslamb](https://github.com/jameslamb)

## 🛠️ Improvements

- Disable codecov comments ([#170](https://github.com/rapidsai/nx-cugraph/pull/170)) [@bdice](https://github.com/bdice)
- Removes references to CUDA 11 in docs, notebooks ([#169](https://github.com/rapidsai/nx-cugraph/pull/169)) [@rlratzel](https://github.com/rlratzel)
- Updates benchmarks to allow for running in a CPU-only environment ([#165](https://github.com/rapidsai/nx-cugraph/pull/165)) [@rlratzel](https://github.com/rlratzel)
- Use CUDA 12.9 in Conda, Devcontainers, Spark, GHA, etc. ([#161](https://github.com/rapidsai/nx-cugraph/pull/161)) [@jakirkham](https://github.com/jakirkham)
- refactor(shellcheck): enable for all files and fix remaining warnings ([#156](https://github.com/rapidsai/nx-cugraph/pull/156)) [@gforsyth](https://github.com/gforsyth)
- Remove nvidia and dask channels ([#155](https://github.com/rapidsai/nx-cugraph/pull/155)) [@vyasr](https://github.com/vyasr)
- Add Colab Demo link to README ([#154](https://github.com/rapidsai/nx-cugraph/pull/154)) [@nv-rliu](https://github.com/nv-rliu)
- Add Leiden to Notebook ([#152](https://github.com/rapidsai/nx-cugraph/pull/152)) [@nv-rliu](https://github.com/nv-rliu)
- Remove CUDA 11 from dependencies.yaml ([#151](https://github.com/rapidsai/nx-cugraph/pull/151)) [@KyleFromNVIDIA](https://github.com/KyleFromNVIDIA)
- Demo notebook for 25.06 features ([#150](https://github.com/rapidsai/nx-cugraph/pull/150)) [@rlratzel](https://github.com/rlratzel)
- Use environment as default for RAPIDS_DATASET_ROOT_DIR, improve benchmark docs ([#148](https://github.com/rapidsai/nx-cugraph/pull/148)) [@bdice](https://github.com/bdice)
- Remove CUDA 11 devcontainers and update CI scripts ([#146](https://github.com/rapidsai/nx-cugraph/pull/146)) [@bdice](https://github.com/bdice)
- stop uploading packages to downloads.rapids.ai ([#145](https://github.com/rapidsai/nx-cugraph/pull/145)) [@jameslamb](https://github.com/jameslamb)
- Forward-merge branch-25.06 into branch-25.08 ([#139](https://github.com/rapidsai/nx-cugraph/pull/139)) [@gforsyth](https://github.com/gforsyth)
- Forward-merge branch-25.06 into branch-25.08 ([#134](https://github.com/rapidsai/nx-cugraph/pull/134)) [@gforsyth](https://github.com/gforsyth)

# nx-cugraph 25.06.00 (5 Jun 2025)

## 🐛 Bug Fixes

- Fix update-version.sh ([#124](https://github.com/rapidsai/nx-cugraph/pull/124)) [@raydouglass](https://github.com/raydouglass)
- Fix edgecase when converting to networkx w/o fallback ([#118](https://github.com/rapidsai/nx-cugraph/pull/118)) [@eriknw](https://github.com/eriknw)

## 📖 Documentation

- Fix Typo in Contributing Guide ([#111](https://github.com/rapidsai/nx-cugraph/pull/111)) [@nv-rliu](https://github.com/nv-rliu)

## 🛠️ Improvements

- use &#39;rapids-init-pip&#39; in wheel CI, other CI changes ([#141](https://github.com/rapidsai/nx-cugraph/pull/141)) [@jameslamb](https://github.com/jameslamb)
- Finish CUDA 12.9 migration and use branch-25.06 workflows ([#138](https://github.com/rapidsai/nx-cugraph/pull/138)) [@bdice](https://github.com/bdice)
- Quote head_rev in conda recipes ([#137](https://github.com/rapidsai/nx-cugraph/pull/137)) [@bdice](https://github.com/bdice)
- Build and test with CUDA 12.9.0 ([#135](https://github.com/rapidsai/nx-cugraph/pull/135)) [@bdice](https://github.com/bdice)
- Add support for Python 3.13 ([#132](https://github.com/rapidsai/nx-cugraph/pull/132)) [@gforsyth](https://github.com/gforsyth)
- Branch 25.06 merge 25.04 ([#130](https://github.com/rapidsai/nx-cugraph/pull/130)) [@nv-rliu](https://github.com/nv-rliu)
- Branch 25.06 merge 25.04 (Trying Again) ([#128](https://github.com/rapidsai/nx-cugraph/pull/128)) [@nv-rliu](https://github.com/nv-rliu)
- Dispatching `to_numpy_array` ([#127](https://github.com/rapidsai/nx-cugraph/pull/127)) [@nv-rliu](https://github.com/nv-rliu)
- Download build artifacts from Github for CI ([#122](https://github.com/rapidsai/nx-cugraph/pull/122)) [@VenkateshJaya](https://github.com/VenkateshJaya)
- feat(rattler): port conda build recipe to rattler build ([#121](https://github.com/rapidsai/nx-cugraph/pull/121)) [@gforsyth](https://github.com/gforsyth)
- Add `to_scipy_sparse_array` ([#120](https://github.com/rapidsai/nx-cugraph/pull/120)) [@eriknw](https://github.com/eriknw)
- Add ARM conda environments ([#119](https://github.com/rapidsai/nx-cugraph/pull/119)) [@bdice](https://github.com/bdice)
- Branch 25.06 merge 25.04 ([#108](https://github.com/rapidsai/nx-cugraph/pull/108)) [@rlratzel](https://github.com/rlratzel)
- Forward-merge branch-25.04 into branch-25.06 ([#105](https://github.com/rapidsai/nx-cugraph/pull/105)) [@nv-rliu](https://github.com/nv-rliu)
- Moving wheel builds to specified location and uploading build artifacts to Github ([#104](https://github.com/rapidsai/nx-cugraph/pull/104)) [@VenkateshJaya](https://github.com/VenkateshJaya)

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
