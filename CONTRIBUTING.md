# Contributing to nx-cugraph

If you are interested in contributing to nx-cugraph, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/nx-cugraph/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - Please run and paste the output of the `print_env.sh` script while
    reporting a bug to gather and report relevant environment details.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention,
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    assist.

## Code Contributions

### Your First Issue

1. Read the project's [README.md](https://github.com/rapidsai/nx-cugraph/blob/main/README.md)
    to learn how to setup the development environment. # FIXME ADD THIS SECTION
2. Find an issue to work on. [Issues](https://github.com/rapidsai/nx-cugraph/issues)
    or [help wanted](https://github.com/rapidsai/nx-cugraph/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it.
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/nx-cugraph/compare).
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/), or fix if needed.
7. Wait for other developers to review your code and update code as needed.
8. Once reviewed and approved, a RAPIDS developer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!


## Building and Installing From Source
These instructions are tested on supported versions/distributions of Linux,
CUDA, and Python - See [RAPIDS Getting Started](https://rapids.ai/start.html)
for the list of supported environments.  Other environments _might be_
compatible, but are not currently tested. Further details about requirements are described in the [RAPIDS System Requirements page](https://docs.rapids.ai/install#system-req).

### Create the build environment
- Clone the repository:

```bash
git clone https://github.com/rapidsai/nx-cugraph.git
cd nx-cugraph
```

#### Create conda environment and install dependencies

```bash
# for CUDA 13 environments
conda env create --name nxcg-dev --file conda/environments/all_cuda-131_arch-$(uname -m).yaml

# activate the environment
conda activate nxcg-dev
```

- **Note**: the conda environment files are updated frequently, so the
  development environment may also need to be updated if dependency versions or
  pinnings are changed.

#### Build nx-cugraph from source

```bash
# for details and usage, run with the --help flag
./build.sh
```

- **Note**: if Python files (`*.py`) have changed, the build must be rerun.

To verify your install with tests, run
```bash
pytest -v nx_cugraph/tests
```

## Code Formatting

Consistent code formatting is important in the nx-cugraph project to ensure
readability, maintainability, and thus simplifies collaboration.

### Using pre-commit hooks

nx-cugraph uses [pre-commit](https://pre-commit.com) to execute code linters and
formatters that check the code for common issues, such as syntax errors, code
style violations, and help to detect bugs. Using pre-commit ensures that linter
versions and options are aligned for all developers. The same hooks are executed
as part of the CI checks. This means running pre-commit checks locally avoids
unnecessary CI iterations.

To use `pre-commit`, install the tool via `conda` or `pip` into your development
environment:

```bash
conda install -c conda-forge pre-commit
```
Alternatively:
```bash
pip install pre-commit
```

After installing pre-commit, it is recommended to install pre-commit hooks to
run automatically before creating a git commit. In this way, it is less likely
that style checks will fail as part of CI checks. To install pre-commit hooks,
simply run the following command within the repository root directory:

```bash
pre-commit install
```

By default, pre-commit runs on staged files only, meaning only on changes that
are about to be committed. To run pre-commit checks on all files, execute:

```bash
pre-commit run --all-files
```

To skip the checks temporarily, use `git commit --no-verify` or its short form
`-n`.

_Note_: If the auto-formatters' changes affect each other, you may need to go
through multiple iterations of `git commit` and `git add -u`.

<!-- nx-cugraph also
mistakes, and this check is run as part of the pre-commit hook. To apply the suggested spelling
fixes, you can run  `codespell -i 3 -w .` from the command-line in the nx-cugraph root directory.
This will bring up an interactive prompt to select which spelling fixes to apply.


If you want to ignore errors highlighted by codespell you can:
 * Add the word to the ignore-words-list in pyproject.toml, to exclude for all of cuML
 * Exclude the entire file from spellchecking, by adding to the `exclude` regex in .pre-commit-config.yaml
 * Ignore only specific lines as shown in https://github.com/codespell-project/codespell/issues/1212#issuecomment-654191881
-->

### Summary of pre-commit hooks

The pre-commit hooks configured for this repository consist of a number of
linters and auto-formatters that we summarize here. For a full and current list,
please see the `.pre-commit-config.yaml` file.

### Managing PR labels

Each PR must be labeled according to whether it is a "breaking" or "non-breaking" change (using Github labels). This is used to highlight changes that users should know about when upgrading.

For nx-cugraph, a "breaking" change is one that modifies the public, non-experimental, Python API in a
non-backward-compatible way.

Additional labels must be applied to indicate whether the change is a feature, improvement, bugfix, or documentation change. See the shared RAPIDS documentation for these labels: https://github.com/rapidsai/kb/issues/42.

<!--
### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/nx-cugraph/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.
-->

### Branches and Versions

The nx-cugraph repository has two main branches:

1. `main` branch: it contains the last released version. Only hotfixes are targeted and merged into it.
2. `branch-x.y`: it is the development branch which contains the upcoming release. All the new features should be based on this branch and Merge/Pull request should target this branch (with the exception of hotfixes).

### Additional details

For every new version `x.y` of nx-cugraphs there is a corresponding branch called `branch-x.y`, from where new feature development starts and PRs will be targeted and merged before its release. The exceptions to this are the 'hotfixes' that target the `main` branch, which target critical issues raised by Github users and are directly merged to `main` branch, and create a new subversion of the project. While trying to patch an issue which requires a 'hotfix', please state the intent in the PR.

For all development, your changes should be pushed into a branch (created using the naming instructions below) in your own fork of nx-cugraph and then create a pull request when the code is ready.

A few days before releasing version `x.y` the code of the current development branch (`branch-x.y`) will be frozen and a new branch, 'branch-x+1.y' will be created to continue development.

### Branch naming

Branches used to create PRs should have a name of the form `<type>-<name>`
which conforms to the following conventions:
- Type:
    - fea - For if the branch is for a new feature(s)
    - enh - For if the branch is an enhancement of an existing feature(s)
    - bug - For if the branch is for fixing a bug(s) or regression(s)
- Name:
    - A name to convey what is being worked on
    - Please use dashes or underscores between words as opposed to spaces.

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
