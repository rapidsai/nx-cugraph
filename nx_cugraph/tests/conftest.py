# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
def pytest_configure(config):
    if config.getoption("--all", False):
        # Run benchmarks AND tests
        config.option.benchmark_skip = False
        config.option.benchmark_enable = True
    elif config.getoption("--bench", False) or config.getoption(
        "--benchmark-enable", False
    ):
        # Run benchmarks (and only benchmarks) with `--bench` argument
        config.option.benchmark_skip = False
        config.option.benchmark_enable = True
        if not config.option.keyword:
            config.option.keyword = "bench_"
    else:
        # Run only tests
        config.option.benchmark_skip = True
        config.option.benchmark_enable = False
        if not config.option.keyword:
            config.option.keyword = "test_"
