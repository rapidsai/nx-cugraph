# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
SHELL= /bin/bash

.PHONY: all
all: plugin-info readme

.PHONY: plugin-info
plugin-info:
	python _nx_cugraph/__init__.py

.PHONY: readme
readme:
	python scripts/update_readme.py README.md
