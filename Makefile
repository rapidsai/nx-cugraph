# SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
SHELL= /bin/bash

.PHONY: all
all: plugin-info readme

.PHONY: plugin-info
plugin-info:
	python _nx_cugraph/__init__.py

objects.inv:
	wget https://networkx.org/documentation/stable/objects.inv

.PHONY: readme
readme: objects.inv
	python scripts/update_readme.py README.md objects.inv
