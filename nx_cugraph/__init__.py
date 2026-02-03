# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from networkx.exception import *

from _nx_cugraph._version import __git_commit__, __version__
from _nx_cugraph import _check_networkx_version

_nxver: tuple[int, int] | tuple[int, int, int] = _check_networkx_version()

from . import utils

from . import classes
from .classes import *

from . import convert
from .convert import *

from . import convert_matrix
from .convert_matrix import *

from . import relabel
from .relabel import *

from . import generators
from .generators import *

from . import algorithms
from .algorithms import *

from . import linalg
from .linalg import *

from . import drawing
from .drawing import *

from .interface import BackendInterface

BackendInterface.Graph = classes.Graph
BackendInterface.DiGraph = classes.DiGraph
BackendInterface.MultiGraph = classes.MultiGraph
BackendInterface.MultiDiGraph = classes.MultiDiGraph
del BackendInterface
