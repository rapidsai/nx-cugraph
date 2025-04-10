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
from __future__ import annotations

from functools import partial, update_wrapper
from textwrap import dedent

import networkx as nx
from networkx import NetworkXError
from networkx.utils.decorators import nodes_or_number, not_implemented_for

import nx_cugraph as nxcg
from nx_cugraph import _nxver
from nx_cugraph.interface import BackendInterface

from .misc import _And_NotImplementedError

try:
    from networkx.utils.backends import _registered_algorithms
except ModuleNotFoundError:
    from networkx.classes.backends import _registered_algorithms


__all__ = ["not_implemented_for", "nodes_or_number", "networkx_algorithm"]


def networkx_class(api):
    def inner(func):
        func.__doc__ = getattr(api, func.__name__).__doc__
        return func

    return inner


class networkx_algorithm:
    name: str
    extra_doc: str | None
    extra_params: dict[str, str] | None
    version_added: str
    is_incomplete: bool
    is_different: bool
    _fallback: bool
    _plc_names: set[str] | None

    def __new__(
        cls,
        func=None,
        *,
        name: str | None = None,
        # Extra parameter info that is added to NetworkX docstring
        extra_params: dict[str, str] | str | None = None,
        # Applies `nodes_or_number` decorator compatibly across versions (3.3 changed)
        nodes_or_number: list[int] | int | None = None,
        # Alternative way to provide the docstring to add to the networkx docstring
        docstring: str | None = None,
        # Metadata (for introspection only)
        version_added: str,  # Required
        is_incomplete: bool = False,  # See self.extra_doc for details if True
        is_different: bool = False,  # See self.extra_doc for details if True
        fallback: bool = False,  # Change non-nx exceptions to NotImplementedError
        # The position of `create_using` arg; sets `can_run` to check `create_using`
        create_using_arg: int | None = None,
        _plc: str | set[str] | None = None,  # Hidden from user, may be removed someday
    ):
        if func is None:
            return partial(
                networkx_algorithm,
                name=name,
                extra_params=extra_params,
                nodes_or_number=nodes_or_number,
                docstring=docstring,
                version_added=version_added,
                is_incomplete=is_incomplete,
                is_different=is_different,
                fallback=fallback,
                create_using_arg=create_using_arg,
                _plc=_plc,
            )
        if docstring:
            func.__doc__ = docstring
        instance = object.__new__(cls)
        if nodes_or_number is not None and _nxver >= (3, 3):
            func = nx.utils.decorators.nodes_or_number(nodes_or_number)(func)
        # update_wrapper sets __wrapped__, which will be used for the signature
        update_wrapper(instance, func)
        instance.__defaults__ = func.__defaults__
        instance.__kwdefaults__ = func.__kwdefaults__
        instance.name = func.__name__ if name is None else name
        if extra_params is None:
            pass
        elif isinstance(extra_params, str):
            extra_params = {extra_params: ""}
        elif not isinstance(extra_params, dict):
            raise TypeError(
                f"extra_params must be dict, str, or None; got {type(extra_params)}"
            )
        instance.extra_params = extra_params
        if _plc is None or isinstance(_plc, set):
            instance._plc_names = _plc
        elif isinstance(_plc, str):
            instance._plc_names = {_plc}
        else:
            raise TypeError(
                f"_plc argument must be str, set, or None; got {type(_plc)}"
            )
        instance.version_added = version_added
        instance.is_incomplete = is_incomplete
        instance.is_different = is_different
        instance.fallback = fallback
        instance.create_using_arg = create_using_arg
        # The docstring on our function is added to the NetworkX docstring.
        instance.extra_doc = (
            dedent(func.__doc__.lstrip("\n").rstrip()) if func.__doc__ else None
        )
        # Copy __doc__ from NetworkX
        if instance.name in _registered_algorithms:
            instance.__doc__ = _registered_algorithms[instance.name].__doc__
        if instance.create_using_arg is None:
            instance.can_run = _default_can_run
        else:
            instance.can_run = instance._check_create_using_can_run
        instance.should_run = _default_should_run
        setattr(BackendInterface, instance.name, instance)
        # Set methods so they are in __dict__
        instance._can_run = instance._can_run
        instance._should_run = instance._should_run
        if nodes_or_number is not None and _nxver < (3, 3):
            instance = nx.utils.decorators.nodes_or_number(nodes_or_number)(instance)
        return instance

    def _can_run(self, func):
        """Set the `can_run` attribute to the decorated function."""
        if not func.__name__.startswith("_"):
            raise ValueError(
                "The name of the function used by `_can_run` must begin with '_'; "
                f"got: {func.__name__!r}"
            )
        self.can_run = func

    def _should_run(self, func):
        """Set the `should_run` attribute to the decorated function."""
        if not func.__name__.startswith("_"):
            raise ValueError(
                "The name of the function used by `_should_run` must begin with '_'; "
                f"got: {func.__name__!r}"
            )
        self.should_run = func

    def __call__(self, /, *args, **kwargs):
        if not self.fallback:
            return self.__wrapped__(*args, **kwargs)
        try:
            return self.__wrapped__(*args, **kwargs)
        except NetworkXError:
            raise
        except Exception as exc:
            raise _And_NotImplementedError(exc) from exc

    def __reduce__(self):
        return _restore_networkx_dispatched, (self.name,)

    def _check_create_using_can_run(self, *args, **kwargs):
        """``can_run`` function to check whether ``create_using`` argument is valid."""
        if self.create_using_arg < len(args):
            create_using = args[self.create_using_arg]
        else:
            create_using = kwargs.get("create_using")
        if (
            create_using is None
            or isinstance(create_using, (nxcg.Graph, nxcg.CudaGraph))
            or isinstance(create_using, type)
            and (
                issubclass(create_using, (nxcg.Graph, nxcg.CudaGraph))
                or create_using
                in {nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph}
            )
        ):
            return True
        if isinstance(create_using, nx.Graph):
            return "`create_using=G` with instance of `nx.Graph` is not supported"
        return "Invalid `create_using=` argument"


def _default_can_run(*args, **kwargs):
    return True


def _default_should_run(*args, **kwargs):
    return True


def _restore_networkx_dispatched(name):
    return getattr(BackendInterface, name)


def _gpu_cpu_api(nx_class, module_name):
    def _gpu_cpu_graph_method(attr, *, edge_data=False, node_data=False):
        """Dispatch property to NetworkX or CudaGraph based on cache.

        For example, this will use any cached CudaGraph for ``len(G)``, which
        prevents creating NetworkX data structures.
        """
        nx_func = getattr(nx_class, attr)

        def inner(self, *args, **kwargs):
            cuda_graph = self._get_cudagraph(edge_data=edge_data, node_data=node_data)
            if cuda_graph is None:
                return nx_func(self, *args, **kwargs)
            return getattr(cuda_graph, attr)(*args, **kwargs)

        inner.__name__ = nx_func.__name__
        inner.__doc__ = nx_func.__doc__
        inner.__qualname__ = nx_func.__qualname__
        inner.__defaults__ = nx_func.__defaults__
        inner.__kwdefaults__ = nx_func.__kwdefaults__
        inner.__module__ = module_name
        inner.__dict__.update(nx_func.__dict__)
        inner.__wrapped__ = nx_func
        return inner

    return _gpu_cpu_graph_method
