"""Helpers around PyXspec: iterate over model parameters, silence XSPEC, count CPUs."""

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from xspec import AllData, AllModels, Component, Model, Parameter, Xset


def available_cpu_count() -> int:
    """Number of CPUs available to this process (respects SLURM/cgroup affinity on Linux)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def base_component_name(name: str) -> str:
    """Strip the ``_<position>`` suffix XSPEC adds to a repeated component.

    For example ``powerlaw_3`` becomes ``powerlaw``.
    """
    return re.sub(r"_\d+$", "", name)


@dataclass
class XspecParameter:
    """One parameter of a loaded XSPEC model.

    Attributes:
        name: Unique name ``<component>_<position>_<parameter>``, prefixed by
            ``<model>_`` for a named model, e.g. ``TBabs_1_nH`` or
            ``cluster_bapec_2_kT``. ``<position>`` is the 1-based position of
            the component in the model expression. These names are the columns
            of the fit results.
        model: The PyXspec model (its copy in its first data group, see
            [`first_model_copy`][bsixsa.convenience.first_model_copy]).
        component: The PyXspec component.
        parameter: The PyXspec parameter.
    """

    name: str
    model: Model
    component: Component
    parameter: Parameter


def model_data_groups(model_name: str) -> list[int]:
    """Data groups holding a copy of a loaded model (`''` for the unnamed one), in increasing order.

    XSPEC keeps one copy of a model per data group whose spectra have a
    response for its source: a background model of the third spectrum only
    exists in that spectrum's data group. Without data, the only copy is in
    data group 1.
    """
    groups = []
    for group in range(1, max(AllData.nGroups, 1) + 1):
        try:
            AllModels(group, model_name)
        except Exception:  # PyXspec: "No model object exists for model name ..."
            continue
        groups.append(group)
    return groups


def first_model_copy(model_name: str) -> Model:
    """The copy of a loaded model in its first data group, whose parameters bsixsa fits.

    Its parameter numbers are XSPEC's `<model>:<number>`; the parameters of
    the copies in later data groups must be frozen or linked.

    Raises:
        ValueError: If no data group holds a copy of the model.
    """
    groups = model_data_groups(model_name)
    if not groups:
        raise ValueError(f"No data group holds a copy of the XSPEC model {model_name!r}.")
    return AllModels(groups[0], model_name)


def iter_components(model: Model, additive_only: bool = False) -> Iterator[Component]:
    """Iterate over the components of an XSPEC model, in expression order.

    Parameters:
        model: A PyXspec model.
        additive_only: Yield only components with a ``norm`` parameter.
    """
    for component_name in model.componentNames:
        component = getattr(model, component_name)
        if additive_only and "norm" not in component.parameterNames:
            continue
        yield component


def iter_all_parameters() -> Iterator[XspecParameter]:
    """Iterate over every parameter of every loaded model, frozen and linked ones included.

    Each model is read from its first data group (see
    [`first_model_copy`][bsixsa.convenience.first_model_copy]).

    Yields:
        One [`XspecParameter`][bsixsa.convenience.XspecParameter] per parameter.
    """
    for model_name in AllModels.sources.values():
        model = first_model_copy(model_name)
        for position, component in enumerate(iter_components(model), start=1):
            component_name = f"{base_component_name(component.name)}_{position}"
            prefix = f"{model_name}_" if model_name else ""
            for parameter_name in component.parameterNames:
                parameter = getattr(component, parameter_name)
                name = f"{prefix}{component_name}_{parameter.name}"
                yield XspecParameter(name, model, component, parameter)


def iter_free_parameters() -> Iterator[XspecParameter]:
    """Iterate over the free parameters (neither frozen nor linked), i.e. the prior axes.

    Yields:
        One [`XspecParameter`][bsixsa.convenience.XspecParameter] per free parameter.
    """
    for xspec_parameter in iter_all_parameters():
        if not (xspec_parameter.parameter.frozen or xspec_parameter.parameter.link):
            yield xspec_parameter


@contextmanager
def XSilence() -> Iterator[None]:
    """Silence XSPEC's console and log output inside a ``with`` block.

    Example:
        ```python
        with XSilence():
            xspec.AllData("spectrum.pha")
        ```
    """
    previous = Xset.chatter, Xset.logChatter
    Xset.chatter, Xset.logChatter = 0, 0
    try:
        yield
    finally:
        Xset.chatter, Xset.logChatter = previous
