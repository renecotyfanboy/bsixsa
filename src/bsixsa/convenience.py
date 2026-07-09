from xspec import Xset, Model, Component, Parameter, AllModels
import os
import re
from dataclasses import dataclass
from typing import Iterator
from collections.abc import Callable
from contextlib import contextmanager
from time import perf_counter


def available_cpu_count() -> int:
    """CPUs actually available to this process.

    Respects cgroup/cpuset/SLURM affinity on Linux; falls back to the full
    core count on platforms without ``sched_getaffinity`` (e.g. macOS).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# Components defined more than once (e.g. ``tbabs*(powerlaw + powerlaw)``) are
# reported by XSPEC as ``powerlaw`` and ``powerlaw_3``: an ``_<int>`` suffix.
_COMPONENT_INDEX_RE = re.compile(r'.*_\d+$')


def has_indexed_suffix(name: str) -> bool:
    """True if a component name ends in an ``_<int>`` index suffix."""
    return bool(_COMPONENT_INDEX_RE.fullmatch(name))


@contextmanager
def catchtime(desc="Task", print_time=True) -> Callable[[], float]:
    """Context manager to measure time taken by a task.

    Parameters:
        desc: Description of the task.
        print_time: Whether to print the time taken by the task.

    Returns:
        Callable that returns the time taken by the task in seconds.
    """

    t1 = t2 = perf_counter()
    yield lambda: t2 - t1
    t2 = perf_counter()
    if print_time:
        print(f"{desc}: {t2 - t1:.3f} seconds")


@dataclass
class XspecParameter:
    """Class for keeping track of xspec parameters."""
    name: str
    model: Model
    component: Component
    parameter: Parameter
    absolute_idx: int # Model independent index for thawn parameters


def iter_components(model: Model, additive_only: bool = False) -> Iterator[Component]:
    """Iterate over components of a given xspec model.

    Parameters:
        additive_only: If ``True``, yield only additive components as
            identified by the presence of a ``norm`` parameter.
    """

    for component_name in model.componentNames:
        component = getattr(model, component_name)
        if additive_only and "norm" not in component.parameterNames:
            continue
        yield component
def iter_parameters(component: Component) -> Iterator[Parameter]:
    """Iterate over all parameters of a given xspec model component."""

    for parameter_name in component.parameterNames:
        yield getattr(component, parameter_name)


def iter_all_parameters() -> Iterator[XspecParameter]:
    """
    Iterate over all possible parameters as defined in the Xspec models.
    """

    absolute_idx_tracker = -1

    for model_idx, model_name in AllModels.sources.items():

        # Assume a single datagroup is active
        model = AllModels(1, model_name)

        for component_index, component in enumerate(iter_components(model)):

            component_name = component.name

            # Handle the weird situation where a component is defined multiple times
            # EG tbabs*(powerlaw + powerlaw) will yield tbabs, powerlaw & powerlaw_3 as component names
            # We explicitly remove the _index if it is here and reapply it so that every component has an index
            if has_indexed_suffix(component.name):
                component_name = component_name.split('_')[0]

            component_name = f"{component_name}_{component_index + 1}"

            for parameter in iter_parameters(component):

                if not parameter.frozen:
                    absolute_idx_tracker += 1
                    absolute_idx = absolute_idx_tracker

                else:
                    absolute_idx = -1

                if model_name == "":
                    name = f"{component_name}_{parameter.name}"
                else:
                    name = f"{model_name}_{component_name}_{parameter.name}"

                yield XspecParameter(name, model, component, parameter, absolute_idx)


def iter_thawn_parameters() -> Iterator[XspecParameter]:
    """
    Iterate over all thawn parameters as defined in the Xspec models.
    """

    for xspec_parameter in iter_all_parameters():
        if not (xspec_parameter.parameter.frozen or bool(xspec_parameter.parameter.link)):
            yield xspec_parameter


class XSilence(object):
    """Context for temporarily making xspec quiet."""

    def __enter__(self):
        self.oldchatter = Xset.chatter, Xset.logChatter
        Xset.chatter, Xset.logChatter = 0, 0

    def __exit__(self, *args):
        Xset.chatter, Xset.logChatter = self.oldchatter
