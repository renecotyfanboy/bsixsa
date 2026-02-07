from __future__ import print_function


from xspec import Xset, Model, Component, Parameter, AllModels, AllData
import re
from dataclasses import dataclass
from typing import Iterator


@dataclass
class XspecParameter:
    """Class for keeping track of xspec parameters."""
    name: str
    model: Model
    component: Component
    parameter: Parameter
    absolute_idx: int # Model independent index for thawn parameters


def iter_components(model: Model) -> Iterator[Component]:
    """Iterate over all components of a given xspec model."""

    for component_name in model.componentNames:
        yield getattr(model, component_name)


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
            if bool(re.fullmatch(r'.*_\d+$', component.name)):
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
        if not xspec_parameter.parameter.frozen:
            yield xspec_parameter


def build_parameter_name():
    """
    Return {XSPEC parameter index -> unique parameter name}.
    """

    sources_models = AllModels.sources

    name_map = [{} for k in range(len(sources_models))]

    for model_nb, (source, model_name) in enumerate(zip(sources_models.keys(), sources_models.values())):
        xspec_model = AllModels(1, model_name)
        for comp_index, comp_name in enumerate(xspec_model.componentNames):
            comp = getattr(xspec_model, comp_name)

            # Handle the weird situation where a component is defined multiple times
            # EG tbabs*(powerlaw + powerlaw) will yield tbabs, powerlaw & powerlaw_3 as component names
            # Doing so we ensure that every parameter is linked to its component number and avoid duplicates
            # like powerlaw_3_3
            if bool(re.fullmatch(r'.*_\d+$', comp_name)):
                comp_name = comp_name.split('_')[0]

            for par_name in comp.parameterNames:
                par = getattr(comp, par_name)
                name_map[model_nb][par.index] = f"{str(comp_name)}_{comp_index + 1}_{str(par_name)}"

    return name_map


class XSilence(object):
    """Context for temporarily making xspec quiet."""

    def __enter__(self):
        self.oldchatter = Xset.chatter, Xset.logChatter
        Xset.chatter, Xset.logChatter = 0, 0

    def __exit__(self, *args):
        Xset.chatter, Xset.logChatter = self.oldchatter
