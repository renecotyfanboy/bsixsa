from __future__ import print_function

import xspec
from xspec import Xset
import re

def build_parameter_name():
    """
    Return {XSPEC parameter index -> unique parameter name}.
    """

    sources_models = xspec.AllModels.sources

    name_map = [{} for k in range(len(sources_models))]

    for model_nb, (source, model_name) in enumerate(zip(sources_models.keys(), sources_models.values())):
        xspec_model = xspec.AllModels(1, model_name)
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


def load_xspec_data(
    path="spectrum_opt.pha", low_energy=0.3, high_energy=12.0, lmod=None
):
    xspec.AllData.clear()
    xspec.AllModels.clear()

    with XSilence():
        if lmod is not None:
            xspec.AllModels.lmod(lmod)

        xspec.Xset.restore("model.xcm")
        xspec.Fit.statMethod = "cstat"
        xspec.Fit.bayes = "on"

        xspec_observation = xspec.Spectrum(path)
        xspec_observation.background = None
        xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**")

        xspec_model = xspec.AllModels(1)

    return xspec_model, xspec_observation


class XSilence(object):
    """Context for temporarily making xspec quiet."""

    def __enter__(self):
        self.oldchatter = Xset.chatter, Xset.logChatter
        Xset.chatter, Xset.logChatter = 0, 0

    def __exit__(self, *args):
        Xset.chatter, Xset.logChatter = self.oldchatter
