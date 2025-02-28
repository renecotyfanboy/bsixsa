#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xspec

xspec.AllData.clear()
xspec.AllModels.clear()

xspec.Xset.restore("model.xcm")

xspec.Fit.statMethod = "cstat"
xspec.Fit.bayes = "on"

xspec_observation = xspec.Spectrum(
    "spectrum_raw.pha"
)
xspec_observation.background = None

low_energy, high_energy = 0.3, 12.
xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**")

xspec_model = xspec.AllModels(1)
xspec_model.show()


# In[2]:


from bsixsa.convenience import set_prior_and_build_transform

define_prior = [
    ("bapec", "kT", 3, 6, "uniform"),
    ("bapec", "Abundanc", 0.5, 2.5, "uniform"),
    ("bapec", "Redshift", 0., 0.05, "uniform"),
    ("bapec", "Velocity", 50, 200, "uniform"),
    ("bapec", "norm", 0.5, 1.5, "loguniform"),
    ("bapec_3", "kT", 1, 4, "uniform"),
    ("bapec_3", "Abundanc", 0.5, 2.5, "uniform"),
    ("bapec_3", "Redshift", 0., 0.05, "uniform"),
    ("bapec_3", "Velocity", 50, 200, "uniform"),
    ("bapec_3", "norm", 1, 2, "loguniform")
]

transformations = set_prior_and_build_transform(xspec_model, define_prior)


# In[4]:


import bxa.xspec as bxa

outputfiles_basename = 'bxa_results/'
solver = bxa.BXASolver(transformations=transformations, outputfiles_basename=outputfiles_basename)
results = solver.run(resume=True, n_live_points=1000, speed="safe")


# In[ ]:


from bsixsa import SIXSASolver

outputfiles_basename = "sixsa_result/"

solver = SIXSASolver(
    transformations,
    outputfiles_basename=outputfiles_basename
)


# In[ ]:


import pandas as pd

parameter_names = solver.parameter_names_uniques
param_dict = {}

for i, t in enumerate(solver.transformations):

    j = t["index"]
    name = parameter_names[j-1]
    if t["name"].startswith("log"):
        param_dict[name] = [10**sample[i] for sample in results["samples"]]
    else:
        param_dict[name] = [sample[i] for sample in results["samples"]]
df = pd.DataFrame.from_dict(param_dict)


# In[ ]:


df.to_csv('results_df/bxa.csv')

