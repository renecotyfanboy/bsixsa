#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xspec

xspec.AllData.clear()
xspec.AllModels.clear()
xspec.AllModels.lmod("relxill")

xspec.Xset.restore("model.xcm")

xspec.Fit.statMethod = "cstat"
xspec.Fit.bayes = "on"
xspec.Fit.query = "no"
xspec.Fit.criticalDelta = 1e-3
xspec.Fit.nIterations = 50
xspec.Fit.bayes = "cons"

xspec_observation = xspec.Spectrum("spectrum_opt.pha")
xspec_observation.background = None
low_energy, high_energy = 0.3, 10.
xspec_observation.ignore(f"0.0-{low_energy:.1f} {high_energy:.1f}-**")

xspec_model = xspec.AllModels(1)
xspec_model.show()


# In[2]:


from bsixsa.convenience import set_prior_and_build_transform

define_prior = [
    ("TBabs", "nH", 0.1, 0.3, "uniform"),       # 10^22    0.100000     +/-  0.0
    ("relxillNS", "a", 0.05, 0.4, "uniform"),      #   keV      0.550000     +/-  0.0
    ("relxillNS", "kTbb", 0.6, 2.0, "uniform"),      #  keV      2.50000      +/-  0.0
    ("relxillNS", "logxi", 1, 4.7, "uniform"),    #            2.50000      +/-  0.0
    ("relxillNS", "refl_frac", 0.5, 4.5, "uniform"),    #            1.00000      +/-  0.0
    ("relxillNS", "norm", 1e-3, 1e-1, "loguniform")
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

# I don't know why I have to do this, but the logxi values are not the same
df["logxi"] = df["logxi"].apply(lambda x: np.log10(x))
df.to_csv('results_df/bxa.csv')

