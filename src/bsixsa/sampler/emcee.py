#%%
import numpy as np
import emcee
import torch

def is_truthy(x):
    return bool(x) and torch.is_tensor(x[0]) and x[0].numel() > 0

def log_prob(x):

    x = torch.asarray(x, dtype=torch.float32)
    prior = solver.log_prior_fn(x, None)
    likelihood = torch.zeros_like(prior)
    finite_idx = torch.where(~torch.isinf(prior))

    if is_truthy(finite_idx):
        likelihood[finite_idx] = solver.log_likelihood_fn(x[finite_idx], None, progress_bar=False, no_pool=False)

    log_prob = prior + likelihood

    return log_prob.numpy()

ndim, nwalkers = len(model.names), 100
p0 = live_points_to_array(sampler.posterior_samples, names=model.names, copy=True)[:nwalkers]  #np.median(, axis=0)[None, :] + np.random.normal(0, 1e-6, size=(nwalkers, ndim))

emcee_sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True)
emcee_sampler.run_mcmc(
    p0, 1000, progress=True
)
#%%
emcee_results = emcee_sampler.get_chain(flat=True)