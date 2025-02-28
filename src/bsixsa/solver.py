from __future__ import print_function

from typing import Any

import numpy as np
from ultranest.plot import PredictionBand
import os
from math import isnan, isinf
import numpy
import warnings
import sys
import pandas as pd
from torchmin import minimize
from xspec import Xset, AllModels, Fit, Plot
from chainconsumer import ChainConsumer, Chain, Truth
from sbi.neural_nets import posterior_nn
from .summary_stats import summary_statistics_func
import xspec
import torch
import torch.nn as nn
import cmasher as cmr
from sbi.inference import NPE
from sbi.utils import BoxUniform
from sbi.utils import RestrictedPrior, get_density_thresholder
import matplotlib.pyplot as plt
from .xspec_stuff import parallel_folding, transform_parameters_for_xspec
from tqdm.auto import tqdm
from matplotlib.lines import Line2D
from .scaterring_stuff import compress_with_wst


class XSilence(object):
	"""Context for temporarily making xspec quiet."""

	def __enter__(self):
		self.oldchatter = Xset.chatter, Xset.logChatter
		Xset.chatter, Xset.logChatter = 0, 0

	def __exit__(self, *args):
		Xset.chatter, Xset.logChatter = self.oldchatter


def create_prior_function(transformations):
	"""
	Create a single prior transformation function from a list of
	transformations for each parameter. This assumes the priors factorize.
	"""

	def prior(cube):
		params = cube.copy()
		for i, t in enumerate(transformations):
			transform = t['transform']
			params[i] = transform(cube[i])
		return params

	return prior


def store_chain(chainfilename, transformations, posterior, fit_statistic):
	"""Writes a MCMC chain file in the same format as the Xspec chain command."""
	import astropy.io.fits as pyfits

	group_index = 1
	old_model = transformations[0]['model']
	names = []
	for t in transformations:
		if t['model'] != old_model:
			group_index += 1
		old_model = t['model']
		names.append('%s__%d' % (t['name'], t['index'] + (group_index - 1) * old_model.nParameters))

	columns = [pyfits.Column(
		name=name, format='D', array=t['aftertransform'](posterior[:, i]))
		for i, name in enumerate(names)]
	columns.append(pyfits.Column(name='FIT_STATISTIC', format='D', array=fit_statistic))
	table = pyfits.ColDefs(columns)
	header = pyfits.Header()
	header.add_comment("""Created with B6A""")
	header.add_comment("""Based by BXA (Bayesian X-ray spectal Analysis) for Xspec""")
	header.add_comment("""refer to https://github.com/JohannesBuchner/""")
	header['TEMPR001'] = 1.
	header['STROW001'] = 1
	header['EXTNAME'] = 'CHAIN'
	tbhdu = pyfits.BinTableHDU.from_columns(table, header=header)
	tbhdu.writeto(chainfilename, overwrite=True)


def set_parameters(transformations, values):
	"""Set current parameters."""
	assert len(values) == len(transformations)
	pars = []
	for i, t in enumerate(transformations):
		v = float(t['aftertransform'](values[i]))
		assert not isnan(v) and not isinf(v), 'ERROR: parameter %d (index %d, %s) to be set to %f' % (
			i, t['index'], t['name'], v)
		pars += [t['model'], {t['index']:v}]
	AllModels.setPars(*pars)


class SIXSASolver(object):

	allowed_stats = ['cstat', 'pstat']

	def __init__(
			self, transformations, prior_function=None, outputfiles_basename='chains/', use_background=False
	):

		if prior_function is None:
			prior_function = create_prior_function(transformations)

		self.prior_function = prior_function
		self.transformations = transformations
		self.set_paramnames()
		self.background_to_compute = use_background
		self.epoch_trained = []
		self.training_loss = []
		self.validation_loss = []
		self.embedding = None
		self.embedding_net = nn.Identity()

		# for convenience. Has to be a directory anyway for ultranest
		if not outputfiles_basename.endswith('/'):
			outputfiles_basename = outputfiles_basename + '/'

		if not os.path.exists(outputfiles_basename):
			os.mkdir(outputfiles_basename)

		if self.background_to_compute:
			self._background = (np.asarray(xspec.AllData(1).background.values) * xspec.AllData(1).background.exposure).astype(int)
			self._backratio = np.asarray(xspec.AllData(1).exposure / xspec.AllData(1).background.exposure)
		self.outputfiles_basename = outputfiles_basename

	def set_paramnames(self, paramnames=None):
		if paramnames is None:
			self.paramnames = [str(t['name']) for t in self.transformations]
		else:
			self.paramnames = paramnames

	def set_best_fit(self):
		"""Sets model to the best fit values."""

		posterior_func = self.fitted_posteriors[-1]
		x0 = self.posterior_unit_cube[:, posterior_func.log_prob(self.posterior_unit_cube.T).argmax()]
		result = minimize(lambda p: -posterior_func.log_prob(p, track_gradients=True), x0, method='newton-exact')
		best_fit = result.x
		params = self.prior_function(best_fit.numpy())
		set_parameters(transformations=self.transformations, values=params)

	def log_likelihood(self, params):
		""" returns -0.5 of the fit statistic."""
		set_parameters(transformations=self.transformations, values=params)
		like = -0.5 * Fit.statistic
		# print("like = %.1f" % like)
		if not numpy.isfinite(like):
			return -1e100
		return like

	def unit_cube_to_xspec(self, theta):

		parameters_bxa = self.prior_function(theta)

		parameters_xspec = np.apply_along_axis(
			lambda p: transform_parameters_for_xspec(self.transformations, p),
			0,
			parameters_bxa)

		return parameters_xspec

	def run(
			self,
			num_rounds=5,
			num_simulations=5_000,
			embedding='bd2025_summary_statistics',
			npe_kwargs=None,
			training_kwargs=None,
			prune_summaries=False,
			device="cpu"
	):

		if Fit.statMethod.lower() not in SIXSASolver.allowed_stats:
			raise RuntimeError(
				'ERROR: not using cstat or pstat! set Fit.statMethod to cash before analysing (currently: %s)!' % Fit.statMethod)

		if training_kwargs is None:
			training_kwargs = {}
		if npe_kwargs is None:
			npe_kwargs = {}

		num_parameters = len(self.transformations)
		posteriors = []

		# Gather spectrum insights
		observed_spectrum = np.asarray(xspec.AllData(1).values) * xspec.AllData(1).exposure
		energy_low_observation, energy_high_observation = np.asarray(xspec.AllData(1).energies).T

		if embedding is None or isinstance(embedding, nn.Module):

			self.embedding = "none"

			if embedding is not None:
				self.embedding_net = embedding.to(device=device)

			def embedding(x):
				return x, None
			embedding_list = [embedding]*num_rounds

		elif embedding == 'bd2025_summary_statistics':

			self.embedding = "callable"
			embedding_list = [lambda x: summary_statistics_func(
				x,
				energy_low_observation=energy_low_observation,
				energy_high_observation=energy_high_observation
			)]*num_rounds

		elif embedding == "bd2025_wst":

			self.embedding = "callable"
			embedding_list = [compress_with_wst]*num_rounds

		elif isinstance(embedding, list):
			self.embedding = "callable"
			embedding_list = embedding

		else:
			self.embedding = "callable"
			embedding_list = [embedding]*num_rounds

		if not isinstance(training_kwargs, list):
			training_kwargs = [training_kwargs]*num_rounds

		prior = BoxUniform(
			low=torch.zeros(num_parameters).to(device=device),
			high=torch.ones(num_parameters).to(device=device),
		)

		proposal = prior

		self.density_estimator_build_fun = posterior_nn(
			model="maf",
			hidden_features=100,
			num_transforms=10,
			embedding_net=self.embedding_net #if self.embedding_net is not None else None
		)

		inference = NPE(prior=prior, density_estimator=self.density_estimator_build_fun, device=device, **npe_kwargs)

		cc = ChainConsumer()
		colors = self.round_colors(num_rounds)

		for rounds in range(num_rounds):

			posterior, proposal, inference = self.perform_inference_round(
				prior,
				proposal,
				inference,
				num_simulations,
				embedding=embedding_list[rounds],
				observation=observed_spectrum,
				round_number=rounds + 1,
				is_last_round=rounds == num_rounds - 1,
				training_kwargs=training_kwargs[rounds],
				device=device,
				prune_summaries=prune_summaries
			)

			round_samples = self.unit_cube_to_xspec(posterior.sample((10000,)).numpy().T)

			chain = self.chain_from_sample(round_samples, name=f"Round {rounds + 1}", color=colors[rounds])
			cc.add_chain(chain)


			posteriors.append(posterior)

		self.fitted_posteriors = posteriors

		posterior_unit_cube = posteriors[-1].sample((1000,)).numpy().T
		posterior_bxa = self.prior_function(posterior_unit_cube)
		posterior_xspec = self.unit_cube_to_xspec(posterior_unit_cube)
		posterior_stat = self.simulate(posterior_xspec, return_stat=True, desc="Computing posterior statistic - ")

		self.posterior_unit_cube =posterior_unit_cube
		self.inference = inference
		self.posterior = posterior_bxa.T

		chainfilename = '%schain.fits' % self.outputfiles_basename
		store_chain(chainfilename, self.transformations, self.posterior, posterior_stat)
		xspec.AllChains.clear()
		xspec.AllChains += chainfilename
		# set current parameters to best fit
		self.set_best_fit()

		# plot stuff

		cc.plotter.plot(filename=f"{self.outputfiles_basename}posterior_per_round.pdf")
		plt.close('all')

		self.plot_training_summary(filename=f"{self.outputfiles_basename}training_summary.pdf")

		return posteriors[-1] #self.results

	def create_flux_chain(self, spectrum, erange="2.0 10.0", nsamples=None):
		"""
		For each posterior sample, computes the flux in the given energy range.

		The so-created chain can be combined with redshift information to propagate
		the uncertainty. This is especially important if redshift is a variable
		parameter in the fit (with some prior).

		Returns erg/cm^2 energy flux (first column) and photon flux (second column)
		for each posterior sample.
		
		:param spectrum: spectrum to use for spectrum.flux
		:param erange: argument to AllModels.calcFlux, energy range
		:param nsamples: number of samples to consider (the default, None, means all)
		"""
		# prefix = analyzer.outputfiles_basename
		# modelnames = set([t['model'].name for t in transformations])

		with XSilence():
			# plot models
			flux = []
			for k, row in enumerate(tqdm(self.posterior[:nsamples], disable=None)):
				set_parameters(values=row, transformations=self.transformations)
				AllModels.calcFlux(erange)
				f = spectrum.flux
				# compute flux in current energies
				flux.append([f[0], f[3]])

			return numpy.array(flux)

	def posterior_predictions_convolved(
			self, component_names=None, plot_args=None, nsamples=400, plottype='counts'
	):
		"""Plot convolved model posterior predictions.

		Also returns data points for plotting.

		:param component_names: labels to use. Set to 'ignore' to skip plotting a component
		:param plot_args: matplotlib.pyplot.plot arguments for each component
		:param nsamples: number of posterior samples to use (lower is faster)
		"""
		# get data, binned to 10 counts
		# overplot models
		# can we do this component-wise?
		data = [None]  # bin, bin width, data and data error
		models = []  #
		if component_names is None:
			component_names = ['convolved model'] + ['component%d' for i in range(100 - 1)]
		if plot_args is None:
			plot_args = [{}] * 100
			for i, c in enumerate(plt.rcParams['axes.prop_cycle'].by_key()['color']):
				plot_args[i] = dict(color=c)
				del i, c
		bands = []
		Plot.background = True
		Plot.add = True

		for content in self.posterior_predictions_plot(plottype=plottype, nsamples=nsamples):
			xmid = content[:, 0]
			ndata_columns = 6 if Plot.background else 4
			ncomponents = content.shape[1] - ndata_columns
			if data[0] is None:
				data[0] = content[:, 0:ndata_columns]
			model_contributions = []
			for component in range(ncomponents):
				y = content[:, ndata_columns + component]
				if component >= len(bands):
					bands.append(PredictionBand(xmid))
				bands[component].add(y)

				model_contributions.append(y)
			models.append(model_contributions)

		for band, label, component_plot_args in zip(bands, component_names, plot_args):
			if label == 'ignore': continue
			lineargs = dict(drawstyle='steps', color='k')
			lineargs.update(component_plot_args)
			shadeargs = dict(color=lineargs['color'])
			band.shade(alpha=0.5, **shadeargs)
			band.shade(q=0.495, alpha=0.1, **shadeargs)
			band.line(label=label, **lineargs)

		if Plot.background:
			results = dict(list(zip('bins,width,data,error,background,backgrounderr'.split(','), data[0].transpose())))
		else:
			results = dict(list(zip('bins,width,data,error'.split(','), data[0].transpose())))
		results['models'] = numpy.array(models)
		return results

	def posterior_predictions_unconvolved(
			self, component_names=None, plot_args=None, nsamples=400,
			plottype='model',
	):
		"""
		Plot unconvolved model posterior predictions.

		:param component_names: labels to use. Set to 'ignore' to skip plotting a component
		:param plot_args: list of matplotlib.pyplot.plot arguments for each component, e.g. [dict(color='r'), dict(color='g'), dict(color='b')]
		:param nsamples: number of posterior samples to use (lower is faster)
		:param plottype: type of plot string, passed to `xspec.Plot()`
		"""
		if component_names is None:
			component_names = ['model'] + ['component%d' for i in range(100 - 1)]
		if plot_args is None:
			plot_args = [{}] * 100
			for i, c in enumerate(plt.rcParams['axes.prop_cycle'].by_key()['color']):
				plot_args[i] = dict(color=c)
				del i, c
		Plot.add = True
		bands = []

		for content in self.posterior_predictions_plot(plottype=plottype, nsamples=nsamples):
			xmid = content[:, 0]
			ncomponents = content.shape[1] - 2
			for component in range(ncomponents):
				y = content[:, 2 + component]

				if component >= len(bands):
					bands.append(PredictionBand(xmid))
				bands[component].add(y)

		for band, label, component_plot_args in zip(bands, component_names, plot_args):
			if label == 'ignore': continue
			lineargs = dict(drawstyle='steps', color='k')
			lineargs.update(component_plot_args)
			shadeargs = dict(color=lineargs['color'])
			band.shade(alpha=0.5, **shadeargs)
			band.shade(q=0.495, alpha=0.1, **shadeargs)
			band.line(label=label, **lineargs)

	def posterior_predictions_plot(self, plottype, nsamples=None):
		"""
		Internal Routine used by posterior_predictions_unconvolved, posterior_predictions_convolved
		"""
		# for plotting, we don't need so many points, and especially the
		# points that barely made it into the analysis are not that interesting.
		# so pick a random subset of at least nsamples points
		posterior = self.posterior[:nsamples]

		with XSilence():
			olddevice = Plot.device
			Plot.device = '/null'

			# plot models
			maxncomp = 100 if Plot.add else 0
			for k, row in enumerate(tqdm(posterior, disable=None)):
				set_parameters(values=row, transformations=self.transformations)
				Plot(plottype)
				# get plot data
				if plottype == 'model':
					base_content = numpy.transpose([
						Plot.x(), Plot.xErr(), Plot.model()])
				elif Plot.background:
					base_content = numpy.transpose([
						Plot.x(), Plot.xErr(), Plot.y(), Plot.yErr(),
						Plot.backgroundVals(), numpy.zeros_like(Plot.backgroundVals()),
						Plot.model()])
				else:
					base_content = numpy.transpose([
						Plot.x(), Plot.xErr(), Plot.y(), Plot.yErr(),
						Plot.model()])
				# get additive components, if there are any
				comp = []
				for i in range(1, maxncomp):
					try:
						comp.append(Plot.addComp(i))
					except Exception:
						print(
							'The error "***XSPEC Error: Requested array does not exist for this plot." can be ignored.')
						maxncomp = i
						break
				content = numpy.hstack((base_content, numpy.transpose(comp).reshape((len(base_content), -1))))
				yield content
			Plot.device = olddevice


	@property
	def parameter_names_uniques(self):
		"""
		Return a list of unique parameter names, with component names appended if necessary.
		The list is ordered as the parameters appear in the XSPEC model.
		"""

		xspec_model = xspec.AllModels(1)

		def rename_parameters(params, comps):
			from collections import Counter

			# Count how many times each parameter name appears
			counts = Counter(params)

			new_names = []
			back_num = 0
			for param, comp in zip(params, comps):
				if counts[param] > 1:
					# If this parameter appears more than once,
					# append the first two letters of its component.
					back_num += 1
					new_names.append(f"{param} ({comp[:2]}_{back_num})")
				else:
					# If it appears only once, keep it as-is
					new_names.append(param)
			return new_names

		parameter_names = []
		component_names = []
		parameter_index = []

		for component in xspec_model.componentNames:
			for parameter in getattr(xspec_model, component).parameterNames:
				parameter_names.append(parameter)
				component_names.append(component)
				parameter_index.append(getattr(getattr(xspec_model, component), parameter).index - 1)

		parameter_names_vanilla = list(np.asarray(parameter_names)[parameter_index])
		component_names = list(np.asarray(component_names)[parameter_index])
		return rename_parameters(parameter_names_vanilla, component_names)

	def simulate(self, parameters_xspec, **kwargs):

		if (not self.background_to_compute) or (kwargs.get("return_stat", False)):
			return parallel_folding(parameters_xspec, **kwargs)

		if self.background_to_compute:

			#background = np.random.negative_binomial(
			#		np.repeat(self._background[None, :], len(parameters_xspec), axis=0) + 1,
			#	1 / 2
			#) * self._backratio

			background = np.random.poisson(
				np.repeat(self._background[None, :], len(parameters_xspec), axis=0)
			) * self._backratio

			spectra = parallel_folding(parameters_xspec, **kwargs)

			return spectra + background

	def perform_inference_round(
			self,
			prior,
			proposal: RestrictedPrior|Any,
			inference,
			num_simulations,
			round_number=None,
			embedding=None,
			observation=None,
			is_last_round=False,
			training_kwargs=None,
			prune_summaries=False,
			device="cpu"
	):

		theta = proposal.sample((num_simulations,)).cpu().numpy().T

		x_o = torch.from_numpy(np.squeeze(embedding(observation)[0]).astype(np.float32)).to(device)

		parameters_xspec = self.unit_cube_to_xspec(theta)
		all_simulations = self.simulate(parameters_xspec, desc=f"Round {round_number} - " if round_number is not None else "")

		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			features, feature_names = embedding(all_simulations)

		theta_train = torch.from_numpy(theta.T.astype(np.float32)).to(device)
		x_train = torch.from_numpy(features.astype(np.float32)).to(device)

		if prune_summaries:
			low = torch.quantile(x_train, 0.05, dim=0)
			high = torch.quantile(x_train, 0.95, dim=0)

			mask = (x_o > low) & (x_o < high)

			x_train = x_train[:, mask]
			x_o = x_o[mask]
			feature_names = np.asarray(feature_names)[mask]

		if training_kwargs.get("retrain_from_scratch", True) or prune_summaries:
			inference = NPE(prior=prior, density_estimator=self.density_estimator_build_fun)

		density_estimator = inference.append_simulations(
			theta_train, x_train, proposal=proposal
		).train(**training_kwargs)

		posterior = inference.build_posterior(
			density_estimator,
		)

		if x_o is not None:
			posterior = posterior.set_default_x(x_o)

		original_sample = posterior.sample
		# This is just to avoid annoying repeated progress bar
		def sample(*args, **kwargs):
			kwargs["show_progress_bars"] = kwargs.get("show_progress_bars", False)
			return original_sample(*args, **kwargs)
		posterior.sample = sample

		if not is_last_round:
			accept_reject_fn = get_density_thresholder(
				posterior,
				num_samples_to_estimate_support=100_000,
				quantile=1e-4
			)

			proposal = RestrictedPrior(
				prior,
				accept_reject_fn,
				posterior=posterior,
				sample_with="sir",
				device=device
			)

		else:
			proposal = None

		#print(inference.summary)
		num_epochs = inference.summary["epochs_trained"][-1]

		self.epoch_trained.append(num_epochs)
		self.training_loss.extend(inference.summary["training_loss"][-num_epochs:])
		self.validation_loss.extend(inference.summary["validation_loss"][-num_epochs:])

		# Plot summary stat stuff
		if self.embedding == "callable":
			cols = int(np.ceil(np.sqrt(len(feature_names))))
			rows = int(np.ceil(len(feature_names) / cols))

			fig, axes = plt.subplots(
				rows,
				cols,
				figsize=(cols * 4, rows * 4)
			)

			axes = axes.flatten()

			for i, feature_name in enumerate(feature_names):
				axes[i].hist(x_train[:, i], bins=30, color='skyblue', edgecolor='black', log=True)
				axes[i].axvline(x_o[i].numpy(), color='red', linestyle='dashed', linewidth=2)
				axes[i].set_title(feature_name)

			plt.suptitle(f"Summary stats - Round {round_number}")
			plt.tight_layout()
			plt.savefig(
				f"{self.outputfiles_basename}features_round_{round_number}.pdf",
					bbox_inches="tight"
			)
			plt.close()

		return posterior, proposal, inference

	def get_xspec_best_fit(self):
		"""
		Perform a fit with Xspec and yield best fit value and covariance matrix.
		"""

		with XSilence():
			xspec.Fit.perform()

		def build_covariance_matrix_np(covar_elements):
			covar_elements = np.asarray(covar_elements, dtype=float)

			M = len(covar_elements)
			N = int((np.sqrt(1 + 8 * M) - 1) // 2)

			cov_matrix = np.zeros((N, N), dtype=float)
			i, j = np.tril_indices(N)
			cov_matrix[i, j] = covar_elements
			cov_matrix[j, i] = covar_elements

			return cov_matrix

		xspec_model = xspec.AllModels(1)
		best_fit_parameters = np.asarray([xspec_model(i + 1).values[0] for i in range(xspec_model.nParameters)])
		covariance = build_covariance_matrix_np(xspec.Fit.covariance)

		return best_fit_parameters.ravel(), covariance

	def chain_from_sample(self, samples, **kwargs):

		parameter_names = self.parameter_names_uniques
		param_dict = {}

		for i, t in enumerate(self.transformations):

			j = t["index"]
			name = parameter_names[j-1]
			param_dict[name] = [sample[j] for sample in samples] # maybe change to i

		return Chain(samples=pd.DataFrame.from_dict(param_dict), **kwargs)

	def round_colors(self, num_rounds):

		return cmr.take_cmap_colors(cmr.cosmic_r, num_rounds, cmap_range=(0.1, 0.8))

	def plot_training_summary(self, figsize=(10, 7), filename=None):

		plt.figure(figsize=figsize)

		prev_num = 0
		colors = self.round_colors(len(self.epoch_trained))

		for round, num_epoch in enumerate(self.epoch_trained):
			steps = np.arange(prev_num + 1, prev_num + 1 + num_epoch)

			plt.plot(steps,
					 self.training_loss[steps.min() - 1:steps.max()],
					 color=colors[round],
					 linestyle='dotted'
					 )

			plt.plot(steps,
					 self.validation_loss[steps.min() - 1:steps.max()],
					 color=colors[round]
					 )

			plt.axvline(prev_num + num_epoch, color="black", linestyle="dotted", alpha=0.3)
			prev_num += num_epoch

		plt.xlabel("Epoch")
		plt.ylabel("Validation loss")

		custom_lines = [
			Line2D([0], [0], color="black", linestyle="dotted"),
			Line2D([0], [0], color="black")
		]

		plt.legend(custom_lines, ['Training loss', 'Validation loss'])
		plt.savefig(filename)
		plt.close()

