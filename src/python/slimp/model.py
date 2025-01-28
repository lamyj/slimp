import formulaic
import numpy
import pandas

from . import _slimp, action_parameters, sample_data_as_df, stats
from .misc import sample_data_as_df
from .samples import Samples

from . import multilevel, multivariate, univariate

class Model:
    def __init__(
            self, formula, data, seed=-1, num_chains=1, sampler_parameters=None,
            **kwargs):
        ModelData = None
        if isinstance(formula, str):
            ModelData = univariate.ModelData
        elif isinstance(formula, list):
            if len(formula) == 2 and isinstance(formula[1], tuple):
                ModelData = multilevel.ModelData
            else:
                ModelData = multivariate.ModelData
        self._model_data = ModelData(formula, data)
        self._model_name = ModelData.__module__.split(".")[1]
        
        if sampler_parameters is None:
            self._sampler_parameters = action_parameters.Sample(
                seed=seed, num_chains=num_chains, **kwargs)
        else:
            self._sampler_parameters = sampler_parameters
        
        self._samples = None
        self._generated_quantities = {}
    
    @property
    def formula(self):
        return (
            self._model_data.formula if len(self._model_data.formula)>1
            else self._model_data.formula[0])
    
    @property
    def data(self):
        return self._model_data.data
    
    @property
    def predictors(self):
        return self._model_data.predictors
    
    @property
    def outcomes(self):
        return self._model_data.outcomes
    
    @property
    def fit_data(self):
        return self._model_data.fit_data
    
    @property
    def sampler_parameters(self):
        return self._sampler_parameters
    
    @property
    def draws(self):
        return self._samples.draws if self._samples is not None else None
    
    @property
    def prior_predict(self):
        if "y_prior" not in self._generated_quantities:
            draws = self._generate_quantities("predict_prior")
            self._generated_quantities["y_prior"] = draws.filter(like="y")
        return self._generated_quantities["y_prior"]
    
    @property
    def posterior_epred(self):
        if "mu_posterior" not in self._generated_quantities:
            draws = self._generate_quantities("predict_posterior")
            self._generated_quantities["mu_posterior"] = draws.filter(like="mu")
            self._generated_quantities["y_posterior"] = draws.filter(like="y")
        return self._generated_quantities["mu_posterior"]
    
    @property
    def posterior_predict(self):
        if "y_posterior" not in self._generated_quantities:
            # Update cached data
            self.posterior_epred
        return self._generated_quantities["y_posterior"]
    
    @property
    def log_likelihood(self):
        if "log_likelihood" not in self._generated_quantities:
            draws = self._generate_quantities("log_likelihood")
            self._generated_quantities["log_likelihood"] = draws.filter(like="log_likelihood")
        return self._generated_quantities["log_likelihood"]
    
    @property
    def hmc_diagnostics(self):
        return stats.hmc_diagnostics(
            self._samples.diagnostics, self._sampler_parameters.hmc.max_depth)
    
    def sample(self, sampler=None):
        if sampler is None:
            sampler = getattr(_slimp, f"{self._model_name}_sampler")
        data = sampler(self._model_data.fit_data, self._sampler_parameters)
        self._samples = Samples(
            sample_data_as_df(data),
            self._model_data.predictor_mapper, data["parameters_columns"])
        self._generated_quantities = {}
    
    def summary(self, percentiles=(5, 50, 95)):
        return stats.summary(
            self._samples.samples[["lp__"]].join(self._samples.draws),
            self._sampler_parameters.num_chains,
            percentiles)
    
    def predict(self, data):
        predictors = self._model_data.new_predictors(data)
        draws = self._generate_quantities(
            "predict_posterior", predictors.values)
        return draws.filter(like="mu"), draws.filter(like="y")
    
    def _generate_quantities(self, name, *args, **kwargs):
        new_data = self._model_data.new_data(*args, **kwargs)
        
        # NOTE: must only include model parameters
        draws = self._samples.samples[self._samples.parameters_columns].values.T
        chains = self._sampler_parameters.num_chains
        draws = draws.reshape(-1, chains, draws.shape[1]//chains)
        data = getattr(_slimp, f"{self._model_name}_{name}")(
            new_data, draws, self._sampler_parameters)
        
        return sample_data_as_df(data)
    
    def __getstate__(self):
        return {
            "formula": self.formula, "data": self.data,
            "sampler_parameters": self._sampler_parameters,
            "model_name": self._model_name,
            **(
                {
                    "samples": self._samples.samples,
                    "parameters_columns": self._samples.parameters_columns}
                if self._samples is not None else {}),
            "generated_quantities": self._generated_quantities
        }
    
    def __setstate__(self, state):
        self.__init__(state["formula"], state["data"])
        self._sampler_parameters = state["sampler_parameters"]
        self._model_name = state["model_name"]
        if "samples" in state:
            self._samples = Samples(
                state["samples"], self._model_data.predictor_mapper,
                state["parameters_columns"])
        self._generated_quantities = state["generated_quantities"]
