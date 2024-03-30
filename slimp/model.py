import formulaic
import numpy
import pandas

from . import _slimp, action_parameters
from .model_data import ModelData
from .samples import Samples

class Model:
    def __init__(self, formula, data, seed=-1, num_chains=1):
        self._model_data = ModelData(formula, data)
        
        self._sampler_parameters = action_parameters.Sample()
        self._sampler_parameters.seed = seed
        self._sampler_parameters.num_chains = num_chains
        
        self._model_name = (
            "multivariate" if len(self._model_data.formula)>1
            else "univariate")
        
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
        return (
            self._model_data.predictors if len(self._model_data.formula)>1
            else self._model_data.predictors[0])
    
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
        data = (
            self._samples.diagnostics.groupby("chain__")
            .agg(
                divergent=("divergent__", lambda x: numpy.sum(x!=0)),
                depth_exceeded=(
                    "treedepth__", lambda x: numpy.sum(
                        x >= self._sampler_parameters.hmc.max_depth)),
                e_bfmi=(
                    "energy__", 
                    lambda x: (
                        numpy.sum(numpy.diff(x)**2)
                        / numpy.sum((x-numpy.mean(x))**2)))))
        data.index = data.index.rename("chain").astype(int)
        return data
    
    def sample(self):
        data = _slimp.sample(
            self._model_name, self._model_data.fit_data,
            self._sampler_parameters)
        self._samples = Samples(
            self._get_df(data["array"], data["columns"]),
            self._model_data.predictor_mapper, data["parameters_columns"])
        self._generated_quantities = {}
    
    def summary(self, percentiles=(5, 50, 95)):
        lp = self._samples.samples["lp__"].values
        draws = self._samples.draws.values
        
        summary = numpy.empty((1+draws.shape[1], 5+len(percentiles)))
        
        measures = [
            (numpy.mean, 0, ()),
            (numpy.std, 2, ()),
            (
                lambda *args, **kwargs: numpy.quantile(*args, **kwargs).T,
                slice(3, 3+len(percentiles)), (numpy.array(percentiles)/100, ))]
        
        for f, c, args in measures:
            summary[0, c] = f(lp, *args)
            summary[1:, c] = f(draws, *args, axis=0)
        
        measures = [
            (_slimp.get_effective_sample_size, -2),
            (_slimp.get_potential_scale_reduction, -1)]
        for f, c in measures:
            summary[0, c] = f(lp, self._sampler_parameters.num_chains)
            summary[1:, c] = f(draws, self._sampler_parameters.num_chains)
        
        summary[:, 1] = numpy.sqrt(summary[:, 2])/numpy.sqrt(summary[:, -2])
        
        return pandas.DataFrame(
            summary,
            columns=[
                "Mean", "MCSE", "StdDev",
                *[f"{p}%" for p in percentiles],
                "N_Eff", "R_hat"],
            index=["lp__", *self._samples.draws.columns])
    
    def predict(self, data):
        data = data.astype({
            k: v for k, v in self.data.dtypes.items() if k in data.columns})
        predictors = pandas.DataFrame(
            formulaic.model_matrix(self.formula.split("~")[1], data))
        draws = self._generate_quantities(
            "predict_posterior", predictors.shape[0], predictors.values)
        return draws.filter(like="mu"), draws.filter(like="y")
    
    def _get_df(self, data, names):
        return pandas.DataFrame(
            data.reshape(-1, data.shape[-1], order="A"), columns=names)
    
    def _generate_quantities(self, name, N_new=None, X_new=None):
        if N_new is None:
            N_new = self._model_data.fit_data["N"]
            X_new = self._model_data.fit_data["X"]
        
        parameters = action_parameters.GenerateQuantities()
        parameters.seed = self._sampler_parameters.seed
        parameters.num_chains = self._sampler_parameters.num_chains
        
        data = _slimp.generate_quantities(
            self._model_name, name, 
            self.fit_data | { "N_new": N_new, "X_new": X_new},
            # NOTE: must only include model parameters
            self._samples.samples[self._samples.parameters_columns].values,
            parameters)
        return self._get_df(data["array"], data["columns"])
    
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
