import os
import pathlib
import shutil
import tempfile

import cmdstanpy
import formulaic
import numpy
import pandas

from .predictor_mapper import PredictorMapper   

class Model:
    def __init__(self, formula, data):
        self._update_model_data(formula, data)
        self._update_programs()
        self._update_draws()
        self._update_generated_quantities()
    
    def __del__(self):
        if getattr(self, "_fit") is not None:
            directory = os.path.dirname(self._fit.runset.csv_files[0])
            if os.path.isdir(directory):
                shutil.rmtree(directory)
    
    @property
    def formula(self):
        return self._formula
    
    @property
    def data(self):
        return self._data
    
    @property
    def predictors(self):
        return self._predictors
    
    @property
    def outcomes(self):
        return self._outcomes
    
    @property
    def fit_data(self):
        return self.fit_data
    
    @property
    def draws(self):
        return self._draws
    
    @property
    def prior_predict(self):
        if self._y_prior is None:
            fit = self._programs["predict_prior"].generate_quantities(
                self._fit_data | {
                    "N_new": self._fit_data["N"], "X_new": self._fit_data["X"]},
                self._fit)
            draws = fit.draws_pd()
            self._y_prior = draws.filter(like="y")
        return self._y_prior
    
    @property
    def posterior_epred(self):
        if self._mu_posterior is None:
            fit = self._programs["predict_posterior"].generate_quantities(
                self._fit_data | {
                    "N_new": self._fit_data["N"], "X_new": self._fit_data["X"]},
                self._fit)
            draws = fit.draws_pd()
            self._mu_posterior = draws.filter(like="mu")
            self._y_posterior = draws.filter(like="y")
        return self._mu_posterior
    
    @property
    def posterior_predict(self):
        if self._y_posterior is None:
            fit = self._programs["predict_posterior"].generate_quantities(
                self._fit_data | {
                    "N_new": self._fit_data["N"], "X_new": self._fit_data["X"]},
                self._fit)
            draws = fit.draws_pd()
            self._mu_posterior = draws.filter(like="mu")
            self._y_posterior = draws.filter(like="y")
        return self._y_posterior
    
    @property
    def log_likelihood(self):
        if self._log_likelihood is None:
            fit = self._programs["log_likelihood"].generate_quantities(
                self._fit_data, self._fit)
            self._log_likelihood = fit.draws_pd().filter(like="log_likelihood")
        return self._log_likelihood
    
    @property
    def hmc_diagnostics(self):
        max_depth = self._fit.metadata.cmdstan_config["max_depth"]
        data = (
            self._diagnostics.groupby("chain__")
            .agg(
                divergent=("divergent__", lambda x: numpy.sum(x!=0)),
                depth_exceeded=(
                    "treedepth__", lambda x: numpy.sum(x >= max_depth)),
                e_bfmi=(
                    "energy__", 
                    lambda x: (
                        numpy.sum(numpy.diff(x)**2)
                        / numpy.sum((x-numpy.mean(x))**2)))))
        data.index = data.index.rename("chain").astype(int)
        return data
    
    def sample(self, **kwargs):
        # NOTE: this directory must remain during the lifetime of the object
        directory = tempfile.mkdtemp()
        kwargs["output_dir"] = directory
        kwargs["sig_figs"] = 18
        
        self._fit = self._programs["sampler"].sample(self._fit_data, **kwargs)
        self._update_draws()
        
        self._log_likelihood = None
        self._mu_posterior = None
        self._y_posterior = None
    
    def summary(self, percentiles=(5, 50, 95)):
        summary = self._fit.summary(percentiles, sig_figs=18)
        summary = summary.iloc[[not x.split("[")[0].endswith("_") for x in summary.index], :]
        summary.index = self._predictor_mapper(summary.index)
        return summary
    
    def predict(self, data, **kwargs):
        data = data.astype(
            {k: v for k, v in self._data.dtypes.items() if k in data.columns})
        predictors = pandas.DataFrame(
            formulaic.model_matrix(self.formula.split("~")[1], data))
        fit_data = self._fit_data | {
            "N_new": predictors.shape[0], "X_new": predictors.values.tolist()}
        
        fit = self._programs["predict_posterior"].generate_quantities(
            fit_data, self._fit, **kwargs)
        draws = fit.draws_pd()
        
        return draws.filter(like="mu"), draws.filter(like="y")
    
    def _update_model_data(self, formula, data):
        self._formula = formula
        self._data = data
        
        self._outcomes, self._predictors = [
            pandas.DataFrame(a) for a in formulaic.model_matrix(formula, data)]
        self._predictor_mapper = PredictorMapper(self._predictors)
        outcomes = numpy.array(self._outcomes).squeeze()
        outcomes_mean = numpy.mean(outcomes)
        outcomes_scale = numpy.std(outcomes)
        
        predictors_scale = numpy.std(
            self._predictors.filter(regex="^(?!.*Intercept)"))
        predictors_scale[predictors_scale==0] = 1e-20
        
        self._fit_data = {
            "N": self._predictors.shape[0], "K": self._predictors.shape[1],
            "y": outcomes, "X": self._predictors.values,
            
            "mu_alpha": outcomes_mean, "sigma_alpha": outcomes_scale,
            "sigma_beta": (outcomes_scale/predictors_scale),
            "lambda_sigma": 1/outcomes_scale
        }
    
    def _update_programs(self, chains=None):
        self._programs = {
            x: cmdstanpy.CmdStanModel(
                exe_file=os.path.join(
                    os.path.dirname(__file__), f"univariate_{x}"))
            for x in [
                "sampler", "predict_prior", "predict_posterior",
                "log_likelihood"]
        }
        
        if chains is not None:
            directory = pathlib.Path(tempfile.mkdtemp())
            paths = []
            for path, chain in chains.items():
                with (directory/path).open("w") as fd:
                    fd.write(chain)
                paths.append(str(directory/path))
            self._fit = cmdstanpy.from_csv(paths)
            self._fit._sig_figs = 18
        else:
            self._fit = None
    
    def _update_draws(self):
        self._diagnostics = None
        self._draws = None
        if self._fit is not None:
            self._diagnostics = self._fit.draws_pd().filter(regex="_$")
            self._draws = self._fit.draws_pd().filter(regex="[^_]$")
            self._draws.columns = self._predictor_mapper(self._draws.columns)
    
    def _update_generated_quantities(self, state=None):
        self._log_likelihood = None
        self._y_prior = None
        self._mu_posterior = None
        self._y_posterior = None
        if state is not None:
            members = [
                "log_likelihood", "y_prior", "mu_posterior", "y_posterior"]
            for member in members:
                setattr(self, f"_{member}", state.get(member))
    
    def __getstate__(self):
        chains = None
        if self._fit is not None:
            with tempfile.TemporaryDirectory() as directory:
            # NOTE: need to keep this directory after __getstate__
                directory = pathlib.Path(tempfile.mkdtemp())
                self._fit.save_csvfiles(directory)
                chains = {}
                for chain in directory.glob("*csv"):
                    with open(chain) as fd:
                        chains[chain.name] = fd.read()
        
        members = [
            "log_likelihood", "y_prior", "mu_posterior", "y_posterior"]
        return {
            "formula": self._formula, "data": self._data,
            **({"chains": chains} if chains else {}),
            **{member: getattr(self, f"_{member}") for member in members}
        }
    
    def __setstate__(self, state):
        self._update_model_data(state["formula"], state["data"])
        self._update_programs(state.get("chains"))
        self._update_draws()
        self._update_generated_quantities(state)

