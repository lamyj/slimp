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
        self._formula = formula
        self._data = data
        
        self._update_model_data(formula, data)
        self._update_programs()
        self._update_draws()
        
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
    def posterior_epred(self):
        return self._draws.filter(like="mu_rep")
    
    @property
    def posterior_predict(self):
        return self._draws.filter(like="y_rep")
    
    @property
    def log_likelihood(self):
        return self._draws.filter(like="log_likelihood")
    
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
        
        self._fit = self._model.sample(self._fit_data, **kwargs)
        
        self._draws = self._fit.draws_pd()
        
        self._predictors_columns = []
        for index, column in enumerate(self._draws.columns):
            if not column.split("[")[0].endswith("_"):
                self._predictors_columns.append(index)
        
        self._diagnostics = self._fit.draws_pd().filter(regex="_$")
        self._draws = self._fit.draws_pd().filter(regex="[^_]$")
        self._draws.columns = self._predictor_mapper(self._draws.columns)
    
    def summary(self, percentiles=(5, 50, 95)):
        summary = self._fit.summary(percentiles, sig_figs=18)
        summary = summary.iloc[[not x.split("[")[0].endswith("_") for x in summary.index], :]
        summary.index = self._predictor_mapper(summary.index)
        return summary
    
    def predict(self, data, use_prior=False, **kwargs):
        data = data.astype(
            {k: v for k, v in self._data.dtypes.items() if k in data.columns})
        predictors = pandas.DataFrame(
            formulaic.model_matrix(self.formula.split("~")[1], data))
        fit_data = self._fit_data | {
            "N_new": predictors.shape[0], "X_new": predictors.values.tolist(),
            "use_prior": int(use_prior)}
        
        fit = self._model.generate_quantities(fit_data, self._fit, **kwargs)
        draws = fit.draws_pd()
        
        predictor_mapper = PredictorMapper(predictors)
        draws.columns = predictor_mapper(draws.columns)
        return draws.filter(like="mu_rep"), draws.filter(like="y_rep")
    
    def _update_model_data(self, formula, data):
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
        self._model = cmdstanpy.CmdStanModel(
            exe_file=os.path.join(os.path.dirname(__file__), "univariate"))
        if chains is not None:
            directory = pathlib.Path(tempfile.mkdtemp())
            chains = []
            for path, chain in state["chains"].items():
                with (directory/path).open("w") as fd:
                    fd.write(chain)
                chains.append(str(directory/path))
            self._fit = cmdstanpy.from_csv(chains)
        else:
            self._fit = None
    
    def _update_draws(self):
        self._diagnostics = None
        self._draws = None
        if self._fit is not None:
            self._diagnostics = self._fit.draws_pd().filter(regex="_$")
            self._draws = self._fit.draws_pd().filter(regex="[^_]$")
            self._draws.columns = self._predictor_mapper(self._draws.columns)
    
    def __getstate__(self):
        chains = None
        if self._fit is not None:
            with tempfile.TemporaryDirectory() as directory:
            # NOTE: need to keep this directory after __getstate__
                directory = pathlib.Path(tempfile.mkdtemp())
                self._fit.save_csvfiles(directory)
                chains = {}
                for chain in directory.glob("*csv"):
                    chains[chain.name] = chain.open().read()
        
        return {
            "formula": self._formula, "data": self._data,
            **({"chains": chains} if chains else {})
        }
    
    def __setstate__(self, state):
        self._update_model_data(state["formula"], state["data"])
        self._update_programs(state.get("chains"))
        self._update_draws()

