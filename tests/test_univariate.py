import os
import pickle
import tempfile
import unittest

import formulaic
import numpy
import pandas
import rpy2.robjects
import rpy2.robjects.pandas2ri
import scipy

import slimp

from test_model import TestModel

class TestUnivariate(TestModel):
    @classmethod
    def setUpClass(cls):
        ctl = [4.17, 5.58, 5.18, 6.11, 4.50, 4.61, 5.17, 4.53, 5.33, 5.14]
        trt = [4.81, 4.17, 4.41, 3.59, 5.87, 3.83, 6.03, 4.89, 4.32, 4.69]
        cls.data = pandas.DataFrame({
            "weight": [*ctl, *trt],
            "group": [*(len(ctl)*["Ctl"]), *(len(ctl)*["Trt"])]})
        
        cls.formula = "weight ~ 1 + group"
        cls.outcomes, cls.predictors = [
            pandas.DataFrame(a)
            for a in formulaic.model_matrix(cls.formula, cls.data)]
        cls.predictors = (cls.predictors, )
        
        with rpy2.robjects.pandas2ri.converter.context():
            data_r = rpy2.robjects.conversion.get_conversion().py2rpy(cls.data)
        model = rpy2.robjects.r["lm"](cls.formula, data_r)
        summary = rpy2.robjects.r["summary"](model)
        
        mapper = {"(Intercept)": "Intercept", "groupTrt": "group[T.Trt]"}
        
        cls.parameters= {
            **{
                mapper[k]: v for k, v in zip(
                    summary.rx2("coefficients").names[0],
                    numpy.array(summary.rx2["coefficients"])[:, 0])},
            "sigma": summary.rx2("sigma")[0]}
        cls.predicted = numpy.array(rpy2.robjects.r["predict"](model))
        cls.log_likelihood = scipy.stats.norm.logpdf(
            cls.data["weight"], cls.predicted, cls.parameters["sigma"])
        cls.r_squared = summary.rx2("adj.r.squared")[0]
    
    def test_no_sample(self):
        def dump(path):
            model = slimp.Model(self.formula, self.data, seed=42, num_chains=4)
            with open(path, "wb") as fd:
                pickle.dump(model, fd)
        def load(path):
            with open(path, "rb") as fd:
                return pickle.load(fd)
        
        with tempfile.TemporaryDirectory() as dir:
            dump(os.path.join(dir, "model.pkl"))
            model = load(os.path.join(dir, "model.pkl"))
        
        self._test_data(model)
        self._test_sampler_parameters(model, 42, 4, 1000)
        self.assertTrue(model.draws is None)
    
    def test_sample(self):
        def dump(path):
            model = slimp.Model(self.formula, self.data, seed=42, num_chains=4)
            model.sample()
            with open(path, "wb") as fd:
                pickle.dump(model, fd)
        def load(path):
            with open(path, "rb") as fd:
                return pickle.load(fd)
        
        with tempfile.TemporaryDirectory() as dir:
            dump(os.path.join(dir, "model.pkl"))
            model = load(os.path.join(dir, "model.pkl"))
        
        self._test_data(model)
        self._test_sampler_parameters(model, 42, 4, 1000)
        self._test_hmc_diagnostics(model)
        self._test_draws(model, 0.25)
        self._test_log_likelihood(model, 0.4)
        self._test_prior_predict(model, 0.4)
        self._test_posterior_epred(model, 0.4)
        self._test_posterior_predict(model, 0.4)
        self._test_r_squared(model, 0.4)

if __name__ == "__main__":
    unittest.main()
