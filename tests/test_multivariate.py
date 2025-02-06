# NOTE: informational messages are logged at the ERROR level
import logging
logging.basicConfig(level=logging.CRITICAL)

import os
import pickle
import tempfile
import unittest

import formulaic
import numpy
import pandas
import rpy2.robjects
import rpy2.robjects.pandas2ri

import slimp

from test_model import TestModel

class TestMultivariate(TestModel):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), "data", "mvreg.csv"))
        
        lhs = ["locus_of_control", "self_concept", "motivation"]
        rhs = "1+read+write+science+prog"
        cls.formula = [f"{x} ~ {rhs}" for x in lhs]
        
        cls.outcomes, cls.predictors = zip(
            *[formulaic.model_matrix(f, cls.data) for f in cls.formula])
        cls.outcomes = pandas.concat(cls.outcomes, axis="columns")
        
        with rpy2.robjects.pandas2ri.converter.context():
            data_r = rpy2.robjects.conversion.get_conversion().py2rpy(cls.data)
        
        rpy2.robjects.r["library"]("systemfit")
        model = rpy2.robjects.r["systemfit"](
            [rpy2.robjects.Formula(x) for x in cls.formula], data=data_r)
        summary = rpy2.robjects.r["summary"](model)
        
        mapper = {}
        for index, name in enumerate(lhs):
            sub_mapper = {
                "(Intercept)": "Intercept", "read": "read", "write": "write",
                "science": "science", "proggeneral": "prog[T.general]",
                "progvocational": "prog[T.vocational]",
                "sigma": "sigma"}
            mapper.update({
                f"eq{1+index}_{from_}": f"{name}/{to}"
                for from_, to in sub_mapper.items()})
        cls.parameters= {
            **{
                mapper[k]: v for k, v in zip(
                    summary.rx2("coefficients").names[0],
                    numpy.array(summary.rx2["coefficients"])[:, 0])},
            **{
                mapper[f"{k}_sigma"]: v for k, v in zip(
                    summary.rx2("residCov").names[0],
                    numpy.diag(summary.rx2("residCov"))**0.5)}
            }
        cls.predicted = numpy.ravel(rpy2.robjects.r["predict"](model))
        cls.r_squared = [x.rx2("adj.r.squared")[0] for x in summary.rx2("eq")]
    
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
        self._test_posterior_epred(model, 0.4)
        self._test_posterior_predict(model, 0.4)
        self._test_r_squared(model, 0.4)

if __name__ == "__main__":
    unittest.main()
