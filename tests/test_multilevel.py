import os
import pickle
import unittest
import tempfile

import formulaic
import numpy
import pandas
import rpy2.robjects
import rpy2.robjects.pandas2ri

import slimp

from test_model import TestModel

class TestMultiLevel(TestModel):
    @classmethod
    def setUpClass(cls):
        cls.data = pandas.read_csv(
            os.path.join(os.path.dirname(__file__), "data", "sleepstudy.csv"))
        cls.data["Days"] = cls.data["Days"].astype(float)
        cls.data["Subject"] = cls.data["Subject"].astype("category")

        cls.formula = ["Reaction ~ 1+Days", ("Subject", "1+Days")]
        
        cls.outcomes, unmodeled_predictors = formulaic.model_matrix(
            cls.formula[0], cls.data)
        
        modeled_predictors = formulaic.model_matrix(cls.formula[1][1], cls.data)
        modeled_predictors.index = cls.data[cls.formula[1][0]]
        cls.predictors = [unmodeled_predictors, modeled_predictors]
        
        with rpy2.robjects.pandas2ri.converter.context():
            data_r = rpy2.robjects.conversion.get_conversion().py2rpy(cls.data)
        
        rpy2.robjects.r["library"]("lme4")
        model = rpy2.robjects.r["lmer"](
            f"{cls.formula[0]} + ({cls.formula[1][1]} | {cls.formula[1][0]})",
            data_r)
        summary = rpy2.robjects.r["summary"](model)
        
        mapper = {"(Intercept)": "Intercept", "Days": "Days"}
        
        cls.parameters= {
            **{
                mapper[k]: v for k, v in zip(
                    summary.rx2("coefficients").names[0],
                    numpy.array(summary.rx2["coefficients"])[:, 0])},
            "sigma_y": summary.rx2("sigma")[0]}
        
        ranef = rpy2.robjects.r["ranef"](model)[0]
        rows, columns = [list(rpy2.robjects.r[f"{x}names"](ranef)) for x in ["row", "col"]]
        for name, column in zip(columns, numpy.array(ranef)):
            cls.parameters.update({
                f"{cls.formula[1][0]}[{k}]/{mapper[name]}": v
                for k, v in zip(rows, column)})
        
        cls.predicted = numpy.ravel(rpy2.robjects.r["predict"](model))
    
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
        self._test_draws(model, 0.3)
        self._test_posterior_epred(model, 0.4)
        self._test_posterior_predict(model, 0.4)

if __name__ == "__main__":
    unittest.main()
