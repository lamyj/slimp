import os
import pickle
import unittest
import tempfile

import formulaic
import numpy
import pandas

import slimp

class TestUnivariate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = numpy.mgrid[0:10, 0:10]
        numpy.random.seed(42)
        z = 10 + x + 2*y + numpy.random.normal(0, 2, (10, 10))
        cls.data = pandas.DataFrame({
            "x": x.ravel(), "y": y.ravel(), "z": z.ravel()})

        cls.formula = "z ~ 1 + x + y"
        cls.outcomes, cls.predictors = [
            pandas.DataFrame(a)
            for a in formulaic.model_matrix(cls.formula, cls.data)]
    
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
            
            self._test_model_data(model)
            self._test_sampler_parameters(model)
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
            
            self._test_model_data(model)
            self._test_sampler_parameters(model)
            self._test_model_diagnostics(model)
            self._test_model_draws(model)
    
    def _test_model_data(self, model):
        self.assertEqual(self.formula, model.formula)
        self.assertTrue(self.data.equals(model.data))
        
        self.assertTrue(self.predictors.equals(model.predictors))
        self.assertTrue(self.outcomes.equals(model.outcomes))
    
    def _test_sampler_parameters(self, model):
        self.assertEqual(model.sampler_parameters.seed, 42)
        self.assertEqual(model.sampler_parameters.num_chains, 4)
        self.assertEqual(model.sampler_parameters.num_samples, 1000)
    
    def _test_model_diagnostics(self, model):
        numpy.testing.assert_equal(
            model.hmc_diagnostics.max().values[:2], [0, 0])
        numpy.testing.assert_allclose(
            model.hmc_diagnostics["e_bfmi"].values,
            [1.07163565, 1.03752015, 1.04480104, 1.10317826])
        numpy.testing.assert_allclose(
            model.summary()["R_hat"].values,
            [
                1.00023455, 1.00010931, 1.00056218, 1.00085384, 0.99970619,
                1.00022703])
        numpy.testing.assert_allclose(
            model.summary()["N_Eff"].values,
            [
                3536.70781323, 4276.99039814, 3699.20715418, 3480.21091919,
                3902.67680926, 3684.75594434])
    
    def _test_model_draws(self, model):
        self.assertEqual(
            list(model.draws.columns),
            ["Intercept_c", "x", "y", "sigma", "Intercept"])
        numpy.testing.assert_allclose(
            model.draws.describe().values,
            [
                [4000, 4000, 4000, 4000, 4000],
                [
                    2.32920494e+01, 9.07393946e-01, 2.03522925e+00,
                    1.83309822e+00, 1.00502450e+01],
                [
                    1.82854186e-01, 6.24573568e-02, 6.38350738e-02,
                    1.32399923e-01, 4.51111133e-01],
                [
                    2.25925673e+01, 6.84856120e-01, 1.81185481e+00,
                    1.44426085e+00, 8.46896348e+00],
                [
                    2.31733098e+01, 8.66278326e-01, 1.99290926e+00,
                    1.73854068e+00, 9.75405391e+00],
                [
                    2.32906485e+01, 9.07013313e-01, 2.03545372e+00,
                    1.82685314e+00, 1.00437696e+01],
                [
                    2.34117638e+01, 9.48450080e-01, 2.07808404e+00,
                    1.91598202e+00, 1.03500511e+01],
                [
                    2.40042409e+01, 1.11888080e+00, 2.27348756e+00,
                    2.43737262e+00, 1.17039339e+01]])

if __name__ == "__main__":
    unittest.main()
