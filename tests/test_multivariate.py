import os
import pickle
import unittest
import tempfile

import formulaic
import numpy
import pandas

import slimp

class TestMultivariate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        y, x = numpy.mgrid[0:10, 0:10]
        numpy.random.seed(42)
        epsilon1 = numpy.random.normal(0, 2, (10, 10))
        epsilon2 = numpy.random.normal(0, 2, (10, 10))
        z1 = 10 + x + 2*y + epsilon1
        z2 = -3 + 5*x - 7*y + epsilon2
        cls.data = pandas.DataFrame({
            "x": x.ravel(), "y": y.ravel(), "z1": z1.ravel(), "z2": z2.ravel()})

        cls.formula = ["z1 ~ 1 + x + y", "z2 ~ 1 + x + y"]
        cls.outcomes, cls.predictors = zip(
            *[formulaic.model_matrix(f, cls.data) for f in cls.formula])
        cls.outcomes = pandas.concat(cls.outcomes, axis="columns")
    
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
            self._test_model_log_likelihood(model)
    
    def _test_model_data(self, model):
        self.assertEqual(self.formula, model.formula)
        self.assertTrue(self.data.equals(model.data))
        
        for p1, p2 in zip(self.predictors, model.predictors):
            self.assertTrue(p1.equals(p2))
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
            [1.09786749, 1.18009938, 1.1609919,  0.99172618])
        numpy.testing.assert_allclose(
            model.summary()["R_hat"].values,
            [
                1.00105561, 0.99960792, 0.99952068, 1.00016764, 0.99968783,
                0.99960895, 0.99962703, 0.99970713, 0.99971569, numpy.nan,
                0.99973984, numpy.nan, 0.99956509, 1.00003308, 0.99960281,
                numpy.nan, 0.99973984, 0.99973984, numpy.nan])
        numpy.testing.assert_allclose(
            model.summary()["N_Eff"].values,
            [
                3395.494419, 3912.278219, 3999.496209, 3984.778412, 3881.130415,
                4059.257418, 3760.436512, 3829.355139, 4189.738843, numpy.nan,
                3753.980293, numpy.nan, 3771.462377, 3882.499809, 4021.782726,
                numpy.nan, 3753.980293, 3753.980293, numpy.nan])
    
    def _test_model_draws(self, model):
        self.assertEqual(
            list(model.draws.columns),
            [
                "z1/Intercept_c", "z2/Intercept_c",
                "z1/x", "z1/y", "z2/x", "z2/y",
                "z1/sigma", "z2/sigma",
                "L.1.1", "L.2.1", "L.1.2", "L.2.2",
                "z1/Intercept", "z2/Intercept",
                "Sigma.1.1", "Sigma.2.1", "Sigma.1.2", "Sigma.2.2"])
        
        numpy.testing.assert_allclose(
            model.draws.iloc[:5,:5].values,
            [
                [ 23.2660226,-11.9262003,  0.9088701,  1.9621971,  4.9687332],
                [ 23.2807593,-11.8603589,  1.0579523,  2.0172004,  5.0199548],
                [ 23.2530863,-11.672336 ,  0.7379906,  2.129682 ,  5.1128176],
                [ 23.1146539,-12.183051 ,  0.9163538,  2.0082782,  5.1746667],
                [ 23.2309701,-12.0327003,  0.8934087,  2.1269978,  5.173563 ]])
    
    def _test_model_log_likelihood(self, model):
        self.assertEqual(
            model.log_likelihood.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.log_likelihood.iloc[:5,:5].values,
            [
                [-4.4402421,-3.3641898,-3.4745691,-4.9062166,-3.2324176],
                [-4.6233423,-3.3707746,-3.6820983,-5.1131563,-3.2237245],
                [-3.8385064,-3.2420162,-3.3174715,-4.3918281,-3.1988372],
                [-3.6992945,-3.2302977,-3.5341838,-4.726541 ,-3.2277161],
                [-4.2840764,-3.1872561,-3.5349734,-4.8327181,-3.1293843]])

if __name__ == "__main__":
    unittest.main()
