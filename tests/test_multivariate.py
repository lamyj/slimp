# NOTE: informational messages are logged at the ERROR level
import logging
logging.basicConfig(level=logging.CRITICAL)

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
            self._test_model_prior_predict(model)
            self._test_model_posterior_epred(model)
            self._test_model_posterior_predict(model)
            self._test_r_squared(model)
    
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
    
    def _test_model_prior_predict(self, model):
        self.assertEqual(
            model.prior_predict.shape, (4000, len(self.formula)*len(self.data)))
        numpy.testing.assert_allclose(
            model.prior_predict.iloc[:5,:5].values,
            [
                [48.6570697,31.823942 ,26.8655552,16.4069899, 6.0898614],
                [11.4029285,15.2350219,17.5113849,30.8652432,27.1702647],
                [94.7920922,85.4160147,79.9455973,70.0779007,63.7733472],
                [65.9746989,62.0146905,42.0816617,35.382645 ,22.5919671],
                [96.5068115,80.8457643,59.8711939,55.3557259,53.9276823]])
    
    def _test_model_posterior_epred(self, model):
        self.assertEqual(
            model.posterior_epred.shape, (4000, len(self.formula)*len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_epred.iloc[:5,:5].values,
            [
                [10.3462201,11.2550902,12.1639604,13.0728305,13.9817006],
                [ 9.6681166,10.5864564,11.5047961,12.4231358,13.3414756],
                [10.5162627,11.3718041,12.2273456,13.082887 ,13.9384285],
                [ 9.9679279,10.8652681,11.7626083,12.6599485,13.5572887],
                [ 9.4425723,10.5005246,11.5584769,12.6164293,13.6743816]])
    
    def _test_model_posterior_predict(self, model):
        self.assertEqual(
            model.posterior_predict.shape, (4000, len(self.formula)*len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_predict.iloc[:5,:5].values,
            [
                [10.2116486,12.4029454,14.3543352,12.2280595,13.7668074],
                [11.6516064,10.5055876,11.2716413,13.104573 ,14.3552479],
                [14.3348934,11.9871694,12.7829482,14.8559529,14.7204133],
                [11.3787327, 9.5071152,11.4974825,16.4666559,15.9496412],
                [ 9.8451985,14.0575395,13.5025906,11.1136866,11.873199 ]])
    
    def _test_r_squared(self, model):
        r_squared = slimp.stats.r_squared(model)
        
        self.assertEqual(r_squared.shape, (4000, len(self.formula)))
        numpy.testing.assert_allclose(
            r_squared.iloc[:10],
            [
                [0.9828498,0.9735252],
                [0.981204 ,0.9766605],
                [0.9822883,0.9808393],
                [0.9852849,0.9768808],
                [0.9844661,0.978359 ],
                [0.9807788,0.9751221],
                [0.9842689,0.9735007],
                [0.9835481,0.9774315],
                [0.9825075,0.9752729],
                [0.9833985,0.9804559]])

if __name__ == "__main__":
    unittest.main()
