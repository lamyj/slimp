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
                1.00024531, 0.99986922, 0.99976707, 0.99984164, 0.99969618,
                0.99968495, 0.99975501, 0.99961358, 0.99976065, numpy.nan,
                0.99955651, numpy.nan, 0.99980195, 0.99982063, 0.99988733,
                numpy.nan, 0.99955651, 0.99955651, numpy.nan])
        numpy.testing.assert_allclose(
            model.summary()["N_Eff"].values,
            [
                1826.29936652, 7241.87853593, 6638.25525456, 6373.29667807,
                6352.21980307, 6909.38521697, 8028.92616944, 6878.05521,
                6648.02384053, numpy.nan, 6461.12004302, numpy.nan,
                3971.60651777, 7187.96325534, 7189.52437465, numpy.nan,
                6461.12004302, 6461.12004302, numpy.nan])
    
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
                [ 23.26602262, -11.92620028, 0.90887012, 1.96219710, 4.96873323],
                [ 23.23097006, -12.03270029, 0.89340874, 2.12699781, 5.17356305],
                [ 23.31411954, -11.90428078, 0.94075611, 1.95075665, 4.99088000],
                [ 23.17974763, -12.03963485, 0.84634525, 2.10564242, 5.13481246],
                [ 23.37709780, -11.85510748, 0.96276733, 2.00321889, 5.00349654]])
    
    def _test_model_log_likelihood(self, model):
        self.assertEqual(
            model.log_likelihood.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.log_likelihood.iloc[:5,:5].values,
            [
                [-4.44024210, -3.36418982, -3.47456909, -4.90621664, -3.23241763],
                [-3.71649349, -3.00786749, -3.56476896, -5.21525809, -3.01645182],
                [-4.39994455, -3.32592379, -3.43483485, -4.81318751, -3.22437137],
                [-3.80309789, -3.02472854, -3.44726044, -5.04387057, -3.01585067],
                [-4.56688219, -3.36194513, -3.59200055, -5.15338957, -3.25205569]])
    
    def _test_model_prior_predict(self, model):
        self.assertEqual(
            model.prior_predict.shape, (4000, len(self.formula)*len(self.data)))
        numpy.testing.assert_allclose(
            model.prior_predict.iloc[:5,:5].values,
            [
                [48.65706972, 31.82394200, 26.86555515, 16.40698992, 6.089861360],
                [96.50681152, 80.84576426, 59.87119393, 55.35572589, 53.92768225],
                [40.09735446, 48.02049867, 49.10997206, 55.90953048, 59.72760867],
                [33.49662541, 42.69101628, 43.95407864, 42.41969572, 42.05608402],
                [35.23973869, 25.93652203, 19.89544264, 17.67578903,  6.61698050]])
    
    def _test_model_posterior_epred(self, model):
        self.assertEqual(
            model.posterior_epred.shape, (4000, len(self.formula)*len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_epred.iloc[:5,:5].values,
            [
                [10.34622012, 11.25509024, 12.16396037, 13.07283049, 13.98170061],
                [ 9.63914062, 10.53254935, 11.42595809, 12.31936682, 13.21277556],
                [10.30231213, 11.24306823, 12.18382434, 13.12458045, 14.06533656],
                [ 9.89580311, 10.74214836, 11.58849362, 12.43483887, 13.28118413],
                [10.03015980, 10.99292713, 11.95569446, 12.91846180, 13.88122913]])
    
    def _test_model_posterior_predict(self, model):
        self.assertEqual(
            model.posterior_predict.shape, (4000, len(self.formula)*len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_predict.iloc[:5,:5].values,
            [
                [10.21164859, 12.40294542, 14.35433520, 12.22805946, 13.76680741],
                [10.01013867, 13.81014441, 13.21735270, 10.93467148, 11.55308417],
                [13.29756193,  8.21040301, 11.61171467, 14.86005629, 14.00376177],
                [ 5.94159958, 11.25884689, 15.92101793, 13.93521402, 14.02952034],
                [10.22161123, 11.05684236,  8.73832382, 14.44996461, 10.90163229]])
                    
    def _test_r_squared(self, model):
        r_squared = slimp.stats.r_squared(model)
        
        self.assertEqual(r_squared.shape, (4000, len(self.formula)))
        numpy.testing.assert_allclose(
            r_squared.iloc[:10],
            [
                [0.98284984, 0.9735252 ],
                [0.98472746, 0.97890647],
                [0.98257990, 0.97514798],
                [0.98406700, 0.97944039],
                [0.98316291, 0.97336551],
                [0.98227856, 0.96950293],
                [0.98596613, 0.98286283],
                [0.98174953, 0.97663381],
                [0.98493809, 0.97699799],
                [0.98316891, 0.97759876]])

if __name__ == "__main__":
    unittest.main()
