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
            self._test_model_log_likelihood(model)
            self._test_model_prior_predict(model)
            self._test_model_posterior_epred(model)
            self._test_model_posterior_predict(model)
    
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
        self.assertEqual(len(model.draws), 4000)
        numpy.testing.assert_allclose(
            model.draws.iloc[:5,:].values,
            [
                [23.4592030, 0.94076130, 2.00342470, 1.806797348, 10.21036598],
                [23.3905170, 0.98711629, 1.91642734, 1.881650569, 10.32457061],
                [23.4079064, 0.91977690, 2.14917482, 1.783852860, 9.597623651],
                [23.0649425, 0.98298884, 2.12437401, 1.843825196, 9.081809700],
                [23.2145337, 0.96373905, 2.10546865, 1.875964407, 9.403099096]])
    
    def _test_model_log_likelihood(self, model):
        self.assertEqual(
            model.log_likelihood.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.log_likelihood.iloc[:5,:5].values,
            [
                [-1.6044112, -1.5385061, -1.7323319, -2.9013034, -1.5403785],
                [-1.4785300, -1.5549794, -1.6483398, -3.0005365, -1.4751144],
                [-1.5577002, -1.6164186, -1.6576977, -2.6633665, -1.5912700],
                [-1.6905174, -1.5387873, -1.8659833, -3.1639919, -1.5364730],
                [-1.6142649, -1.5999491, -1.6913405, -2.6269443, -1.6287000]])
    
    def _test_model_prior_predict(self, model):
        self.assertEqual(
            model.prior_predict.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.prior_predict.iloc[:5,:5].values,
            [
                [-18.0628612,-12.585432 , -0.2146551, -8.3131347,  1.1200769],
                [ 78.955988 , 75.438115 , 71.8809921, 68.3484672, 64.6451405],
                [ 78.2182381, 77.1482612, 49.8129586, 47.0666833, 23.8685327],
                [-75.2312012,-46.1937697,-18.170517 , 10.4951465, 37.1835187],
                [ 13.7191093, 13.491873 ,  1.2439165,  4.5655172, -3.7208326]])
    
    def _test_model_posterior_epred(self, model):
        self.assertEqual(
            model.posterior_epred.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_epred.iloc[:5,:5].values,
            [
                [10.210366 ,11.1511273,12.0918886,13.0326499,13.9734112],
                [10.6838411,11.4645385,12.2452359,13.0259334,13.8066308],
                [10.5832679,11.4779999,12.3727319,13.2674639,14.1621959],
                [ 9.9617344,10.8754661,11.7891979,12.7029297,13.6166614],
                [10.3245706,11.3116869,12.2988032,13.2859195,14.2730358]])
    
    def _test_model_posterior_predict(self, model):
        self.assertEqual(
            model.posterior_predict.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_predict.iloc[:5,:5].values,
            [
                [10.0789502,13.0932424,13.2128267,14.2029392,16.1124219],
                [12.4243714,13.452148 ,12.1742729,13.2278094,13.6020354],
                [14.0521864,11.4358118,12.9317417,14.3156692,14.6669161],
                [11.2820318,12.7343338,10.5181746,12.7999644,13.3685442],
                [ 8.4333726,10.277509 ,10.1294386,13.1056473,12.9209938]]
)

if __name__ == "__main__":
    unittest.main()
