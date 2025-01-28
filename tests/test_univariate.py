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
            self._test_r_squared(model)
    
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
                0.99950108, 0.99950058, 0.99950003, 1.00001659, 0.99949995,
                0.99949997])
        numpy.testing.assert_allclose(
            model.summary()["N_Eff"].values,
            [
                2366.83058175, 1859.30561111, 2031.3147285, 1706.30225769,
                4556.36175122, 3752.54337])
    
    def _test_model_draws(self, model):
        self.assertEqual(
            list(model.draws.columns),
            ["Intercept_c", "x", "y", "sigma", "Intercept"])
        self.assertEqual(len(model.draws), 4000)
        numpy.testing.assert_allclose(
            model.draws.iloc[:5,:].values,
            [
                [23.45920304, 0.94076131, 2.00342471, 1.80679735, 10.21036599],
                [23.21453379, 0.96373905, 2.10546866, 1.87596441,  9.40309910],
                [23.30736547, 0.80978843, 2.04978430, 1.62582089, 10.43928818],
                [23.21592992, 0.81329983, 1.97912950, 1.58232663, 10.64999794],
                [23.15273890, 0.87012727, 2.09907340, 1.94761772,  9.79133590]])
    
    def _test_model_log_likelihood(self, model):
        self.assertEqual(
            model.log_likelihood.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.log_likelihood.iloc[:5,:5].values,
            [
                [-1.60441129, -1.53850613, -1.73233190, -2.90130344, -1.54037856],
                [-1.90739258, -1.56613167, -2.09653720, -3.54786381, -1.55869974],
                [-1.46303636, -1.45720832, -1.69416671, -3.31467313, -1.40902494],
                [-1.40138833, -1.48713917, -1.58510564, -3.12299091, -1.40539647],
                [-1.77602061, -1.58605231, -1.99561233, -3.33620151, -1.59444573]])
    
    def _test_model_prior_predict(self, model):
        self.assertEqual(
            model.prior_predict.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.prior_predict.iloc[:5,:5].values,
            [
                [-18.06286122, -12.58543198,  -0.214655090,  -8.31313469,   1.12007690],
                [ 13.71910935,  13.49187302,   1.243916520,   4.56551720,  -3.72083258],
                [-22.66833701, -19.59448697, -17.837483560, -16.05096583, -12.04535025],
                [ 27.34140564,  25.40452606,  22.490696120,  22.12190440,  20.75926511],
                [ 25.76404785,  27.95887682,  30.575716000,  33.02602902,  35.47149404]])
    
    def _test_model_posterior_epred(self, model):
        self.assertEqual(
            model.posterior_epred.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_epred.iloc[:5,:5].values,
            [
                [10.21036599, 11.15112730, 12.09188860, 13.03264991, 13.97341122],
                [ 9.40309910, 10.36683815, 11.33057720, 12.29431625, 13.25805530],
                [10.43928818, 11.24907661, 12.05886504, 12.86865347, 13.67844190],
                [10.64999794, 11.46329777, 12.27659761, 13.08989744, 13.90319728],
                [ 9.79133590, 10.66146317, 11.53159043, 12.40171770, 13.27184497]])
    
    def _test_model_posterior_predict(self, model):
        self.assertEqual(
            model.posterior_predict.shape, (4000, len(self.data)))
        numpy.testing.assert_allclose(
            model.posterior_predict.iloc[:5,:5].values,
            [
                [10.07895016, 13.09324239, 13.21282666, 14.20293923, 16.11242191],
                [ 7.51761611,  9.33578547,  9.16776823, 12.11458880, 11.91009901],
                [10.77932105, 12.01280211, 15.06289714, 12.25601876, 15.32031859],
                [10.56669465,  9.54664587, 14.04936540, 17.24218700, 13.96252218],
                [12.90982723, 14.56766386,  8.37414420,  9.05623659, 12.67619546]])
    
    def _test_r_squared(self, model):
        r_squared = slimp.stats.r_squared(model)
        self.assertEqual(r_squared.shape, (4000,))
        numpy.testing.assert_allclose(
            r_squared.iloc[:25],
            [
                0.92595347, 0.92698794, 0.93870112, 0.93841782, 0.91898326,
                0.91807913, 0.93219803, 0.92916761, 0.92198804, 0.91996784,
                0.93016495, 0.92527508, 0.93073202, 0.92881158, 0.93278984,
                0.92734158, 0.92289233, 0.91758261, 0.91551991, 0.92341071,
                0.92614523, 0.91026107, 0.93665560, 0.91038699, 0.91225971])

if __name__ == "__main__":
    unittest.main()
