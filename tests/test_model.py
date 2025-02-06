import unittest

import numpy
import pandas
import slimp

class TestModel(unittest.TestCase):
    def _test_data(self, model):
        """Test that model state is unmodified by pickling/unpickling"""
        
        self.assertEqual(self.formula, model.formula)
        self.assertTrue(self.data.equals(model.data))
        
        model_predictors = (
            (model.predictors, ) if not isinstance(model.predictors, tuple)
            else model.predictors)
        for p1, p2 in zip(self.predictors, model_predictors):
            self.assertTrue(p1.equals(p2))
        self.assertTrue(self.outcomes.equals(model.outcomes))
    
    def _test_sampler_parameters(self, model, seed, num_chains, num_samples):
        self.assertEqual(model.sampler_parameters.seed, seed)
        self.assertEqual(model.sampler_parameters.num_chains, num_chains)
        self.assertEqual(model.sampler_parameters.num_samples, num_samples)
    
    def _test_hmc_diagnostics(self, model):
        """Test HMC exploration"""
        
        numpy.testing.assert_allclose(
            model.hmc_diagnostics.apply({
                "divergent": sum, "depth_exceeded": sum,
                "e_bfmi": lambda x: sum(x<0.3)}),
            [0, 0, 0])
    
    def _test_draws(self, model, alpha):
        """Test that the draws of the Slimp model match the frequentist parameters"""
        
        summary = model.summary([45, 55])
        
        self.assertTrue(numpy.nanmax(summary["R_hat"]) < 1.01)
        self.assertTrue(numpy.nanmin(summary["N_Eff"])/len(model.draws) > 1e-3)
        
        for name, value in self.parameters.items():
            low, high = slimp.stats.hdi(model.draws[name], alpha)
            self.assertTrue(low < value < high)
    
        
    def _test_log_likelihood(self, model, alpha):
        """Test that the log-likelihood of the Slimp model match the frequentist one"""
        
        low, high = numpy.array([
            slimp.stats.hdi(row, alpha)
            for _, row in model.log_likelihood.T.iterrows()]).T
        self.assertTrue(
            all((low < self.log_likelihood) & (self.log_likelihood < high)))
    
    def _test_prior_predict(self, model, alpha):
        low, high = numpy.array([
            slimp.stats.hdi(row, alpha)
            for _, row in model.prior_predict.T.iterrows()]).T
        
        y = numpy.ravel(model.outcomes)
        self.assertTrue(all((low < y) & (y < high)))
    
    def _test_posterior_epred(self, model, alpha):
        low, high = numpy.array([
            slimp.stats.hdi(row, alpha)
            for _, row in model.posterior_epred.T.iterrows()]).T
        self.assertTrue(all((low < self.predicted) & (self.predicted < high)))
    
    def _test_posterior_predict(self, model, alpha):
        low, high = numpy.array([
            slimp.stats.hdi(row, alpha)
            for _, row in model.posterior_predict.T.iterrows()]).T
        self.assertTrue(all((low < self.predicted) & (self.predicted < high)))
    
    def _test_r_squared(self, model, alpha):
        r_squared = slimp.stats.r_squared(model)
        if isinstance(r_squared, pandas.DataFrame):
            low, high = r_squared.apply(slimp.stats.hdi, mass=alpha).values
        else:
            low, high = slimp.stats.hdi(r_squared, alpha)
        self.assertTrue(
            numpy.all((low < self.r_squared) & (self.r_squared < high)))

