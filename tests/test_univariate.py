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
            model = slimp.Model(self.formula, self.data)
            with open(path, "wb") as fd:
                pickle.dump(model, fd)
        def load(path):
            with open(path, "rb") as fd:
                return pickle.load(fd)
        
        with tempfile.TemporaryDirectory() as dir:
            dump(os.path.join(dir, "model.pkl"))
            model = load(os.path.join(dir, "model.pkl"))
            
            self._test_model_data(model)
            self.assertTrue(model.draws is None)
    
    def test_sample(self):
        def dump(path):
            model = slimp.Model(self.formula, self.data)
            model.sample(
                seed=42, chains=4, parallel_chains=4, show_progress=False)
            with open(path, "wb") as fd:
                pickle.dump(model, fd)
        def load(path):
            with open(path, "rb") as fd:
                return pickle.load(fd)
        
        with tempfile.TemporaryDirectory() as dir:
            dump(os.path.join(dir, "model.pkl"))
            model = load(os.path.join(dir, "model.pkl"))
            
            self._test_model_data(model)
            self._test_model_diagnostics(model)
            self._test_model_draws(model)
    
    def _test_model_data(self, model):
        self.assertEqual(self.formula, model.formula)
        self.assertTrue(self.data.equals(model.data))
        
        self.assertTrue(self.predictors.equals(model.predictors))
        self.assertTrue(self.outcomes.equals(model.outcomes))
    
    def _test_model_diagnostics(self, model):
        numpy.testing.assert_equal(
            model.hmc_diagnostics.max().values[:2], [0, 0])
        numpy.testing.assert_allclose(
            model.hmc_diagnostics["e_bfmi"].values,
            [1.20497444, 1.08171924, 0.99332212, 1.06591932])
        numpy.testing.assert_allclose(
            model.summary()["R_hat"].values,
            [0.99963234, 0.99971568, 1.00104237, 1.00016844, 0.99981975])
    
    def _test_model_draws(self, model):
        self.assertEqual(
            list(model.draws.columns),
            ["Intercept_c", "x", "y", "sigma", "Intercept"])
        numpy.testing.assert_allclose(
            model.draws.describe().values,
            [
                [4000, 4000, 4000, 4000, 4000],
                [
                    2.32912919e+01, 9.08959100e-01, 2.03775860e+00,
                    1.83256516e+00, 1.00310623e+01],
                [
                    1.79593709e-01, 6.51748096e-02, 6.29292161e-02,
                    1.29806604e-01, 4.55836124e-01],
                [
                    2.26073149e+01, 6.83280776e-01, 1.81157057e+00,
                    1.45433416e+00, 8.52093115e+00],
                [
                    2.31732073e+01, 8.65168009e-01, 1.99397270e+00,
                    1.74159132e+00, 9.71935104e+00],
                [
                    2.32940689e+01, 9.07195863e-01, 2.03738302e+00,
                    1.82734906e+00, 1.00372218e+01],
                [
                    2.34113178e+01, 9.52790002e-01, 2.08050967e+00,
                    1.91459671e+00, 1.03481737e+01],
                [
                    2.40368571e+01, 1.12643006e+00, 2.28690589e+00,
                    2.36840101e+00, 1.14633799e+01]])

if __name__ == "__main__":
    unittest.main()
