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
        
        for p1, p2 in zip(self.predictors, model.predictors):
            self.assertTrue(p1.equals(p2))
        self.assertTrue(self.outcomes.equals(model.outcomes))
    
    def _test_model_diagnostics(self, model):
        numpy.testing.assert_equal(
            model.hmc_diagnostics.max().values[:2], [0, 0])
        numpy.testing.assert_allclose(
            model.hmc_diagnostics["e_bfmi"].values,
            [0.97782139, 1.11264921, 1.10402065, 1.0453477])
        numpy.testing.assert_allclose(
            model.summary()["R_hat"].values,
            [
                0.99931725, 0.99949593,
                0.9995415, 0.99946114, 0.99977369, 0.99940114,
                0.99930313, 0.9993098,
                numpy.nan, numpy.nan, 0.99918394, 0.99930239,
                0.99957314, 0.99931119,
                numpy.nan, 0.99918394, 0.99918394, numpy.nan])
    
    def _test_model_draws(self, model):
        self.assertEqual(
            list(model.draws.columns),
            [
                "z1/Intercept_c", "z2/Intercept_c",
                "z1/x", "z1/y", "z2/x", "z2/y",
                "z1/sigma", "z2/sigma",
                "L[1,1]", "L[2,1]", "L[1,2]", "L[2,2]",
                "z1/Intercept", "z2/Intercept",
                "Sigma[1,1]", "Sigma[2,1]", "Sigma[1,2]", "Sigma[2,2]"])
        numpy.testing.assert_allclose(
            model.draws.describe().values,
            [
                [
                    4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0,
                    4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0,
                    4000.0, 4000.0, 4000.0, 4000.0],
                [
                    23.292915664571225, -11.953762954997192, 0.9066306838528909,
                    2.037453544904818, 5.076821071394089, -7.003248112188733,
                    1.8424218262277359, 1.9451553204801024, 1.0,
                    -0.11790820622382898, 0.0, 0.9882361816742625,
                    10.04453663516153, -3.2848412714212936, 1.0,
                    -0.11790820622382898, -0.11790820622382898, 1.0],
                [
                    0.1829586428967907, 0.19027623926973952, 0.06405571980698378,
                    0.06364606839453098, 0.06617608264497767, 0.06949940717148995,
                    0.13522520591355608, 0.14407998016609697, 0.0,
                    0.09647605543790352, 0.0, 0.013477654885364256,
                    0.44872863473783103, 0.47217821253191167, 0.0,
                    0.09647605543790352, 0.09647605543790352,
                    9.425003409437112e-17],
                [
                    22.638380816509276, -12.677938906105357, 0.6848164284919229,
                    1.825154847157177, 4.8641979452786765, -7.2266722109257495,
                    1.481715048943309, 1.475325260596617, 1.0,
                    -0.4998683277677738, 0.0, 0.8661014114378578,
                    8.392779165529822, -4.872132233491925, 1.0,
                    -0.4998683277677738, -0.4998683277677738,
                    0.9999999999999998],
                [
                    23.172958821026146, -12.082222949686667, 0.8638732700760804,
                    1.9946130147119674, 5.031452979103291, -7.050516119945792,
                    1.748430778102618, 1.8429324641866538, 1.0,
                    -0.1824159925283452, 0.0, 0.9832047082549638,
                    9.737394047529511, -3.599694370440481, 1.0,
                    -0.1824159925283452, -0.1824159925283452, 1.0],
                [
                    23.297535808547288, -11.953755710161275, 0.9072262263281412,
                    2.038415026307854, 5.077023712251806, -7.002834647163905,
                    1.8341276034814342, 1.9342104539936198, 1.0,
                    -0.11914681376106773, 0.0, 0.9926163153785295,
                    10.04712642808761, -3.279726337284848, 1.0,
                    -0.11914681376106773, -0.11914681376106773, 1.0],
                [
                    23.414881445294785, -11.827186645456546, 0.9496262281823566,
                    2.0799788380503896, 5.122575111912635, -6.956987873207689,
                    1.9267010221875194, 2.037552670590557, 1.0,
                    -0.05405597174621992, 0.0, 0.9980144888091018,
                    10.341823298702858, -2.9652090989981446, 1.0,
                    -0.05405597174621992, -0.05405597174621992, 1.0],
                [
                    23.886309231503407, -11.271358583109274, 1.1291002308476907,
                    2.2792935296864023, 5.301100750990051, -6.764390906818612,
                    2.47402755834427, 2.752227482220829, 1.0,
                    0.21706302023830526, 0.0, 0.9999999999992683,
                    11.651165182343037, -1.617992153129629, 1.0,
                    0.21706302023830526, 0.21706302023830526,
                    1.0000000000000002]])

if __name__ == "__main__":
    unittest.main()
