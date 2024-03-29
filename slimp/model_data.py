import formulaic
import numpy
import pandas

from .predictor_mapper import PredictorMapper

class ModelData:
    def __init__(self, formula, data):
        if isinstance(formula, str):
            formula = [formula]
        
        self.formula = formula
        self.data = data
        
        self.outcomes, self.predictors = zip(
            *[formulaic.model_matrix(f, data) for f in formula])
        self.outcomes = pandas.concat(self.outcomes, axis="columns")
        
        self.predictor_mapper = PredictorMapper(self.predictors, self.outcomes)
        
        mu_y = numpy.mean(self.outcomes.values, axis=0)
        sigma_y = numpy.atleast_1d(numpy.std(self.outcomes.values, axis=0))
        sigma_X = [
            numpy.std(x.filter(regex="^(?!.*Intercept)").values, axis=0)
            for x in self.predictors]
        
        self.fit_data = {
            "R": len(self.formula),
            "N": len(data),
            "K": numpy.squeeze([x.shape[1] for x in self.predictors]),
            "y": numpy.squeeze(self.outcomes.values),
            "X": pandas.concat(self.predictors, axis="columns"),
            
            "mu_alpha": numpy.squeeze(mu_y),
            "sigma_alpha": 2.5*numpy.squeeze(sigma_y),
            "sigma_beta": numpy.concatenate(numpy.atleast_2d(
                [2.5*(sy/sx) for sx, sy in zip(sigma_X, sigma_y)])),
            "lambda_sigma": numpy.squeeze(1/sigma_y),
            "eta_L": 1.0}
        