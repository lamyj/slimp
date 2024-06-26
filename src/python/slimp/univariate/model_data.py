import formulaic
import numpy
import pandas

from .predictor_mapper import PredictorMapper

class ModelData:
    def __init__(self, formula, data):
        self.formula = formula
        self.data = data
        
        self.outcomes, self.predictors = formulaic.model_matrix(formula, data)
        
        self.predictor_mapper = PredictorMapper(self.predictors, self.outcomes)
        
        mu_y = numpy.mean(self.outcomes.values)
        sigma_y = numpy.std(self.outcomes.values)
        sigma_X = numpy.std(
            self.predictors.filter(regex="^(?!.*Intercept)").values, axis=0)
        
        self.fit_data = {
            "N": len(data), "K": self.predictors.shape[1],
            "y": numpy.squeeze(self.outcomes), "X": self.predictors,
            
            "mu_alpha": mu_y, "sigma_alpha": 2.5*sigma_y,
            "sigma_beta": 2.5*sigma_y/sigma_X,
            "lambda_sigma": numpy.squeeze(1/sigma_y)}
        