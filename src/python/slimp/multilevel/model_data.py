import formulaic
import numpy
import pandas

from .predictor_mapper import PredictorMapper

class ModelData:
    def __init__(self, formula, data):
        self.formula = formula
        self.data = data
        
        self.outcomes, self.unmodeled_predictors = formulaic.model_matrix(
            formula[0], data)
        
        group_name, group_formula = formula[1]
        self.modeled_predictors = formulaic.model_matrix(group_formula, data)
        self.modeled_predictors.index = data[group_name]
        
        self.predictor_mapper = PredictorMapper(
            self.unmodeled_predictors, self.modeled_predictors, self.outcomes)
        
        mu_y = numpy.mean(self.outcomes.values)
        sigma_y = numpy.std(self.outcomes.values)
        sigma_X = numpy.std(
            self.unmodeled_predictors.filter(regex="^(?!.*Intercept)").values,
            axis=0)
        
        self.fit_data = {
            "N": len(data),
            "K0": self.unmodeled_predictors.shape[1],
            "K": self.modeled_predictors.shape[1],
            "J": data[group_name].nunique(),
            
            "y": numpy.squeeze(self.outcomes),
            "X0": self.unmodeled_predictors,
            "X": self.modeled_predictors,
            
            "group": 1+data[group_name].cat.codes,
            
            "mu_alpha": mu_y, "sigma_alpha": 2.5*sigma_y,
            "sigma_beta": 2.5*sigma_y/sigma_X,
            
            "lambda_sigma_y": 1/sigma_y,
            "lambda_sigma_Beta": 1/sigma_y,
            
            "eta_L": 1.}
    
    @property
    def predictors(self):
        return (self.unmodeled_predictors, self.modeled_predictors)
    
    def new_data(self, X0_new=None, X_new=None):
        if X0_new is None:
            X0_new = self.fit_data["X0"]
        if X_new is None:
            X_new = self.fit_data["X"]
        
        return self.fit_data | {
            "N_new": X0_new.shape[0], "X0_new": X0_new, "X_new": X_new}
