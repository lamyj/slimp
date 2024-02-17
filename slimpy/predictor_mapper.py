class PredictorMapper:
    """ Map the low-level stan names to the high-level predictor names
    """
    
    def __init__(self, predictors):
        self._alpha = {}
        self._beta = {}
        self._sigma = {}
        
        self._alpha = "Intercept"
        
        names = predictors.filter(regex="^(?!Intercept)").columns
        for name_index, name in enumerate(names):
            self._beta[1+name_index] = name
        
    def __call__(self, x):
        if not isinstance(x, str):
            return [self.__call__(item) for item in x]
        
        if "." in x:
            kind, index = x.rsplit(".", 1)
            index = int(index)
        else:
            kind = x
            index = None
        
        if kind == "alpha":
            return self._alpha
        elif kind == "alpha_c":
            return f"{self._alpha}_c"
        elif kind == "beta":
            return self._beta[index]
        else:
            return x
