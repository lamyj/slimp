import re

class PredictorMapper:
    """ Map the low-level stan names to the high-level predictor names
    """
    
    def __init__(self, predictors, outcomes):
        self._common_names = {
            "alpha": "Intercept",
            "alpha_c": "Intercept_c",
            "sigma": "sigma"}
        
        self._outcomes = {1+i: c for i, c in enumerate(outcomes.columns)}
        
        self._beta = {
            1+index: name
            for index, name in enumerate(
                predictors.filter(regex="^(?!Intercept)").columns)}
    
    def __call__(self, x):
        if not isinstance(x, str):
            return [self.__call__(item) for item in x]
        
        match = re.match("(.+)\.(\d+)", x)
        if match:
            kind, index = match.groups()
            index = int(index)
        else:
            kind = x
            index = None
            
        if kind in self._common_names:
            return self._common_names[kind]
        elif kind == "beta":
            return self._beta[index]
        elif kind.endswith("_") and not kind.endswith("__"):
            return f"{kind[:-1]}[{index}]"
        else:
            return x
