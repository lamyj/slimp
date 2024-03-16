import re

class PredictorMapper:
    """ Map the low-level stan names to the high-level predictor names
    """
    
    def __init__(self, predictors, outcomes=None):
        self._common_names = {
            "alpha": "Intercept",
            "alpha_c": "Intercept_c",
            "sigma": "sigma"}
        self._beta = {}
        
        if outcomes is None:
            names = predictors.filter(regex="^(?!Intercept)").columns
            for name_index, name in enumerate(names):
                self._beta[1+name_index] = name
            self._outcomes = []
        
        else:
            self._outcomes = {1+i: c for i, c in enumerate(outcomes.columns)}
            index = 0
            for p, o in zip(predictors, self._outcomes.values()):
                for name in p.filter(regex="^(?!Intercept)").columns:
                    self._beta[1+index] = f"{o}/{name}"
                    index += 1
        
    def __call__(self, x):
        if not isinstance(x, str):
            return [self.__call__(item) for item in x]
        
        match = re.match("(.+)\[(\d+)\]", x)
        if match:
            kind, index = match.groups()
            index = int(index)
        else:
            kind = x
            index = None
            
        if kind in self._common_names:
            name = self._common_names[kind]
            if not self._outcomes:
                return name
            else:
                return f"{self._outcomes[index]}/{name}"
        elif kind == "beta":
            return self._beta[index]
        elif kind.endswith("_") and not kind.endswith("__"):
            return f"{kind[:-1]}[{index}]"
        else:
            return x
