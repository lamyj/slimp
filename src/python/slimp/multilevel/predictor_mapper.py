import re

class PredictorMapper:
    """ Map the low-level stan names to the high-level predictor names
    """
    
    def __init__(self, unmodeled_predictors, modeled_predictors, outcomes):
        self._common_names = {
            "alpha": "Intercept",
            "alpha_c": "Intercept_c",
            "sigma": "sigma"}
        
        self._outcomes = {1+i: c for i, c in enumerate(outcomes.columns)}
        
        self._beta = {
            1+index: name
            for index, name in enumerate(
                unmodeled_predictors.filter(regex="^(?!Intercept)").columns)}
        
        self._group_name = modeled_predictors.index.name
        self._groups = modeled_predictors.index.categories
        self._Beta = {
            1+index: name
            for index, name in enumerate(modeled_predictors.columns)}
        
    def __call__(self, x):
        if not isinstance(x, str):
            return [self.__call__(item) for item in x]
        
        match = re.match("([^.]+)\.(\d+)(?:\.(\d+))?", x)
        if match:
            kind, a, b = match.groups()
            if b is not None:
                group, index = int(a), int(b)
            else:
                index = int(a)
        else:
            kind = x
            index = None
            
        if kind in self._common_names:
            return self._common_names[kind]
        elif kind == "beta":
            return self._beta[index]
        elif kind == "Beta":
            coefficient = self._Beta[index]
            group = self._groups[group-1]
            return f"{self._group_name}[{group}]/{coefficient}"
        elif kind == "Sigma_Beta":
            return f"{kind}[{self._Beta[group]}, {self._Beta[index]}]"
        elif kind.endswith("_") and not kind.endswith("__"):
            return f"{kind[:-1]}[{index}]"
        else:
            return x
