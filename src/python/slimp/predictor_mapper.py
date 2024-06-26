import re

class PredictorMapper:
    """ Map the low-level stan names to the high-level predictor names
    """
    
    def __init__(self, unmodeled_predictors, outcomes, modeled_predictors=None):
        self._common_names = {
            "alpha": "Intercept",
            "alpha_c": "Intercept_c",
            "sigma": "sigma"}
        
        self._outcomes = {1+i: c for i, c in enumerate(outcomes.columns)}
        
        self._beta = {}
        index = 0
        for p, o in zip(unmodeled_predictors, self._outcomes.values()):
            for name in p.filter(regex="^(?!Intercept)").columns:
                self._beta[1+index] = (
                    f"{o}/{name}" if len(self._outcomes)>1
                    else name)
                index += 1
        
        self._Beta = {}
        if modeled_predictors is not None:
            self._group_name = modeled_predictors.index.name
            self._groups = modeled_predictors.index.categories
            index = 0
            for name in modeled_predictors.columns:
                self._Beta[1+index] = name
                index += 1
        
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
            name = self._common_names[kind]
            if len(self._outcomes)>1:
                return f"{self._outcomes[index]}/{name}"
            else:
                return name
        elif kind == "beta":
            return self._beta[index]
        elif kind == "Beta":
            coefficient = self._Beta[index]
            group = self._groups[group-1]
            return f"{self._group_name}[{group}]/{coefficient}"
        elif kind.endswith("_") and not kind.endswith("__"):
            return f"{kind[:-1]}[{index}]"
        else:
            return x
