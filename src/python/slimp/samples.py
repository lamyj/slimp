import pandas

class Samples:
    def __init__(self, samples, predictor_mapper, parameters_columns):
        self.predictor_mapper = predictor_mapper
        
        self.samples = samples
        self.samples["parameter"] = self.predictor_mapper(
            self.samples["parameter"].values)
        
        names = samples["parameter"].values
        diagnostics_names = [x for x in names if x.endswith("_")]
        self.diagnostics = self.samples.sel(parameter=diagnostics_names)
        
        parameters_names = [x for x in names if not x.endswith("_")]
        self.draws = pandas.DataFrame(
            self.samples.sel(parameter=parameters_names)
                .values
                .reshape(len(parameters_names), -1)
                .T,
            columns=predictor_mapper(parameters_names))
        
        self.parameters_columns = parameters_columns
