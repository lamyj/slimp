class Samples:
    def __init__(self, samples, predictor_mapper, parameters_columns):
        self.samples = samples
        self.diagnostics = self.samples.filter(regex="_$")
        self.draws = self.samples.filter(regex="[^_]$")
        self.predictor_mapper = predictor_mapper
        self.draws.columns = predictor_mapper(self.draws.columns)
        self.parameters_columns = parameters_columns
