class NoCorrelation(list):
    def __init__(self, *args, **kwargs):
        list.__init__(self, *args, **kwargs)

from .model_data import ModelData
from .predictor_mapper import PredictorMapper
