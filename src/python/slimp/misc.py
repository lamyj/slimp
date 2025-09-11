import pandas
import xarray

def sample_data_as_df(data):
    return pandas.DataFrame(
        data["array"].reshape((data["array"].shape[0], -1), order="A").T,
        columns=data["columns"])

def sample_data_as_xarray(data):
    return xarray.DataArray(
        data["array"],
        dims=["parameter", "chain", "sample"],
        coords={
            "parameter": data["columns"],
            "chain": range(data["array"].shape[1]),
            "sample": range(data["array"].shape[2])})
