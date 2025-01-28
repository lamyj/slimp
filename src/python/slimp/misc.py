import pandas

def sample_data_as_df(data):
    return pandas.DataFrame(
        data["array"].reshape((data["array"].shape[0], -1), order="A").T,
        columns=data["columns"])
