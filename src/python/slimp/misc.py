import pandas

def sample_data_as_df(data):
    return pandas.DataFrame(
        data["array"].reshape(-1, data["array"].shape[-1], order="A"),
        columns=data["columns"])
