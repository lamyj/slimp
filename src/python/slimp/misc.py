import arviz
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

def to_arviz(model):
    """Convert the slimp mode to arviz inference data"""
    
    # Helper to rename slimp dimensions to arviz dimensions
    def rename(x, observations=False):
        x = x.rename({
            "sample": "draw",
            **({"parameter": "observation"} if observations else {})})
        return (
            x.assign_coords(observation=range(len(model.data))) if observations
            else x)
    # Helper to generate quantities as xarray
    def generate(name):
        return model._generate_quantities(name, converter=sample_data_as_xarray)
    
    log_likelihood = xarray.Dataset({
        model.outcomes.columns[0]: rename(generate("log_likelihood"), True)})
    
    # Split samples in sampling statistics and posterior
    samples = rename(model._samples.samples).to_dataset("parameter")
    sample_stats = (
        samples[[x for x in samples if x.endswith("__")]]
        .rename({
            "lp__": "lp", "accept_stat__": "acceptance_rate",
            "stepsize__": "step_size", "treedepth__": "tree_depth",
            "n_leapfrog__": "n_steps", "divergent__": "diverging",
            "energy__": "energy"}))
    posterior = samples[[x for x in samples if not x.endswith("__")]]
    
    # Get prior-predictive data, split it in y, mu variables in a dataset
    data = generate("predict_prior")
    prior_predictive = xarray.Dataset({
        x: rename(data[data.parameter.str.startswith(f"{x}.")], True)
        for x in ["y", "mu"]})
    
    # Same for posterior-predictive
    data = generate("predict_posterior")
    posterior_predictive = xarray.Dataset({
        x: rename(data[data.parameter.str.startswith(f"{x}.")], True)
        for x in ["y", "mu"]})
    
    ds = arviz.InferenceData(
        # TODO: prior
        sample_stats = sample_stats,
        posterior = posterior,
        log_likelihood = log_likelihood,
        prior_predictive = prior_predictive,
        posterior_predictive = posterior_predictive,
        observed_data = xarray.Dataset(model.outcomes))
    
    return ds
