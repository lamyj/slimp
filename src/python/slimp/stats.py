import numpy
import pandas

from . import _slimp

def r_squared(*args, **kwargs):
    # https://avehtari.github.io/bayes_R2/bayes_R2.html
    
    from .model import Model
    
    if len(args) == 1 and isinstance(args[0], Model):
        return _r_squared_model(*args, **kwargs)
    elif len(args) == 2 and isinstance(args[0], pandas.DataFrame):
        return _r_squared_data_frame(*args, **kwargs)
    else:
        raise NotImplementedError()

def _r_squared_model(model):
    if isinstance(model.formula, list) and isinstance(model.formula[1], tuple):
        return r_squared(model.posterior_epred, model.draws["sigma_y"])
    elif isinstance(model.formula, list):
        df = pandas.concat(
            [
                r_squared(
                    model.posterior_epred.filter(regex=fr"mu\.[^.]+\.{i+1}"),
                    model.draws[f"{c}/sigma"])
                for i, c in enumerate(model.outcomes.columns)],
            axis="columns")
        df.columns = model.outcomes.columns
        return df
    else:
        return r_squared(model.posterior_epred, model.draws["sigma"])

def _r_squared_data_frame(mu, sigma):
    var_mu = mu.var("columns")
    var_sigma = sigma**2
    return var_mu/(var_mu+var_sigma)

def hmc_diagnostics(data, max_depth):
    energy = data.sel(parameter="energy__").values
    diagnostics = pandas.DataFrame({
        "divergent": data.sel(parameter="divergent__").sum(axis=1),
        "depth_exceeded": 
            (data.sel(parameter="treedepth__")>=max_depth)
            .sum(axis=1),
        "e_bfmi": 
            numpy.square(numpy.diff(energy, axis=1)).mean(axis=1) 
                / numpy.var(energy, axis=1, ddof=1)})
    return diagnostics

def summary(data, percentiles=(5, 50, 95)):
    summary = {}
    
    summary["Mean"] = numpy.mean(data, axis=(1,2))
    summary["MCSE"] = None
    summary["StdDev"] = numpy.std(data, axis=(1,2))
    quantiles = numpy.quantile(data, numpy.array(percentiles)/100, axis=(1,2))
    for p, q in zip(percentiles, quantiles):
        summary[f"{p}%"] = q
    
    summary["N_Eff"] = _slimp.get_effective_sample_size(data)
    summary["R_hat"] = _slimp.get_split_potential_scale_reduction(data)
    
    summary["MCSE"] = numpy.sqrt(summary["StdDev"])/numpy.sqrt(summary["N_Eff"])
    
    return pandas.DataFrame(summary, index=data["parameter"])

def hdi(x, mass):
    """ Highest density interval, after "Doing Bayesian Data Analysis",
        J. Kruschke, section 25.2.3
    """
    
    x = numpy.sort(x)
    N = len(x)
    # Number of points in the interval to account for the mass
    count = int(numpy.ceil(mass * N))
    # Widths of all intervals of given mass
    widths = x[count:] - x[:N - count]
    # HDI is the narrowest interval
    min_index = numpy.argmin(widths)
    return (x[min_index], x[min_index+count])
