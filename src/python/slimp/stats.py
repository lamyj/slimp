import numpy
import pandas

from ._slimp import get_effective_sample_size, get_potential_scale_reduction

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
    if isinstance(model.formula, list):
        draws = model.draws
        epred = model.posterior_epred
        
        df = pandas.concat(
            [
                r_squared(epred.filter(like=f"mu.{1+i}"), draws[f"{c}/sigma"])
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
    diagnostics = (
        data.groupby("chain__")
        .agg(
            divergent=("divergent__", lambda x: numpy.sum(x!=0)),
            depth_exceeded=("treedepth__", lambda x: numpy.sum(x >= max_depth)),
            e_bfmi=(
                "energy__", 
                lambda x: (
                    numpy.sum(numpy.diff(x)**2)
                    / numpy.sum((x-numpy.mean(x))**2)))))
    diagnostics.index = diagnostics.index.rename("chain").astype(int)
    return diagnostics

def summary(data, chains, percentiles=(5, 50, 95)):
    summary = {}
    
    summary["Mean"] = numpy.mean(data, axis=0)
    summary["MCSE"] = None
    summary["StdDev"] = numpy.std(data, axis=0)
    quantiles = numpy.quantile(data, numpy.array(percentiles)/100, axis=0)
    for p, q in zip(percentiles, quantiles):
        summary[f"{p}%"] = q
        
    summary["N_Eff"] = get_effective_sample_size(data.values, chains)
    summary["R_hat"] = get_potential_scale_reduction(data.values, chains)
    
    summary["MCSE"] = numpy.sqrt(summary["StdDev"])/numpy.sqrt(summary["N_Eff"])
    
    return pandas.DataFrame(summary)
