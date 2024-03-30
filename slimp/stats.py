import pandas

def r_squared(model):
    # https://avehtari.github.io/bayes_R2/bayes_R2.html
    
    def compute(mu, sigma):
        var_mu = mu.var("columns")
        var_sigma = sigma**2
        return var_mu/(var_mu+var_sigma)
    
    if isinstance(model.formula, list):
        draws = model.draws
        epred = model.posterior_epred
        
        df = pandas.concat(
            [
                compute(epred.filter(like=f"mu.{1+i}"), draws[f"{c}/sigma"])
                for i, c in enumerate(model.outcomes.columns)],
            axis="columns")
        df.columns = model.outcomes.columns
        return df
    else:
        return compute(model.posterior_epred, model.draws["sigma"])
