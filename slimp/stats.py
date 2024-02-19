def r_squared(model, **kwargs):
    # https://avehtari.github.io/bayes_R2/bayes_R2.html
    
    mu, _ = model.predict(model.data, **kwargs)
    var_mu = mu.std("columns")**2
    var_sigma = model.draws["sigma"]**2
    
    return var_mu/(var_mu+var_sigma)
