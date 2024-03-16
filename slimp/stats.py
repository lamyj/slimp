def r_squared(model):
    # https://avehtari.github.io/bayes_R2/bayes_R2.html
    
    mu = model.posterior_epred
    var_mu = mu.var("columns")
    var_sigma = model.draws["sigma"]**2
    
    return var_mu/(var_mu+var_sigma)
