parameters
{
    // Centered intercepts
    vector[R] alpha_c;
    
    // Non-intercept parameters
    vector[sum(K_c)] beta;
    
    // Variance. NOTE: it cannot be 0 or infinity, this causes warnings in the
    // likelihood. Values are taken from std::numeric_limits<float>.
    vector<lower=1.2e-38, upper=3.4e+38>[R] sigma;
    
    cholesky_factor_corr[R] L;
}
