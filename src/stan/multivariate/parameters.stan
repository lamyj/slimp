parameters
{
    // Centered intercepts
    vector[R] alpha_c;
    
    // Non-intercept parameters
    vector[sum(K_c)] beta;
    
    // Variance. NOTE: it cannot be 0 or infinity, this causes warnings in the
    // likelihood. Values are taken from std::numeric_limits<float>.
    // WARNING: very bad exploration may happen with numeric bounds. Better
    // have a few warnings during init/warmup
    vector<lower=0/* 1.2e-38, upper=3.4e+38 */>[R] sigma;
    
    cholesky_factor_corr[use_covariance ? R : 0] L;
}
