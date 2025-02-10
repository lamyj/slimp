parameters
{
    // Centered intercept
    vector[K0?1:0] alpha_c;
    // Vector of unmodeled individual-level, non-intercept, coefficients
    vector[K0?(K0-1):0] beta;
    // Variance of individual-level regression
    real<lower=1.2e-38, upper=3.4e+38> sigma_y;
    
    // Vector of modeled individual-level coefficients
    // NOTE: store as array of vector to allow vectorization in model section
    array[J] vector[K] Beta;
    // Covariance matrix of group-level regression, split as variance and
    // Cholesky-factored correlation
    vector<lower=0>[K] sigma_Beta;
    cholesky_factor_corr[K] L_Omega_Beta;
}
