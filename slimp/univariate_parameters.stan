parameters
{
    // Centered intercept
    real alpha_c;
    
    // Non-intercept parameters
    vector[K-1] beta;
    
    // Variance. NOTE: it cannot be 0 or infinity, this causes warnings in the
    // likelihood. Values are taken from std::numeric_limits<float>.
    real<lower=1.2e-38, upper=3.4e+38> sigma;
}
