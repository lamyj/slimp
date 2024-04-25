functions
{

#include functions.stan

}

data
{
    // Number of outcomes and predictors
    int<lower=1> N, K;
    
    // Outcomes
    vector[N] y;
    // Predictors
    matrix[N, K] X;
}

transformed data
{
    // Center the predictors
    vector[K-1] X_bar = center_columns(X, N, K);
    matrix[N, K-1] X_c = center(X, X_bar, N, K);
}

#include univariate_parameters.stan

generated quantities
{
    vector[N] log_likelihood;
    {
        vector[N] mu = alpha_c + X_c*beta;
        // TODO: vectorize
        for(i in 1:N)
        {
            log_likelihood[i] = normal_lpdf(y[i] | mu[i], sigma);
        }
    }
}
