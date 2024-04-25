functions
{

#include functions.stan

}

data
{
    // Number of outcomes and predictors
    int<lower=1> N, K;
    
    // Predictors
    matrix[N, K] X;
    
    // Number of new outcomes to predict
    int<lower=0> N_new;
    // New predictors
    matrix[N_new, K] X_new;
}

transformed data
{
    // Center the predictors around the *original* predictors
    vector[K-1] X_bar = center_columns(X, N, K);
    matrix[N, K-1] X_c = center(X, X_bar, N, K);
    matrix[N_new, K-1] X_c_new = center(X_new, X_bar, N_new, K);
}

#include univariate_parameters.stan

generated quantities
{
    // Expected value and draws of the posterior predictive distribution
    vector[(N_new>0)?N_new:N] mu = alpha_c + ((N_new > 0)?X_c_new:X_c) * beta;
    vector[(N_new>0)?N_new:N] y = to_vector(normal_rng(mu, sigma));
}
