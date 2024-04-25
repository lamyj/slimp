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
    
    // Location and scale of the intercept prior
    real mu_alpha, sigma_alpha;
    
    // Scale of the non-intercept priors (location is 0)
    vector<lower=0>[K-1] sigma_beta;
    
    // Scale of the variance prior
    real<lower=0> lambda_sigma;
    
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
    // Expected value and draws of the prior predictive distribution
    vector[N] mu, y;
    
    {
        real alpha_c_ = student_t_rng(3, mu_alpha, sigma_alpha);
        vector[K-1] beta_ = to_vector(student_t_rng(3, 0, sigma_beta));
        real sigma_ = exponential_rng(lambda_sigma);
        
        mu = alpha_c_ + ((N_new > 0)?X_c_new:X_c) * beta_;
        y = to_vector(normal_rng(mu, sigma_));
    }
}
