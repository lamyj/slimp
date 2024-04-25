/*
Univariate linear model with normal likelihood: y ~ N(µ, σ) and robust priors.

μ is usually written α_0 + Σ X_i β_i, where α_0 represents the expected value of
y when all predictors equal 0. It is however easier to define a prior on the
intercept after centering the predictors around 0; let Xbar_i be the mean of the
i-th predictors value, we then have:

µ = α_c + Σ (X_i - Xbar_i) β_i
α_c = α_0 + Σ Xbar_i β_i ~ Student(3, μ_α, σ_α)
βᵢ ~ Student(3, 0, σ_βᵢ)
σ ~ Exp(λ_σ)
*/

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
    
    // Location and scale of the intercept prior
    real mu_alpha, sigma_alpha;
    
    // Scale of the non-intercept priors (location is 0)
    vector<lower=0>[K-1] sigma_beta;
    
    // Scale of the variance prior
    real<lower=0> lambda_sigma;
}

transformed data
{
    // Center the predictors
    vector[K-1] X_bar = center_columns(X, N, K);
    matrix[N, K-1] X_c = center(X, X_bar, N, K);
}

#include univariate_parameters.stan

model
{
    alpha_c ~ student_t(3, mu_alpha, sigma_alpha);
    beta ~ student_t(3, 0, sigma_beta);
    sigma ~ exponential(lambda_sigma);
    
    y ~ normal_id_glm(X_c, alpha_c, beta, sigma);
}

generated quantities
{
    // Non-centered intercept
    real alpha = alpha_c - dot_product(X_bar, beta);
}
