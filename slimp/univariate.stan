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

Note than when predicting data, the new predictors must be offset by the mean of
the *original* predictors.

*/
    
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
    
    // Number of new outcomes to predict
    int<lower=0> N_new;
    // New predictors
    matrix[N_new, K] X_new;
    // Whether to generate data from prior or posterior distribution
    int use_prior;
}

transformed data
{
    // Center the predictors
    vector[K-1] X_bar;
    matrix[N, K-1] X_c;
    for(k in 2:K)
    {
        X_bar[k-1] = mean(X[, k]);
        X_c[, k-1] = X[, k] - X_bar[k-1];
    }
    
    // Center the new predictors, if given, around the *original* predictors
    matrix[N_new, K-1] X_c_new;
    if(N_new > 0)
    {
        for(k in 2:K)
        {
            X_c_new[, k-1] = X_new[, k] - X_bar[k-1];
        }
    }
}
 
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
    
    // Expected Value of the prior-or-posterior predictive distribution
    vector[N_new] mu_rep;
    // Draw from the prior-or-posterior predictive distribution
    vector[N_new] y_rep;
    if(N_new > 0)
    {
        real alpha_c_;
        vector[K-1] beta_;
        real sigma_;
        
        if(use_prior == 1)
        {
            // WARNING: the RNG must match the prior
            alpha_c_ = student_t_rng(3, mu_alpha, sigma_alpha);
            beta_ = to_vector(student_t_rng(3, 0, sigma_beta));
            sigma_ = exponential_rng(lambda_sigma);
        }
        else
        {
            alpha_c_ = alpha_c;
            beta_ = beta;
            sigma_ = sigma;
        }
        
        // WARNING: must match the likelihood
        mu_rep = alpha_c_ + X_c_new*beta_;
        y_rep = to_vector(normal_rng(mu_rep, sigma_));
    }
}
