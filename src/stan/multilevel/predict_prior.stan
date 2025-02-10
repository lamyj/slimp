functions
{

#include functions.stan

}

data
{
    // Number of observations, unmodeled individual-level predictors, modeled
    // individual-level predictors, and groups. No group-level predictor is used
    // in this model, as the modeled invididual-level coefficients are centered
    // on 0.
    int<lower=1> N, K0, K, J;
    
    // Matrix of unmodeled individual-level predictors
    matrix[N, K0] X0;
    // Matrix of modeled individual-level predictors
    matrix[N, K] X;
    
    // Map from an observation to its group
    array[N] int<lower=1, upper=J> group;
    
    // Location and scale of the intercept prior
    real mu_alpha, sigma_alpha;
    
    // Scale of the non-intercept unmodeled coefficients priors (location is 0)
    vector<lower=0>[K0?(K0-1):0] sigma_beta;
    
    // Scale of the individual-level variance prior
    real<lower=0> lambda_sigma_y;
    
    // Scale of the group-level variance prior
    real<lower=0> lambda_sigma_Beta;
    
    // Parameter of the LKJ distribution
    real<lower=1> eta_L;
    
    // Number of new outcomes to predict
    int<lower=0> N_new;
    // New predictors
    matrix[N_new, K0] X0_new;
    matrix[N_new, K] X_new;
}

transformed data
{
    // Center the unmodeled predictors around the *original* predictors
    vector[K0?(K0-1):0] X0_bar = center_columns(X0, N, K0);
    matrix[N, K0?(K0-1):0] X0_c = center(X0, X0_bar, N, K0);
    matrix[N_new, K0?(K0-1):0] X0_c_new = center(X0_new, X0_bar, N_new, K0);
    
    vector[K] zeros_K = zeros_vector(K);
    
    // Final number of observations to generate.
    int N_final = (N_new>0)?N_new:N;
}

#include multilevel/parameters.stan

generated quantities
{
    // Expected value and draws of the posterior predictive distribution
    vector[N_final] mu, y;
    {
        real alpha_c_ = student_t_rng(3, mu_alpha, sigma_alpha);
        vector[K0-1] beta_0_ = to_vector(student_t_rng(3, 0, sigma_beta));
        real sigma_y_ = exponential_rng(lambda_sigma_y);
        
        vector[K] sigma_Beta_ = to_vector(
            exponential_rng(lambda_sigma_Beta * ones_vector(K)));
    
        matrix[K, K] L_Omega_Beta_ = lkj_corr_cholesky_rng(K, eta_L);
        
        // Part of the posterior predicted expectation related to unmodeled
        // predictors.
        vector[N_final] mu_0 = 
            K0
            ? (alpha_c_ + ((N_new > 0)?X0_c_new:X0_c) * beta_0_)
            : zeros_vector(N_final);
        
        // Part of the posterior predicted expectation related to modeled
        // predictors. The expected value is 0.
        vector[N_final] mu_1 = zeros_vector(N_final);
        
        mu = mu_0 + mu_1;
        
        // Covariance matrix of group-level regression, reconstructed from
        // variance and Cholesky-factored correlation
        matrix[K, K] Sigma_Beta;
        {
            matrix[K, K] sigma_L = diag_pre_multiply(sigma_Beta_, L_Omega_Beta_);
            Sigma_Beta = sigma_L *  sigma_L';
        }
        
        // Part of the posterior predicted value related to modeled predictors
        array[J] vector[K] Beta_;
        for(j in 1:J)
        {
            Beta_[j] = multi_normal_rng(zeros_K, Sigma_Beta);
        }
        vector[N_final] y_1;
        for(n in 1:N_final)
        {
            y_1[n] = ((N_new > 0)?X_new[n, :]:X[n, :]) * Beta_[group[n]];
        }
        
        y = to_vector(normal_rng(mu_0 + y_1, sigma_y));
    }
}
