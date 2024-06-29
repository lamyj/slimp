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
    
    // Number of new outcomes to predict
    int<lower=0> N_new;
    // New predictors
    matrix[N_new, K0] X0_new;
    matrix[N_new, K] X_new;
}

transformed data
{
    // Center the unmodeled predictors around the *original* predictors
    vector[K0-1] X0_bar = center_columns(X0, N, K0);
    matrix[N, K0-1] X0_c = center(X0, X0_bar, N, K0);
    matrix[N_new, K0-1] X0_c_new = center(X0_new, X0_bar, N_new, K0);
    
    // Final number of observations to generate.
    int N_final = (N_new>0)?N_new:N;
}

#include multilevel/parameters.stan

generated quantities
{
    // Expected value and draws of the posterior predictive distribution
    vector[N_final] mu;
    vector[N_final] y;
    {
        // Part of the posterior predicted expectation related to unmodeled
        // predictors.
        vector[N_final] mu_0 = alpha_c + ((N_new > 0)?X0_c_new:X0_c) * beta;
        
        // Part of the posterior predicted expectation related to modeled
        // predictors
        vector[N_final] mu_1;
        for(n in 1:N_final)
        {
            mu_1[n] = ((N_new > 0)?X_new[n, :]:X[n, :]) * Beta[group[n]];
        }
        
        mu = mu_0 + mu_1;
        
        // Covariance matrix of group-level regression, reconstructed from
        // variance and Cholesky-factored correlation
        matrix[K, K] Sigma_Beta;
        {
            matrix[K, K] sigma_L = diag_pre_multiply(sigma_Beta, L_Omega_Beta);
            Sigma_Beta = sigma_L *  sigma_L';
        }
        
        // Part of the posterior predicted value related to modeled predictors
        array[J] vector[K] B = multi_normal_rng(Beta, Sigma_Beta);
        vector[N_final] y_1;
        for(n in 1:N_final)
        {
            y_1[n] = ((N_new > 0)?X_new[n, :]:X[n, :]) * B[group[n]];
        }
        
        y = to_vector(normal_rng(mu_0 + y_1, sigma_y));
    }
    
}
