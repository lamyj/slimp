functions
{

#include functions.stan

}

data
{
    // Number of reponses and of outcomes
    int<lower=1> R, N;
    // Number of predictors for each response
    array[R] int<lower=1> K;
    
    // Predictors
    matrix[N, sum(K)] X;
    
    // Location and scale of the intercept priors
    vector[R] mu_alpha, sigma_alpha;
    
    // Scale of the non-intercept priors (location is 0)
    vector<lower=0>[sum(K)-R] sigma_beta;
    
    // Scale of the variance priors
    vector<lower=0>[R] lambda_sigma;
    
    // Shape of the correlation matrix prior
    real<lower=1> eta_L;
    
    // Number of new outcomes to predict
    int<lower=0> N_new;
    // New predictors
    matrix[N_new, sum(K)] X_new;
}

transformed data
{
    // Numbers of predictors after centering
    array[R] int K_c = to_int(to_array_1d(to_vector(K) - 1));
    
    // Indices of the first and last columns of the predictors for the sub-model
    // in X and X_c
    array[R] int K_begin, K_end, K_c_begin, K_c_end;
    for(r in 1:R)
    {
        if(r == 1)
        {
            K_begin[r] = 1;
            K_c_begin[r] = 1;
        }
        else
        {
            K_begin[r] = K_begin[r-1] + K[r-1];
            K_c_begin[r] = K_c_begin[r-1] + K_c[r-1];
        }
        K_end[r] = K_begin[r] + K[r] - 1;
        K_c_end[r] = K_c_begin[r] + K_c[r] - 1;
    }
    
    // Center the predictors around the *original* predictors
    vector[sum(K_c)] X_bar;
    matrix[N, sum(K_c)] X_c;
    matrix[N_new, sum(K_c)] X_c_new;
    for(r in 1:R)
    {
        matrix[N, K[r]] X_ = X[, K_begin[r]:K_end[r]];
        vector[K_c[r]] X_bar_ = center_columns(X_, N, K[r]);
        matrix[N, K_c[r]] X_c_ = center(X_, X_bar_, N, K[r]);
        
        X_bar[K_c_begin[r]:K_c_end[r]] = X_bar_;
        X_c[, K_c_begin[r]:K_c_end[r]] = X_c_;
        
        if(N_new > 0)
        {
            matrix[N_new, K[r]] X_new_ = X_new[, K_begin[r]:K_end[r]];
            matrix[N_new, K_c[r]] X_c_new_ = center(X_new_, X_bar_, N_new, K[r]);
            X_c_new[, K_c_begin[r]:K_c_end[r]] = X_c_new_;
        }
    }
}

#include multivariate_parameters.stan

generated quantities
{
    // Expected value and draws of the prior predictive distribution
    array[N] vector[R] mu, y;
    
    {
        matrix[R, R] Sigma = diag_pre_multiply(sigma, L);
        
        for(r in 1:R)
        {
            matrix[(N_new > 0)?N_new:N, K_c[r]] X_c_ = 
                (N_new > 0)
                ? X_c_new[, K_c_begin[r]:K_c_end[r]]
                : X_c[, K_c_begin[r]:K_c_end[r]];
            
            vector[K_c[r]] beta_ = beta[K_c_begin[r]:K_c_end[r]];
            
            for(n in 1:N)
            {
                mu[n, r] = alpha_c[r] + dot_product(X_c_[n], beta_);
            }
        }
        
        y = multi_normal_cholesky_rng(mu, Sigma);
    }
}
