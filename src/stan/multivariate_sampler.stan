/*
Multivariate linear model with normal likelihood and robust priors.

Note that each outcome is multivariate: there are N outcomes of shape R, not
N_1 + N_2 + … + N_R outcomes. The correlation matrix has an LKJ prior, see
univariate model for more details.

*/

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
    
    // Outcomes
    array[N] vector[R] y;
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
    
    // Center the predictors
    vector[sum(K_c)] X_bar;
    matrix[N, sum(K_c)] X_c;
    for(r in 1:R)
    {
        matrix[N, K[r]] X_ = X[, K_begin[r]:K_end[r]];
        vector[K_c[r]] X_bar_ = center_columns(X_, N, K[r]);
        matrix[N, K_c[r]] X_c_ = center(X_, X_bar_, N, K[r]);
        
        X_bar[K_c_begin[r]:K_c_end[r]] = X_bar_;
        X_c[, K_c_begin[r]:K_c_end[r]] = X_c_;
    }
}
 
#include multivariate_parameters.stan

model
{
    alpha_c ~ student_t(3, mu_alpha, sigma_alpha);
    beta ~ student_t(3, 0, sigma_beta);
    sigma ~ exponential(lambda_sigma);
    
    // NOTE:
    // Exception: lkj_corr_cholesky_lpdf: Random variable[2] is 0, but must be positive!
    // https://github.com/stan-dev/math/blob/master/stan/math/prim/prob/lkj_corr_cholesky_lpdf.hpp#L25
    L ~ lkj_corr_cholesky(eta_L);
    matrix[R, R] Sigma = diag_pre_multiply(sigma, L);
    
    array[N] vector[R] mu;
    for(r in 1:R)
    {
        matrix[N, K_c[r]] X_c_ = X_c[, K_c_begin[r]:K_c_end[r]];
        vector[K_c[r]] beta_ = beta[K_c_begin[r]:K_c_end[r]];
        
        for(n in 1:N)
        {
            mu[n, r] = alpha_c[r] + dot_product(X_c_[n], beta_);
        }
    }
    
    y ~ multi_normal_cholesky(mu, Sigma);
}

generated quantities
{
    // Non-centered intercept
    vector[R] alpha;
    for(r in 1:R)
    {
        vector[K_c[r]] X_bar_ = X_bar[K_c_begin[r]:K_c_end[r]];
        vector[K_c[r]] beta_ = beta[K_c_begin[r]:K_c_end[r]];
        alpha[r] = alpha_c[r] - dot_product(X_bar_, beta_);
    }
    
    corr_matrix[R] Sigma = multiply_lower_tri_self_transpose(L);
}
