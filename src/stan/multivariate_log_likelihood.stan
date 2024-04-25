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

generated quantities
{
    vector[N] log_likelihood;
    {
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
        
        matrix[R, R] Sigma = diag_pre_multiply(sigma, L);
        
        // TODO: vectorize
        for(i in 1:N)
        {
            log_likelihood[i] = multi_normal_cholesky_lpdf(y[i] | mu[i], Sigma);
        }
    }
}
