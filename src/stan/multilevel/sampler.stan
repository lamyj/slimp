/*
Data Analysis Using Regression and Multilevel/Hierarchical Models
Gelman and Hill
Cambridge University Press, 2007

The generic form of multilevel linear model is given by equations 13.8 and 13.7.
Let N be the number of observations, K⁰ the number of unmodeled (i.e. which do
not vary by group) individual-level predictors, K the number of modeled (i.e.
which vary by group) individual-level predictors, J the number of groups and L
the number of group-level predictors. We thus form X⁰ the N×K⁰ matrix of
unmodeled individual-level predictors, X the N×K matrix of modeled
individual-level predictors, and G the L×K matrix of group-level predictors. We
also define β⁰ the K⁰ vector of unmodeled individual-level coefficients, B the
J×K matrix of modeled individual-level coefficients and U the J×L matrix of
group-level predictors. Additional terms are σ²_y the individual-level variance
and Σ_B the group-level covariance matrix. The model is hence written as:

y_n ~ norm(X⁰_n β⁰ + X_n B_j[n], σ²_y)
B_j ~ norm(U_j G, Σ_B)

In lme4, no group-level predictor is present, and the modeled invidual-level
coefficients are centered on 0. This gives a simpler model:

y_n ~ norm(X⁰_n β⁰ + X_n B_j[n], σ²_y)
B_j ~ norm(0, Σ_B)

Regarding the implementation, β⁰ is split in an intercept term and 0-centered
non-intercept terms for an easier interpretation (cf. univariate model), and
Σ_B is decomposed in a variance vector and a Cholesky-factored correlation
matrix.

See also:
- https://occasionaldivergences.com/posts/stan-hierarchical/
- https://occasionaldivergences.com/posts/non-centered/
- https://mc-stan.org/docs/stan-users-guide/regression.html#multivariate-hierarchical-priors.section
*/

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
    
    // Observations
    vector[N] y;
    // Matrix of unmodeled individual-level predictors
    matrix[N, K0] X0;
    // Matrix of modeled individual-level predictors
    matrix[N, K] X;
    
    // Map from an observation to its group
    array[N] int<lower=1, upper=J> group;
    
    // Location and scale of the intercept prior
    real mu_alpha, sigma_alpha;
    
    // Scale of the non-intercept unmodeled coefficients priors (location is 0)
    vector<lower=0>[K0-1] sigma_beta;
    
    // Scale of the individual-level variance prior
    real<lower=0> lambda_sigma_y;
    
    // Scale of the group-level variance prior
    real<lower=0> lambda_sigma_Beta;
    
    // Parameter of the LKJ distribution
    real<lower=1> eta_L;
    
    // NOTE: additional groups can be added by renaming X (resp. group) to X1
    // (resp. group1) and by defining X2, X3, etc. (resp. group2, group3, etc.).
}

transformed data
{
    // Center the predictors
    vector[K0-1] X0_bar = center_columns(X0, N, K0);
    matrix[N, K0-1] X0_c = center(X0, X0_bar, N, K0);
    
    vector[K] zeros_K = zeros_vector(K);
}

#include multilevel/parameters.stan

model
{
    vector[N] X_Beta;
    for(n in 1:N)
    {
        X_Beta[n] = X[n, :] * Beta[group[n]];
    }
    // NOTE: faster than y ~ normal(alpha_c+X0_c*beta + X_Beta, sigma_y)
    y ~ normal_id_glm(X0_c, alpha_c + X_Beta, beta, sigma_y);
    
    alpha_c ~ student_t(3, mu_alpha, sigma_alpha);
    beta ~ student_t(3, 0, sigma_beta);
    sigma_y ~ exponential(lambda_sigma_y);
    
    // NOTE: supposedly faster than computing Sigma_Beta in transformed
    // parameters and using Beta ~ multi_normal(zeros_K, Sigma_Beta)
    Beta ~ multi_normal_cholesky(
        zeros_K, diag_pre_multiply(sigma_Beta, L_Omega_Beta));
    sigma_Beta ~ exponential(lambda_sigma_Beta);
    
    L_Omega_Beta ~ lkj_corr_cholesky(eta_L);
}

generated quantities
{
    // Covariance matrix of group-level regression, reconstructed from variance
    // and Cholesky-factored correlation
    matrix[K, K] Sigma_Beta;
    {
        matrix[K, K] sigma_L = diag_pre_multiply(sigma_Beta, L_Omega_Beta);
        Sigma_Beta = sigma_L *  sigma_L';
    }

    // Non-centered intercept
    real alpha = alpha_c - dot_product(X0_bar, beta);
}
