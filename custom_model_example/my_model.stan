data
{
    int<lower=1> N;
    vector[N] x, y, z;
}

parameters
{
    real a, b_x, b_y;
    real<lower=1.2e-38, upper=3.4e+38> sigma;
}

model
{
    a ~ normal(0, 1);
    b_x ~ normal(0, 1);
    b_y ~ normal(0, 1);
    
    sigma ~ exponential(1);
    
    z ~ normal(a + b_x * x + b_y * y, sigma);
}

generated quantities
{
    array[N] real z_hat = normal_rng(a + b_x * x + b_y * y, sigma);
}
