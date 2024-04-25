// Return the center of the columns of X, with the exception of the first
vector center_columns(matrix X, int N, int K)
{
    vector[K-1] X_bar;
    for(k in 2:K)
    {
        X_bar[k-1] = mean(X[, k]);
    }
    return X_bar;
}

// Center of the columns of X on X_bar, with the exception of the first
matrix center(matrix X, vector X_bar, int N, int K)
{
    matrix[N, K-1] X_c;
    for(k in 2:K)
    {
        X_c[, k-1] = X[, k] - X_bar[k-1];
    }
    return X_c;
}
