functions
{

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

// Return a vector from a ragged array
vector get_vector(vector v_flat, array[] int N, int r)
{
    return segment(v_flat, 1+sum(N[1:r-1]), N[r]);
}

// Update the ragged array and return a copy
// NOTE: functions cannot modify their arguments, even in C++: stanc
// declares all arguments const.
vector update_vector(vector v_flat, array[] int N, int r, vector v)
{
    vector[size(v_flat)] v_flat_copy = v_flat;
    
    int begin = 1+sum(N[1:r-1]);
    v_flat_copy[begin:begin+size(v)-1] = v;
    
    return v_flat_copy;
}

// Return a matrix from a ragged array
matrix get_matrix(vector m_flat, int N, array[] int K, int r)
{
    array[size(K)] int sizes = to_int(to_array_1d(N * to_vector(K)));
    // NOTE: last argument means a row-major order.
    return to_matrix(
        segment(m_flat, 1+sum(sizes[1:r-1]), sizes[r]), N, K[r], 0);
}

// Update the ragged array and return a copy
vector update_matrix(vector m_flat, int N, array[] int K, int r, matrix m)
{
    vector[size(m_flat)] m_flat_copy = m_flat;
    
    array[size(K)] int sizes = to_int(to_array_1d(N * to_vector(K)));
    int begin = 1+sum(sizes[1:r-1]);
    m_flat_copy[begin:begin+size(m)-1] = to_vector(m');
    
    return m_flat_copy;
}

}
