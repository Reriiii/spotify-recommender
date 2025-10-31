import math

def standardize(X):
    n_rows = len(X)
    n_cols = len(X[0])
    mean = [0.0] * n_cols
    std = [0.0] * n_cols

    for j in range(n_cols):
        s = 0
        for i in range(n_rows):
            s += X[i][j]
        mean[j] = s / n_rows

    for j in range(n_cols):
        s = 0
        for i in range(n_rows):
            s += (X[i][j] - mean[j]) ** 2
        std[j] = math.sqrt(s / n_rows)
        if std[j] == 0:
            std[j] = 1.0

    X_std = []
    for i in range(n_rows):
        row = [(X[i][j] - mean[j]) / std[j] for j in range(n_cols)]
        X_std.append(row)

    return X_std, mean, std

def covariance_matrix(X):
    n = len(X)
    m = len(X[0])
    cov = [[0.0]*m for _ in range(m)]
    
    for i in range(m):
        for j in range(m):
            s = 0.0
            for k in range(n):
                s += X[k][i] * X[k][j]
            cov[i][j] = s / (n-1)
    return cov

def mat_vec_mult(mat, vec):
    result = [0.0]*len(vec)
    for i in range(len(mat)):
        s = 0.0
        for j in range(len(vec)):
            s += mat[i][j] * vec[j]
        result[i] = s
    return result

def normalize(vec):
    norm = math.sqrt(sum(x*x for x in vec))
    if norm == 0:
        return vec
    return [x/norm for x in vec]

def power_iteration(mat, max_iter=1000, tol=1e-10):
    n = len(mat)
    b_k = [1.0]*n
    b_k = normalize(b_k)

    for _ in range(max_iter):
        b_k1 = mat_vec_mult(mat, b_k)
        b_k1 = normalize(b_k1)
        diff = sum(abs(b_k1[i]-b_k[i]) for i in range(n))
        b_k = b_k1
        if diff < tol:
            break

    Av = mat_vec_mult(mat, b_k)
    lambda_val = sum(Av[i]*b_k[i] for i in range(n))
    return lambda_val, b_k

def pca(X, n_components=None):
    X_std, mean, std = standardize(X)
    cov = covariance_matrix(X_std)

    components = []
    eigenvalues = []

    cov_copy = [row[:] for row in cov]
    k = n_components if n_components is not None else len(cov)

    for _ in range(k):
        val, vec = power_iteration(cov_copy)
        eigenvalues.append(val)
        components.append(vec)

        n = len(cov_copy)
        for i in range(n):
            for j in range(n):
                cov_copy[i][j] -= val * vec[i] * vec[j]

    X_pca = []
    for row in X_std:
        new_row = []
        for vec in components:
            proj = sum(row[i]*vec[i] for i in range(len(row)))
            new_row.append(proj)
        X_pca.append(new_row)

    return X_pca, eigenvalues, components
