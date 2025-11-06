import numpy as np

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0  
    X_std = (X - mean) / std
    return X_std, mean, std

def covariance_matrix(X):
    n_samples = X.shape[0]
    cov = (X.T @ X) / (n_samples - 1)
    return cov

def power_iteration(mat, max_iter=1000, tol=1e-10):
    n = mat.shape[0]
    b_k = np.ones(n)
    b_k = b_k / np.linalg.norm(b_k)

    for _ in range(max_iter):
        b_k1 = mat @ b_k
        b_k1 = b_k1 / np.linalg.norm(b_k1)
        diff = np.sum(np.abs(b_k1 - b_k))
        if diff < tol:
            break
        b_k = b_k1

    lambda_val = np.dot(b_k, mat @ b_k)
    return lambda_val, b_k

def pca(X, n_components=None):
    X_std, mean, std = standardize(X)
    cov = covariance_matrix(X_std)

    components = []
    eigenvalues = []
    cov_copy = np.copy(cov)

    k = n_components if n_components is not None else cov.shape[0]

    for _ in range(k):
        val, vec = power_iteration(cov_copy)
        eigenvalues.append(val)
        components.append(vec)

        cov_copy -= val * np.outer(vec, vec)

    components = np.array(components)
    eigenvalues = np.array(eigenvalues)

    X_pca = X_std @ components.T

    return X_pca, eigenvalues, components
