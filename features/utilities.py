import numpy as np
import sklearn.preprocessing as skpreprocess

__author__ = 'Joseph Robinson'


"""
Utilities for feature encodings.
"""
def cov_matrix(data):
    mean_vec = np.mean(data, axis=0)
    return (data - mean_vec).T.dot((data - mean_vec)) / (data.shape[0] - 1)


def normalize(data):
    """
    # normalize and mean shift (i.e., (features - mu(features))/sigma(features))
    :param data:
    :return: normalized data
    """
    return skpreprocess.StandardScaler().fit_transform(data)


def pca(data):
    """
    pca
    :param data:
    :param k:

    :return:
    """
    cov_mat = cov_matrix(data)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    for ev in eig_vecs:
        # eigenvectors define directions of new axis, which should all the same unit length 1
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
    print('Eigenvectors OKAY')
    """ 
    Eigenvector(s) to drop that minimize information loss in lower-dimensional subspace eigenvalues are inspected.
    Eigenvector(s) with smallest eigenvalue capture least information about data distribution (i.e., to be dropped)
    """
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    return eig_pairs
