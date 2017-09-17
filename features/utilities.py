import numpy as np
import sklearn.preprocessing as skpreprocess
from scipy import linalg
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


def substract_mean(data):
    """
    Substracts mean for every row in data, yeilding a mean equal to 0.

    """
    mean = np.mean(data, axis=0)
    return data - mean


def pca(data, dims_rescaled_data=200):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    # m, n = data.shape
    # mean center the data
    data = substract_mean(data)
    # data -= data.mean(axis=0)
    # calculate the covariance matrix
    cov = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric,
    # the performance gain is substantial
    evals, evecs = linalg.eigh(cov)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims_rescaled_data]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T, evals, evecs
