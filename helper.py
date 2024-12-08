from numba import njit
import numpy as np

@njit
def normalize(x):
    return x / np.linalg.norm(x)


@njit
def single_dot(array1, array2):
    product = []
    for i in range(array1.shape[0]):
        product.append(np.dot(array1[i,:],np.transpose(array2[i,:])))
    return np.array(product)


def single_dot2(array1, array2):
    product = []
    for i in range(array1.shape[0]):
        product.append(np.dot(array1[i,:],array2[i]))
    return np.array(product)

@njit
def norm_by_row(array):
    for i, row in enumerate(array):
        array[i] = normalize(row)
    return array