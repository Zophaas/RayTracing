from numba import njit
import numpy as np

@njit
def normalize(x):
    return x / np.linalg.norm(x)


@njit
def single_dot(array1, array2):
    array1 = array1.astype(np.float32)
    array2 = array2.astype(np.float32)
    product = np.zeros((array1.shape[0],1))
    for i in range(array1.shape[0]):
        product[i]=np.dot(array1[i,:],np.transpose(array2[i,:]))
    return product
'''
@njit
def single_dot_vect(array, vector):
    product = np.empty_like(array)
    # Iterate over rows of the array
    for i in range(array.shape[0]):
        # Manually compute the dot product of array[i, :] and vector[i]
        for j in range(array.shape[1]):
            product[i,j] += array[i, j] * vector[i]

    return product
'''
@njit
def norm_by_row(array):
    for i, row in enumerate(array):
        array[i] = normalize(row)
    return array