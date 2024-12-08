from numba import cuda
import numpy as np
from scipy.cluster.hierarchy import single


@cuda.jit
def single_dot_core(A, B, C):
    i = cuda.grid(1)  # Get the thread ID for the current execution
    if i < A.shape[0]:  # Ensure within bounds
        dot_product = 0
        for j in range(A.shape[1]):  # Compute dot product
            dot_product += A[i, j] * B[i, j]
        C[i] = dot_product

def single_dot(A,B):
    threads_per_block = 256
    blocks_per_grid = (A.shape[0] + (threads_per_block - 1)) // threads_per_block
    C = np.zeros(A.shape[0], dtype=np.float32)
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)
    # Define the number of threads and blocks
    # Launch the kernel
    single_dot_core[blocks_per_grid, threads_per_block](d_A, d_B, d_C)
    # Copy the result back to the CPU
    C = d_C.copy_to_host()
    return C
