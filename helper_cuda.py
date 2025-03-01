# Copyright H.Zhao @ THU

from numba import cuda
import numpy as np
from numba.cuda.cudadrv.devicearray import DeviceNDArray

THREADS_PER_BLOCK = 256


@cuda.jit
def single_dot_core(A, B, C):
    i = cuda.grid(1)  # Get the thread ID for the current execution
    if i < A.shape[0]:  # Ensure within bounds
        dot_product = 0
        for j in range(A.shape[1]):  # Compute dot product
            dot_product += A[i, j] * B[i, j]
        C[i] = dot_product

def single_dot(A,B):
    blocks_per_grid = (A.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    C = np.zeros((A.shape[0],1), dtype=np.float32)
    A_d = cuda.to_device(A)
    B_d = cuda.to_device(B)
    C_d = cuda.to_device(C)
    # Define the number of threads and blocks
    # Launch the kernel
    single_dot_core[blocks_per_grid, THREADS_PER_BLOCK](A_d, B_d, C_d)
    # Copy the result back to the CPU
    C = C_d.copy_to_host()
    return C

def single_dot_d(A_d, B_d):
    blocks_per_grid = (A_d.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    C = np.zeros((A_d.shape[0],1), dtype=np.float32)
    C_d = cuda.to_device(C)
    single_dot_core[blocks_per_grid, THREADS_PER_BLOCK](A_d, B_d, C_d)
    return C_d

@cuda.jit
def norm_by_row_core(array_d, result_d):
    i = cuda.grid(1)
    if i < array_d.shape[0]:
        slice_d = array_d[i]
        norm = 0
        for j in slice_d:
            norm += j ** 2
        norm = norm **0.5
        for j in range(len(slice_d)):
            result_d[i,j] = array_d[i,j]/norm

def norm_by_row(A):
    blocks_per_grid = (A.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    C = np.empty_like(A, dtype=np.float32)
    A_d = cuda.to_device(A)
    C_d = cuda.to_device(C)
    # Define the number of threads and blocks
    # Launch the kernel
    norm_by_row_core[blocks_per_grid, THREADS_PER_BLOCK](A_d, C_d)
    # Copy the result back to the CPU
    C = C_d.copy_to_host()
    return C


def norm_by_row_d(A_d):
    blocks_per_grid = (A_d.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    C = np.empty_like(A_d, dtype=np.float32)
    C_d = cuda.to_device(C)
    norm_by_row_core[blocks_per_grid, THREADS_PER_BLOCK](A_d, C_d)
    return C_d

@cuda.jit
def set_negative_to_zero_core(array_d):
    i = cuda.grid(1)
    if i < array_d.shape[0]:
        if array_d[i,0] < 0.:
            array_d[i,0] = 0.

def set_negative_to_zero(A_d):
    blocks_per_grid = (A_d.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    set_negative_to_zero_core[blocks_per_grid, THREADS_PER_BLOCK](A_d)
    return A_d

@cuda.jit
def add_arrays_kernel(array1_d, array2_d, result_d):
    i = cuda.grid(1)
    if i < array1_d.shape[0]:
        for j in range(3):  # since each array has 3 columns
            result_d[i, j] = array1_d[i, j] + array2_d[i, j]

def add_arrays(array1_d, array2_d):
    # Ensure input arrays are DeviceNDArray
    result_d = cuda.device_array_like(array1_d)
    blocks_per_grid = (array1_d.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    add_arrays_kernel[blocks_per_grid, THREADS_PER_BLOCK](array1_d, array2_d, result_d)
    return result_d

@cuda.jit
def normalize_a(a, c):
    norm = 0
    for j in range(3):
        norm += a[j] ** 2
    norm = norm ** 0.5
    for j in range(3):
        c[j] = a[j]/ norm

@cuda.jit
def calculate_color_sphere_kernel(c_grid_d: DeviceNDArray, directions_d: DeviceNDArray, origins_d: DeviceNDArray,
                                  distances_d: DeviceNDArray, intensities_d: DeviceNDArray, color_d: DeviceNDArray, ambient: float,
                                  light_point_d: DeviceNDArray, light_color_d: DeviceNDArray, position: DeviceNDArray,
                                  diffuse: float, specular_c: float, specular_k: float):
    i = cuda.grid(1)
    if i<c_grid_d.shape[0]:
        origin = origins_d[i]
        direction = directions_d[i]
        distance = distances_d[i]
        intensity = intensities_d[i]
        intersect = cuda.device_array(3, dtype=np.float32)
        # eliminate inf terms
        if distance == np.inf:
            for j in range(3):
                c_grid_d[i][j]=0.
        else:
            # calculate intersect point
            for j in range(3):
                intersect[j] = distance[i] * direction[j] +origin[j]
            #calculate normal vector
            normal = cuda.device_array(3, dtype=np.float32)
            normalize_a((intersect - position),normal)
            #calculate PL
            PL = cuda.device_array(3, dtype=np.float32)
            product = cuda.device_array(3, dtype=np.float32)
            for j in range(3):
                product[j] = light_point_d[j] - intersect[j]
            normalize_a(product, PL)
            PO = cuda.device_array(3, dtype=np.float32)
            for j in range(3):
                product[j] = light_point_d[j] - intersect[j]
            normalize_a(product, PO)
            dot_product = 0
            for j in range(3):  # Compute dot product
                dot_product += normal[j] * PL[j]
            if dot_product<0:
                dot_product = 0
            for j in range(3):
                c_grid_d[i][j] += diffuse * dot_product * light_color_d[j] * color_d[j]



def calculate_color_sphere(scene, directions, origins, distances, intensities, color, position):
    height, width = scene.get_dimensions()
    c_grid_d = cuda.device_array((height*width, 3), dtype=np.float32)
    ambient_color = color * scene.get_ambient()
    light_point = scene.get_light_point()
    light_color = scene.get_light_color()



