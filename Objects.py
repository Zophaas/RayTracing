# Copyright H.Zhao @ THU

import time
from typing import List, Literal
import numpy.typing as npt
import numpy as np
from matplotlib import colormaps as cmps
from numba import cuda
#from helper import single_dot, norm_by_row
from helper_cuda import norm_by_row, single_dot, norm_by_row_d, single_dot_d, set_negative_to_zero, add_arrays


white = np.array([.5,.5,.5])
black = np.array([-.5,-.5,-.5])
allowed_maps = list(cmps)

def normalize(x):
    return x / np.linalg.norm(x)

class Object:
    def __init__(self, position :List[float], color_type:str, color_para,
                 reflection:float, diffuse: float, specular_c: float, specular_k: int):
        self._position = np.array(position)
        self._color_type = color_type
        self._color_para = color_para
        self._reflection = reflection
        self._diffuse = diffuse
        self._specular_c = specular_c
        self._specular_k = specular_k
        self._color = white
        if self._color_type == 'mono':
            self._color = np.array(self._color_para)

    def intersect(self, origin, direction):
        pass

    def intersect_batch(self, origins, directions):
        pass

    def get_normal(self, point):
        pass

    def get_color(self, point):
        if self._color_type == 'mono':
            return self._color_para

    def get_color_batch(self, scene, origins, directions, distances, intensities, reflected_rays):
        pass

    def get_diffuse(self):
        return self._diffuse

    def get_specular_c(self):
        return self._specular_c

    def get_specular_k(self):
        return self._specular_k

    def get_reflection(self):
        return self._reflection

    def set_reflection(self, reflection):
        self._reflection = reflection

class Plane(Object):
    def __init__(self, position :List[float], normal: List[float], color_type:Literal['mono','gradient','squares'], color_para,
                 reflection=.15, diffuse=.75, specular_c=.3, specular_k=50, ):
        super().__init__(position, color_type, color_para, reflection, diffuse, specular_c, specular_k)
        self._normal = np.array(normal)

    def intersect(self, origin, direction):
        dn = np.dot(direction, self._normal)
        if np.abs(dn) < 1e-6:  # 射线与平面几乎平行
            return np.inf  # 交点为无穷远处
        d = np.dot(self._position - origin, self._normal) / dn  # 交点与射线原点的距离（相似三角形原理）
        return d if d > 0 else np.inf

    def intersect_batch(self, origins, directions):
        dn = np.einsum('ij,j->i', directions, self._normal)  # Efficient batch dot product
        # Mask for rays that are parallel to the plane
        parallel_mask = np.abs(dn) < 1e-6
        # Compute distances to the plane for each origin
        d = np.einsum('ij,j->i', self._position - origins, self._normal) / dn  # Batch calculation for distances
        # Assign np.inf for parallel rays or rays going in the opposite direction
        d = np.where((d > 0) & ~parallel_mask, d, np.inf)
        return d

    def get_normal(self, point):
        return self._normal

    def get_color(self, point):
        if self._color_type == 'mono':
            return self._color
        elif self._color_type == 'squares':
            if int(point[0] * 2) % 2== int(point[2] * 2) % 2:
                color = white
            else:
                color = black
        return color+white if point[2] > 0 else -color+white     #这几段写的很迷,改之前一定要看清楚

    def get_color_batch(self, scene, directions, origins, distances, intensities, reflected_rays):
        intersects = origins + directions * distances[:, None]
        color = self._color
        ambient = scene.get_ambient()
        light_point = scene.get_light_point()
        light_color = scene.get_light_color()
        height, width = scene.get_dimensions()
        c_grid = np.tile(ambient * color, (height * width, 1))
        intersects_d = cuda.to_device(intersects)
        normals_d = cuda.to_device(np.tile(self._normal,(height*width,1)))
        PL_d = norm_by_row_d(light_point - intersects)
        PO_d = norm_by_row_d(origins - intersects)

        product_d = single_dot_d(normals_d, PL_d)
        product_d = set_negative_to_zero(product_d)
        product = product_d.copy_to_host()

        c_grid += self._diffuse * product * color.T * light_color

        product_d = single_dot_d(normals_d, norm_by_row_d(add_arrays(PL_d, PO_d)))
        product_d = set_negative_to_zero(product_d)
        product = product_d.copy_to_host()
        c_grid += self._specular_c * product ** self._specular_k * light_color
        c_grid[distances == np.inf] = np.array([0, 0, 0])
        if reflected_rays:
            c_grid = c_grid * self._reflection
        c_clipped = np.clip(c_grid.reshape(height, width, 3), 0, 1)
        normals = normals_d.copy_to_host()
        reflect_intensities = self._reflection * intensities
        reflect_directions = directions - 2 * normals * single_dot(directions, normals_d.copy_to_host())

        reflect_intensities[distances == np.inf] = 0
        reflect_directions[distances == np.inf] = np.array([0,0,0])

        reflect_intensities[distances == np.inf] = 0
        reflect_directions[distances == np.inf] = np.array([0,0,0])
        reflect_intensities[distances == -np.inf] = 0
        reflect_directions[distances == -np.inf] = np.array([0,0,0])
        intersects[distances == np.inf] = np.array([0,0,0])
        intersects[distances == -np.inf] = np.array([0,0,0])

        return  c_clipped, intersects, reflect_directions, reflect_intensities

class Sphere(Object):
    def __init__(self, position :List[float], radius: float, color_type:Literal['mono','map'], color_para,
                 reflection=.35, diffuse=1., specular_c=.6, specular_k=50, ):
        super().__init__(position, color_type, color_para, reflection, diffuse, specular_c, specular_k)
        self._radius = radius
        self._diameter = radius*2


    def intersect(self, origin, direction):
        oc = self._position - origin
        if (np.linalg.norm(oc) < self._radius) or (np.dot(oc, direction) < 0):
            return np.inf
        l = np.linalg.norm(np.dot(oc, direction))
        m_square = np.linalg.norm(oc) * np.linalg.norm(oc) - l * l
        q_square = self._radius * self._radius - m_square
        return (l - np.sqrt(q_square)) if q_square >= 0 else np.inf

    def intersect_batch(self, origins, directions):
        oc = self._position - origins  # Calculate vector from ray origin to sphere center
        l = single_dot(oc, directions).reshape(-1)
        oc_norms = np.linalg.norm(oc, axis=1)
        m_squares = oc_norms ** 2 - l ** 2  # Square distance to sphere along the ray
        q_squares = self._radius ** 2 - m_squares  # Radius squared minus m squared to find the quadratic form
        # Distance to intersection points
        distances = np.where(q_squares >= 0, l - np.sqrt(q_squares), np.inf)
        distances[l < 0] = np.inf
        return distances

    def get_normal(self, point):
        return normalize(point - self._position)

    def get_color(self, point):
        if self._color_type == 'mono':
            return self._color
        elif self._color_type == 'map':
            map_type = self._color_para
            z = (point[0]-self._position[0])/self._diameter + 0.5
            if map_type not in allowed_maps:
                cmp = cmps[0]
            else:
                cmp = cmps[map_type]
            t = (np.sin(z * 3) + 1) / 2  # A simple mapping based on the x-coordinate
            return np.array(cmp(t)[:3])

    def get_color_batch(self, scene, directions, origins, distances, intensities, reflected_rays):
        a=time.time()
        intersects = origins + directions * distances[:, None]
        color = self._color
        ambient = scene.get_ambient()
        light_point = scene.get_light_point()
        light_color = scene.get_light_color()
        height, width = scene.get_dimensions()
        c_grid = np.tile(ambient*color,(height*width,1))
        intersects_d = cuda.to_device(intersects)
        # c_grid_d = cuda.to_device(c_grid)
        print('generate grid: '+str(time.time()-a))
        a=time.time()
        normals_d = norm_by_row_d(intersects - self._position)
        PL_d = norm_by_row_d(light_point - intersects)
        PO_d = norm_by_row_d(origins - intersects)
        product_d = single_dot_d(normals_d, PL_d)
        product_d= set_negative_to_zero(product_d)
        print('calculation 1: ' + str(time.time() - a))
        a = time.time()
        product = product_d.copy_to_host()
        print('copy 1: ' + str(time.time() - a))
        a = time.time()
        c_grid += self._diffuse * product * color.T * light_color
        product_d = single_dot_d(normals_d, norm_by_row_d(add_arrays(PL_d,PO_d)))
        product_d = set_negative_to_zero(product_d)
        print('calculation 2: ' + str(time.time() - a))
        a = time.time()
        product = product_d.copy_to_host()
        print('copy 2: ' + str(time.time() - a))
        a = time.time()
        c_grid += self._specular_c * product ** self._specular_k * light_color
        # c_grid = c_grid_d.copy_to_host()
        c_grid[distances == np.inf]=np.array([0,0,0])
        if reflected_rays:
            c_grid = c_grid * self._reflection
        c_clipped = np.clip(c_grid.reshape(height, width, 3), 0, 1)
        print('calculation 3: ' + str(time.time() - a))
        a = time.time()
        normals = normals_d.copy_to_host()
        reflect_intensities = self._reflection * intensities
        reflect_directions = directions - 2 * normals * single_dot(directions, normals_d.copy_to_host())

        reflect_intensities[distances == np.inf] = 0
        reflect_directions[distances == np.inf] = np.array([0,0,0])
        reflect_intensities[distances == -np.inf] = 0
        reflect_directions[distances == -np.inf] = np.array([0,0,0])
        intersects[distances == np.inf] = np.array([0,0,0])
        intersects[distances == -np.inf] = np.array([0,0,0])

        return  c_clipped, intersects, reflect_directions, reflect_intensities

    def get_color_kernel(self, directions_d, origins_d, distances_d, intensities_d):
        pass

    def get_color_wrapper(self, scene, directions, origins, distances, intensities):
        color = self._color
        ambient = scene.get_ambient()
        light_point = scene.get_light_point()
        light_color = scene.get_light_color()
        height, width = scene.get_dimensions()
        c_grid_d = cuda.device_array()





