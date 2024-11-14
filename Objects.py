from typing import List, Literal
import numpy.typing as npt
import numpy as np

white = np.array([1.,1.,1.])
black = np.array([0.,0.,0.])

def normalize(x):
    return x / np.linalg.norm(x)

class Object:
    def __init__(self, position :List[float], color_type:Literal['mono','gradient','squares'], color_para,
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

    def get_normal(self, point):
        pass

    def get_color(self, point):
        if self._color_type == 'mono':
            return self._color_para

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
    def __init__(self, position :List[float], normal: List[float], color_type:Literal['mono','gradient'], color_para,
                 reflection=.15, diffuse=.75, specular_c=.3, specular_k=50, ):
        super().__init__(position, color_type, color_para, reflection, diffuse, specular_c, specular_k)
        self._normal = np.array(normal)

    def intersect(self, origin, direction):
        dn = np.dot(direction, self._normal)
        if np.abs(dn) < 1e-6:  # 射线与平面几乎平行
            return np.inf  # 交点为无穷远处
        d = np.dot(self._position - origin, self._normal) / dn  # 交点与射线原点的距离（相似三角形原理）
        return d if d > 0 else np.inf

    def get_normal(self, point):
        return self._normal

    def get_color(self, point):
        if self._color_type == 'mono':
            return self._color
        elif self._color_type == 'squares':
            if int(point[0] * 2) % 2== int(point[2] * 2) % 2:
                return white
            else:
                return black

class Sphere(Object):
    def __init__(self, position :List[float], radius: float, color_type:Literal['mono','gradient'], color_para,
                 reflection=.35, diffuse=1., specular_c=.6, specular_k=50, ):
        super().__init__(position, color_type, color_para, reflection, diffuse, specular_c, specular_k)
        self._radius = radius

    def intersect(self, origin, direction):
        oc = self._position - origin
        if (np.linalg.norm(oc) < self._radius) or (np.dot(oc, direction) < 0):
            return np.inf
        l = np.linalg.norm(np.dot(oc, direction))
        m_square = np.linalg.norm(oc) * np.linalg.norm(oc) - l * l
        q_square = self._radius * self._radius - m_square
        return (l - np.sqrt(q_square)) if q_square >= 0 else np.inf

    def get_normal(self, point):
        return normalize(point - self._position)

    def get_color(self, point):
        if self._color_type == 'mono':
            return self._color






