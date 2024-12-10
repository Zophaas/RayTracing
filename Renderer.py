import numpy
import numpy as np
from numpy.ma.core import zeros_like, transpose
from tqdm import tqdm

import helper_cuda as helper


def normalize(x):
    return x / np.linalg.norm(x)

def generate_transpose(theta:float,phi:float,mode='r')->np.array:
    if mode == 'd':
        theta = np.radians(theta)
        phi = np.radians(phi)
    # Rotation matrix around the y-axis (for theta)
    R_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)]])

    # Rotation matrix around the x-axis (for phi)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(phi), np.sin(phi)],
                    [0, -np.sin(phi), np.cos(phi)]])

    # The total rotation matrix is the product of the two rotations
    R = R_y @ R_x  # Apply z-rotation followed by y-rotation
    return R



def enact_transpose(scene):
    camera_ori = scene.get_camera_orientation()
    directions_ori = scene.get_ray_grid()
    R = generate_transpose(camera_ori[0],camera_ori[1],mode='d') # 生成旋转变换矩阵
    directions = np.transpose(np.dot(R, directions_ori))
    directions = helper.norm_by_row(directions)
    return directions


def intersect_color(scene, origin:numpy.array, direction:numpy.array, intensity:float, light_point, light_color, ambient):
    min_distance = np.inf
    for obj in scene.objects:
        current_distance = obj.intersect(origin, direction)
        if current_distance < min_distance:
            min_distance, nearest_object = current_distance, obj
    if (min_distance == np.inf) or (intensity < 0.01):
        return np.array([0., 0., 0.])
    obj = nearest_object
    intersect = origin + direction * min_distance
    color = obj.get_color(intersect)
    normal = obj.get_normal(intersect)


    PL = normalize(light_point - intersect)
    PO = normalize(origin - intersect)
    c=ambient * color
    l = [obj_test.intersect(intersect + normal * .0001, PL) for obj_test in scene.objects if obj!=obj_test]
    if not (l and min(l) < np.linalg.norm(light_point - intersect)):
        c += obj.get_diffuse() * max(np.dot(normal, PL), 0) * color * light_color
        c += obj.get_specular_c() * max(np.dot(normal, normalize(PL + PO)), 0) ** obj.get_specular_k() * light_color

    reflect_ray = direction - 2 * np.dot(direction, normal) * normal  # 计算反射光线
    c += obj.get_reflection() * intersect_color(scene, intersect + normal * .0001, reflect_ray, obj.get_reflection() * intensity, light_point, light_color, ambient)

    return np.clip(c, 0, 1)


def render(scene):
    height, width = scene.get_dimensions()
    camera_pos = scene.get_camera_position()
    camera_ori = scene.get_camera_orientation()
    img = np.zeros((height, width, 3))
    ratio = float(width) / float(height)
    screen = (-1., -1. / ratio + .25, 1., 1. / ratio + .25)
    R = generate_transpose(camera_ori[0],camera_ori[1],mode='d') # 生成旋转变换矩阵
    light_point = scene.get_light_point()
    light_color = scene.get_light_color()
    ambient = scene.get_ambient()
    for i, x in enumerate(tqdm(np.linspace(screen[0], screen[2], width))):
        for j, y in enumerate(np.linspace(screen[1], screen[3], height)):
            ray_direction = R @ np.array([x, y, (1. if abs(camera_ori[0])<=90 else -1.)])
            pixel_color = intersect_color(scene, camera_pos, normalize(ray_direction), 1,light_point,light_color,ambient)
            img[j,i,:]=pixel_color
    return np.flip(img,0)

def render_batch(scene):
    height, width = scene.get_dimensions()
    img = np.zeros((height, width, 3))
    directions = enact_transpose(scene)
    intensities = np.ones((directions.shape[0]))
    origins = np.array(scene.get_camera_position())*np.ones((directions.shape[0],1))
    distance_all = []
    distance_all_unshaped = []
    for obj in scene.objects:
        distances_unshaped = (obj.intersect_batch(origins, directions))
        distances = distances_unshaped.reshape(height,width)
        distance_all.append(distances)
        distance_all_unshaped.append(distances_unshaped)
    min_distance = np.min(distance_all, axis=0)
    for i, distances in enumerate(distance_all):
        distances[distances > min_distance] = np.inf
        distance_all[i] = distances

    for i, obj in enumerate(scene.objects):
        img+=obj.get_color_batch(scene, directions, origins, distance_all_unshaped[i], intensities)

    return np.flip(img, 0)






def test_a(scene):
    directions = []
    height, width = scene.get_dimensions()
    camera_pos = scene.get_camera_position()
    camera_ori = scene.get_camera_orientation()
    img = np.zeros((height, width, 3))
    ratio = float(width) / float(height)
    screen = (-1., -1. / ratio + .25, 1., 1. / ratio + .25)
    R = generate_transpose(camera_ori[0],camera_ori[1],mode='d') # 生成旋转变换矩阵
    light_point = scene.get_light_point()
    light_color = scene.get_light_color()
    ambient = scene.get_ambient()
    for i, x in enumerate(tqdm(np.linspace(screen[0], screen[2], width))):
        for j, y in enumerate(np.linspace(screen[1], screen[3], height)):
            ray_direction = R @ np.array([x, y, (1. if abs(camera_ori[0])<=90 else -1.)])
            directions.append(ray_direction)
    return np.array(directions)

