from typing import Annotated

import numpy
from PIL import Image
import numpy as np
import numpy.typing as npt

import Objects
from Objects import Object
import Renderer


class Scene3d:
    def __init__(self, height:int, width:int, init_camera_position = np.array([0.,.5,-1.]),
                 init_camera_orient = np.array([0.,0.]), light_point=np.array([5., 5., -10.]),
                 light_color=np.array([1., 1., 1.]) ,ambient = 0.05, do_blur = False, to_ratio = True, reflection = True,
                 enable_gpu=False):
        """
        初始化渲染器实例
        :param light_point:
        :param light_color:
        :param height: 图像高度
        :param width: 图像宽度
        :param init_camera_position: 相机初始位置, 顺序为[x,y,z] (默认[0,0,0])
        :param init_camera_orient: 相机倾角, 顺序为[pitch,yaw,roll] (默认[0,0,0])
        :param do_blur: 是否高斯模糊(暂不支持)
        :param to_ratio: 是否等比例缩放画面.若等比例,调整画面大小时将高度优先
        :param reflection: 物体表面是否反光
        """

        self._height = height
        self._width = width
        self._do_blur = do_blur
        self._to_ratio = to_ratio
        self._position = init_camera_position
        self._orient = init_camera_orient
        self._reflection = reflection
        self.objects:[Objects.Object] = []
        self._ratio = self._height / self._width
        self._light_point = light_point
        self._light_color = light_color
        self._ambient = ambient
        self.enable_gpu = enable_gpu

    def set_dimensions(self, height, width) -> None:
        """
        设置渲染结果大小,当to_ratio=True时高度优先
        :param height: 高度(单位:像素)
        :param width: 宽度(单位:像素)
        """
        # 等比例时高度优先
        self._height = height
        if self._to_ratio:
            self._width = width * self._ratio
        else:
            self._width = width

    def get_dimensions(self):
        return self._height, self._width


    def set_blur(self, do_blur = False)->None:
        """
        设置是否背景高斯模糊(该功能暂不支持)
        :param do_blur: 是否模糊
        """
        self._do_blur = do_blur

    def set_to_ratio(self, to_ratio = True)->None:
        """
        设置是否等比例调整宽与高
        :param to_ratio:
        :return:
        """
        self._to_ratio = to_ratio

    def set_reflection(self, reflection = True)->None:
        """
        设置物体表面是否反光
        :param reflection:
        """
        self._reflection = reflection
        for obj in self.objects:
            obj.set_reflection(1 if reflection else 0)

    def set_camera_position(self, position:numpy.array)->npt.NDArray[np.float64]:
        """
        设置相机绝对位置
        :param position: 相机的绝对位置
        :return: 当前相机绝对位置
        """
        self._position = position
        return self._position

    def set_position_delta(self, position:numpy.array)->npt.NDArray[np.float64]:
        """
        设置相机位置变动
        :param position: 相机位置变化量
        :return: 当前相机绝对位置
        """
        self._position = self._position + position
        return self._position

    def get_camera_position(self):
        return self._position

    def set_camera_orientation(self, orient:numpy.array)->npt.NDArray[np.float64]:
        """
        设置相机绝对倾角
        :param orient: 相机绝对倾角
        :return: 当前相机绝对倾角(应与输入参数相同)
        """
        self._orient = orient
        return self._orient

    def use_gpu(self):
        self.enable_gpu = True

    def get_camera_orientation(self):
        return self._orient

    def get_light_point(self):
        return self._light_point

    def get_light_color(self):
        return self._light_color

    def get_ambient(self):
        return self._ambient

    def add_object(self, object3d :Object)->int:
        if object3d in self.objects:
            return -1
        if not self._reflection:
            object3d.set_reflection(0)
        self.objects.append(object3d)

    def add_objects(self, objects:[Objects.Object]):
        for obj in objects:
            if not obj in self.objects:
                self.objects.append(obj)


    def render(self)->Image.Image:
        """
        渲染最终效果
        :return: 渲染出的效果图
        """
        img = Renderer.render(self)
        return img
