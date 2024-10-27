from typing import Annotated

from PIL import Image
import numpy as np
import numpy.typing as npt


class Renderer:
    def __init__(self, height:int, width:int, init_position = np.zeros(3), init_orient = np.zeros(3), do_blur = False,
                 to_ratio = True, reflection = True):
        """
        初始化渲染器实例
        :param height: 图像高度
        :param width: 图像宽度
        :param init_position: 相机初始位置, 顺序为[x,y,z] (默认[0,0,0])
        :param init_orient: 相机倾角, 顺序为[pitch,yaw,roll] (默认[0,0,0])
        :param do_blur: 是否高斯模糊(暂不支持)
        :param to_ratio: 是否等比例缩放画面.若等比例,调整画面大小时将高度优先
        :param reflection: 物体表面是否反光
        """
        self._height = height
        self._width = width
        self._do_blur = do_blur
        self._to_ratio = to_ratio
        self._position = init_position
        self._orient = init_orient
        self._reflection = reflection

        self._ratio = self._height / self._width

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


    def set_blur(self, do_blur = False)->None:
        """
        设置是否背景高斯模糊(该功能暂不支持)
        :param do_blur: 是否模糊
        """

    def set_to_ratio(self, to_ratio = True)->None:
        """
        设置是否等比例调整宽与高
        :param to_ratio:
        :return:
        """

    def set_reflection(self, reflection = True)->None:
        """
        设置物体表面是否反光
        :param reflection:
        """

    def set_position(self, position :Annotated[npt.NDArray[np.float64],(3,)])->npt.NDArray[np.float64]:
        """
        设置相机绝对位置
        :param position: 相机的绝对位置
        :return: 当前相机绝对位置
        """
        return self._position

    def set_position_delta(self, position :Annotated[npt.NDArray[np.float64],(3,)])->npt.NDArray[np.float64]:
        """
        设置相机位置变动
        :param position: 相机位置变化量
        :return: 当前相机绝对位置
        """
        return self._position

    def set_orientation(self, orient: Annotated[npt.NDArray[np.float64],(3,)])->npt.NDArray[np.float64]:
        """
        设置相机绝对倾角
        :param orient: 相机绝对倾角
        :return: 当前相机绝对倾角(应与输入相同)
        """
        return self._orient

    def render(self)->Image.Image:
        """
        渲染最终效果
        :return: 渲染出的效果图
        """
        return Image.new('RGBA', (self._width, self._height), (0, 0, 0, 0))