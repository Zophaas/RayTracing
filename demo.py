import numpy
from matplotlib import pyplot as plt

import Objects
import Scene


myScene = Scene.Scene3d(300, 400)
sphere1 = Objects.Sphere([.75,.1,1.],.6,'mono',[.8,.3,0.])

sphere2 = Objects.Sphere([-.3, .01, .2], .3, 'mono',[.0, .0, .9])
sphere3 = Objects.Sphere([-2.75, .1, 3.5], .6, 'mono',[.1, .572, .184])
plane = Objects.Plane([0., -0.5, 0.], [0., 1., 0.],'mono',[.5,.5,.5])  # 平面上一点的位置，法向量

myScene.add_objects([sphere2,sphere1,sphere3,plane])
myScene.set_camera_position([0.,.5,-1.])
myScene.set_camera_orientation([-30,-10])
image = myScene.render()
plt.figure()
plt.imshow(image)
plt.axis('off')
plt.show()

plt.imsave('ray-traced.png', image)
