import numpy
from matplotlib import pyplot as plt
import time
import Objects
import Scene


myScene = Scene.Scene3d(300, 500)
sphere1 = Objects.Sphere([.75,.1,1.],.6,'mono',[.8,.3,0.])

sphere2 = Objects.Sphere([-.3, .1, .2], .3, 'map','rainbow')
sphere3 = Objects.Sphere([-2.75, .1, 3.5], .6, 'mono',[.1, .572, .184])
plane = Objects.Plane([0., -0.5, 0.], [0., 1., 0.],'mono',[.3,.5,.5])  # 平面上一点的位置，法向量

myScene.add_objects([sphere1, sphere3, plane])
myScene.set_camera_position([0.,.5,-.5])
myScene.set_camera_orientation([0,-10])
myScene.create_ray_grid()
# myScene.set_reflection(False)
a = time.time()
image = myScene.render()
print(time.time() - a)
plt.figure()
plt.imshow(image)
plt.axis('off')
plt.show()

plt.imsave('ray-traced.png', image)
