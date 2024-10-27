import numpy

import Renderer

renderer = Renderer.Renderer(100,100)
renderer.set_position_delta(numpy.array((3.,3.,2.)))
renderer.set_orientation(numpy.array((0.,0.,0.1)))

image = renderer.render()

