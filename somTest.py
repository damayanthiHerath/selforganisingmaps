from som import som
import numpy as numpy
input = numpy.array(
         [[1., 0., 0.],
          [1., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

somCol = 3
somRow = 3
radius = 2

som = som(input,12,4,somCol,somRow,2)
ans = som.trainmodel()
print 'trained model is',ans
