from neuron import neuron
from gsom import gsom
from gridPosition import gridPosition
import numpy as np

#Test Neuron class
vectorA=[0.1,0.2,0.3]
a = neuron(vectorA,0)
vectorB=[0.2,0.3,0.4]
b=neuron(vectorB,0)
vectorC=[0.3,0.4,0.5]
c= neuron(vectorC,0)
vectorD=[0.5 ,0.6 ,0.7]
d=neuron(vectorD,0)

#Initial neural network



#Initial SOM

som = {}
som[gridPosition(0,0)] = a
som[gridPosition(1,0)] = b
som[gridPosition(0,1)] = c
som[gridPosition(1,1)] = d
for i,value in som.iteritems():
	print value.getweightVector()
#	print len(i.getNeighbours())
#Test GSOM class
input = np.array(
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

gsomOb = gsom(input,0.3,100)
trainedGSOM = gsomOb.trainNetwork()
