import numpy as numpy
import math as math
from random import randint
class som:
    def __init__(self, input,maxIterations=10,somCol=3, somRow=3):
		self.somCol = somCol
		self.somRow = somRow
                self.input  = input
                self.maxIterations = maxIterations;
    
 
    def trainmodel(self):
        input = self.input
       
        somCol = self.somCol
        somRow = self.somRow
        inputvectorlen = len(input[0,:])
        inputsSize = len(input[:,0])
        #initialise neurons layer.
        somMap = numpy.zeros(shape=(somCol,somRow,inputvectorlen))
        #print somMap

        #Max number of iterations
        maxIterations = self.maxIterations

        # Initial effective width
        sigmaInitial = 5

        # Time constant for sigma
        t1 = maxIterations / numpy.log(sigmaInitial)

        #Initialise matrix to store eucledian distances.
        #euclideanD = numpy.zeros(shape =(somRow, somCol))
       
        # Initialize 10x10 matrix to store neighbourhood functions
        # of each neurons on the map
        neighbourhoodFunctionVal = numpy.zeros(shape =(somRow, somCol));

        # initial learning rate
        learningRateInitial = 0.1;

        #time constant for eta
        t2 = maxIterations;
        #Assign random weight vectors for all the neurons
        for num in range (0,(somRow)):
            for iter in range (0,(somCol)):
                #Squeezed the matrix into an ndarray.
                somMap[num,iter,:] = numpy.squeeze(numpy.random.rand(inputvectorlen,1))
                    
        ##print "Again #printing the som with randomly initialised weight vectors"
        #print somMap

        count = 1;
        while(count < maxIterations):
            
            sigma = sigmaInitial * numpy.exp(-count/t1)
            variance = pow(sigma,2) 	
            eta = learningRateInitial * numpy.exp(-count/t2)
    
            #Prevent eta from falling below 0.01
            if (eta < 0.01):
                eta = 0.01
            #Randomly select a weight vector from the input weight vectors.
            inputIndex = randint(0,inputsSize-1)
            selectedWeightVector = input[inputIndex,:]
          
            #Select the winning neuron which has the weight vector closest to that of selected input weight vector.
            #Find the indices of minimum eucledian distance element.
            mineuclideanD=numpy.linalg.norm(selectedWeightVector-somMap[0,0,:])
            minr=0
            minc=0
            for num in range (0,somRow):
                for iter in range (0,somCol):
                    
                    temp  = numpy.linalg.norm(selectedWeightVector-somMap[num,iter,:])
                    
                    if(temp <=mineuclideanD):
                        minr=num
                        minc=iter
                        mineuclideanD=temp
            #print euclideanD
                   
            #print 'indices are',minr,minc
        
            #compute the neighbourhood function for all the neurons
            #For the winning noe
            for r in range (0,somRow):
                for c in range (0,somCol):
                    if (r == minr & c == minc):  
                        neighbourhoodFunctionVal[r, c] = 1;
                        continue;
                    else:
                        distance = (minr - r)^2 + (minc - c)^2;
                        neighbourhoodFunctionVal[r, c] = numpy.exp(-distance/(2*variance));
            
            #print 'neighbourhood functions are',neighbourhoodFunctionVal
            #Update weights 

            for r in range (0,somRow):
                for c in range (0,somCol):
                    oldWeightVector = somMap[r, c,:]
                    somMap[r, c,:]     = oldWeightVector + eta*neighbourhoodFunctionVal[r, c]*(selectedWeightVector - oldWeightVector)
   
            #Increment the counter
            count +=1
        
        #Return updated map of neurons.
        return somMap
        
