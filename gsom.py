import numpy as np
import math as math
from random import randint
from gridPosition import gridPosition
from neuron import neuron
class gsom:

	def __init__(self, input,spreadFactor,maxIterations):

                self.input  = input
                self.maxIterations = maxIterations;
                self.spreadFactor = spreadFactor

	#Method for training the neural network.
	def trainNetwork(self):

		input = self.input
		inputVectorLength = len(input[0,:])
		growingThreshold = -inputVectorLength * np.log(self.spreadFactor)
		learningRateReduction = 0.3 #Refers to alpha in original paper
		learningRate = 4 #Initial learning rate set to a higher value
		print 'The growing threshold value is',growingThreshold
		#Initialization phase.
		maxError = 0;
		#Initialize the weight vectors of 4 nodes with random values.
		gsom = self.initGSOM(inputVectorLength)
		#printing the initial weight vectors.
		#for neuro in gsom:
			#print neuro.getweightVector()

		numIter = len(input[:,0])
		#print 'Number of iterations is equal to number of inputs = ',numIter

		#*************** Growing Phase **********************************

		#Train the network by presenting all the inputs.
		for iter in range (0,numIter):
			#select input instance

			selectedWeightVector = input[iter,:]

			#Get the wining node

			#Select the winning neuron which has the weight vector closest to that of selected input weight vector.
			#Find the indices of minimum eucledian distance element.
			mineuclideanD=np.linalg.norm(selectedWeightVector-gsom[gridPosition(0,0)].getweightVector())
			winningNeuron = neuron([],0)
			winningPosition = gridPosition(0,0)
			#Iterate the GSOM map hashed by the x,y position
			for position,neuro in gsom.iteritems():
				temp  = np.linalg.norm(selectedWeightVector-neuro.getweightVector())

				if(temp <=mineuclideanD):
					winningNeuron = neuro
					mineuclideanD=temp
					winningPosition = position

			#Update weights of the neighbourhood
			#for neighbour in winningNeuron.getNeighbours():
			#If the wining neuron is at (x,y)
			#then the neighbourhood of the winning neurons posses the positions of (x-1,y),(x+1,y),(x,y-1),(x,y+1)
			#print 'x y position is', winningPosition.getXposition(), winningPosition.getYposition()
			winX= winningPosition.getXposition()
			winY = winningPosition.getYposition()
			tempNeuro = gsom.get(gridPosition(winX-1,winY))
                        #The variable isBoundary is used to track whehter the wining node is a boundary node, i.e it has atleast one neighbouring nodeposition free.
			#print gsom.get(gridPosition(winX-1,winY))
                        isWinningNodeBoundary = False
                        #Maintain a list of positions to add new nodes
                        addedNodePositions = []
			if(tempNeuro!= None):
                            weightVectorTemp =  tempNeuro.getweightVector()
                            updatedWeightVector  = weightVectorTemp + (learningRate * (selectedWeightVector - weightVectorTemp))
				#update the weight vector
                            gsom.get(gridPosition(winX-1,winY)).setweightVector(updatedWeightVector)
			else:
                            isWinningNodeBoundary = True
                            addedNodePositions.append(gridPosition(winX-1,winY))
                        tempNeuro = gsom.get(gridPosition(winX+1,winY))
			#print gsom.get(gridPosition(winX-1,winY))
			if(tempNeuro!= None):
                            weightVectorTemp =  tempNeuro.getweightVector()
                            updatedWeightVector  = weightVectorTemp + (learningRate * (selectedWeightVector - weightVectorTemp))
                            gsom.get(gridPosition(winX+1,winY)).setweightVector(updatedWeightVector)
                            #update the weight vector
                        else:
                            isWinningNodeBoundary = True
                            addedNodePositions.append(gridPosition(winX+1,winY))
			tempNeuro = gsom.get(gridPosition(winX,winY-1))
			#print gsom.get(gridPosition(winX-1,winY))
			if (tempNeuro!= None):

				#update the weight vector
                            weightVectorTemp =  tempNeuro.getweightVector()
                            updatedWeightVector  = weightVectorTemp + (learningRate * (selectedWeightVector - weightVectorTemp))
                            gsom.get(gridPosition(winX,winY-1)).setweightVector(updatedWeightVector)
			else:
                            isWinningNodeBoundary = True
                            addedNodePositions.append(gridPosition(winX,winY-1))
			tempNeuro = gsom.get(gridPosition(winX,winY+1))
			#print gsom.get(gridPosition(winX-1,winY))
			if(tempNeuro!= None):

				#update the weight vector
					weightVectorTemp =  tempNeuro.getweightVector()
					updatedWeightVector  = weightVectorTemp + (learningRate * (selectedWeightVector - weightVectorTemp))
					gsom.get(gridPosition(winX,winY+1)).setweightVector(updatedWeightVector)
                        else:
                            isWinningNodeBoundary = True
                            addedNodePositions.append(gridPosition(winX-1,winY))
			#update accumulatedError of the winning neuron
			err = winningNeuron.getAccumulatedError()+mineuclideanD
			winningNeuron.setAccumulatedError(err)
			#print 'wining neuron error',winningNeuron.getAccumulatedError()

			#Update maximum accumulated error in the network.
			if (maxError < err ):
				maxErr = err

			if (err <= growingThreshold):
                                if (isWinningNodeBoundary):
                                    print 'Growing new nodes'
			#Assigning weights to the nodes in the newly added nodes list.

                        for newNodes in addedNodePositions:
                            tempx = newNodes.getXposition()
                            tempY = newNodes.getYposition()
                            temMap= {}

                            newNodesWeights = np.zeros(len(winningNeuron.getweightVector()))
                            #If wining node and added nods are in same x axis
                            if (tempY == winY):

                               #Considering the node positions (a)
                                if (gsom.get(gridPosition(winX-1,winY))!= None):
                                    neighbourNode = gsom.get(gridPosition(winX-1,winY))
                                    newWeightVector = self.getNewWeightVector(neighbourNode,winningNeuron)
                                    temMap[gridPosition(tempx,tempY)] = neuron(newWeightVector,0)

                                elif (gsom.get(gridPosition(winX+1,winY))!= None):
                                    neighbourNode = gsom.get(gridPosition(winX+1,winY))
                                    newWeightVector = self.getNewWeightVector(neighbourNode,winningNeuron)
                                    temMap[gridPosition(tempx,tempY)] = neuron(newWeightVector,0)


                                elif (gsom.get(gridPosition(winX,winY+1))!= None):
                                    neighbourNode = gsom.get(gridPosition(winX,winY+1))
                                    newWeightVector = self.getNewWeightVector(neighbourNode,winningNeuron)
                                    temMap[gridPosition(tempx,tempY)] = neuron(newWeightVector,0)

                                elif (gsom.get(gridPosition(winX,winY-1))!= None):
                                    neighbourNode = gsom.get(gridPosition(winX,winY-1))
                                    newWeightVector = self.getNewWeightVector(neighbourNode,winningNeuron)
                                    temMap[gridPosition(tempx,tempY)] = neuron(newWeightVector,0)

                                #If the wining node and the newly added nodes are on the same y axis.
                                if (tempx == winX):

                                #Considering the node positions (a)
                                    if (gsom.get(gridPosition(winX,winY-1))!= None):
                                        neighbourNode = gsom.get(gridPosition(winX,winY-1))
                                        newWeightVector = self.getNewWeightVector(neighbourNode,winningNeuron)
                                        temMap[gridPosition(tempx,tempY)] = neuron(newWeightVector,0)


                                    elif (gsom.get(gridPosition(winX,winY+1))!= None):
                                        neighbourNode = gsom.get(gridPosition(winX+1,winY))
                                        newWeightVector = self.getNewWeightVector(neighbourNode,winningNeuron)
                                        temMap[gridPosition(tempx,tempY)] = neuron(newWeightVector,0)

                        gsom.update(temMap)
		return gsom







        def initGSOM(self,inputVectorLength):
		s = (2,2,inputVectorLength)

		weightVector= np.zeros(s)
		for num in range (0,2):
			for iter in range (0,2):
				weightVector[num,iter,:] =np.squeeze(np.random.rand(inputVectorLength,1))


		a = neuron(weightVector[0,0,:],0)

		b=neuron(weightVector[0,1,:],0)

		c= neuron(weightVector[1,0,:],0)

		d=neuron(weightVector[1,1,:],0)


		#Initial neural network


		#Initialise GSOM with 4 neurons.
		gsom = {}
		gsom[gridPosition(0,0)] = a
		gsom[gridPosition(1,0)] = b
		gsom[gridPosition(0,1)] = c
		gsom[gridPosition(1,1)] = d

		return gsom

        def getNewWeightVector(self,neighbouringNode,winningNode):
            newWeightVector = []
            neighbourWeights = neighbouringNode.getweightVector()

            winingWeights = winningNode.getweightVector()
            for i in range (0,len(neighbourWeights)):
                if (neighbourWeights[i]>winingWeights[i]):

                    newWeightVector =winingWeights[i] - (neighbourWeights[i]- winingWeights[i])
                else:
                    newWeightVector =winingWeights[i] +( winingWeights[i]- neighbourWeights[i])
            return newWeightVector
