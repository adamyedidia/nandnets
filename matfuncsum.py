import numpy as np
import random

def argmax(l):
    bestIndex = -1
    bestVal = -float("Inf")
        
    for i, val in enumerate(l):
        if val > bestVal:
            bestIndex = i
            bestVal = val
            
    return bestIndex

# Takes as input a 784-entry array and displays it as a handwritten digit    
def displayDigit(l):
    indexCounter = 0
    for _ in range(28):
        printString = ""
        for _ in range(28):
            printString += str(int(round(l[indexCounter])))
            indexCounter += 1
        print printString

def invertIfNotZero(x):
    if x == 0.0:
        return 0.0
    else:
        return 1./x
        
def checkThatLinearFuncsAreOrthogonal(rawTrainingData, svdMat):
    for _ in range(100):
        numInputs = len(svdMat)
        
        randomEntry1 = random.randint(0, numInputs-1)
        randomEntry2 = random.randint(0, numInputs-1)
        
        if randomEntry1 != randomEntry2:
            firstFunc = svdMat[randomEntry1]
            secondFunc = svdMat[randomEntry2]
            
       #     firstFunc = np.array([random.random() for _ in range(numInputs)])
            
            runningSum = 0
            for i in rawTrainingData:
 #               print "hi", np.dot(firstFunc,i), np.dot(secondFunc,i)
                runningSum += np.dot(firstFunc,i)*np.dot(secondFunc,i)
                
            print runningSum
        
def checkThatFuncsAreOrthogonal(rawTrainingData, listOfFuncs):
    for _ in range(100):
        numFuncs = len(listOfFuncs)        
        
        randomFuncIndex1 = random.randint(0, numFuncs-1)
        randomFuncIndex2 = random.randint(0, numFuncs-1)
        
        if randomFuncIndex1 != randomFuncIndex2:
            func1 = listOfFuncs[randomFuncIndex1]
            func2 = listOfFuncs[randomFuncIndex2]
            
            runningSum = 0
            
            for i in rawTrainingData:
                runningSum += func1(i)*func2(i)
                
            print runningSum 
                

def weightedSigmoidFuncMaker(svdMat, i):
    relevantEigenvector = svdMat[i]
    def weightedSigmoidFunc(inputs):
        activation = np.dot(inputs, relevantEigenvector)
        if activation >= 0.0:
            return 1.0
        else:
            return -1.0
            
    return weightedSigmoidFunc

# Same algorithm as the FuncSum algorithm, but entirely in matrix form
# Should be faster that way
class MatFuncSum:
    def __init__(self):
        pass
        
    def train(self, trainingSet):
#        print trainingSet[0]
#        print np.dot(np.linalg.inv(trainingSet[0]), trainingSet[1])
        
        self.computeFunctionCoeffs(trainingSet[0])
        
#        checkThatFuncsAreOrthogonal(trainingSet[0], self.C)
        
        self.computeFunctionWeights(trainingSet[1])
        
    def computeFunctionCoeffs(self, inputData):
                
        self.X = inputData
        
        middleMatrix = np.dot(np.transpose(self.X), self.X)
        print "Done computing the data summary"
        
        eigenvalues, eigenvectors = np.linalg.eig(middleMatrix)
        print "Found an orthogonal set of functions"
#        print eigenvalues
        
#       listOfSigmoidFuncs = []
#        for i in range(len(eigenvectors)):
            
            # matrix of random values between -1 and 1    
#            numInputs = len(eigenvectors)
            
#            coeffArray = 2*(0.5*np.ones((numInputs,numInputs))-\
#                np.random.rand(numInputs,numInputs))
            
                    
#            listOfSigmoidFuncs.append(weightedSigmoidFuncMaker(eigenvectors, i))
#            listOfSigmoidFuncs.append(weightedSigmoidFuncMaker( \
#                coeffArray, i))    
                
#        checkThatFuncsAreOrthogonal(inputData, listOfSigmoidFuncs)
        
        self.C = np.transpose(eigenvectors)
 #       print self.C
 
        listOfSigmoidFuncs = []
        for i in range(len(eigenvectors)):
            
            # matrix of random values between -1 and 1    
            numInputs = len(eigenvectors)
            
            coeffArray = 2*(0.5*np.ones((numInputs,numInputs))-\
                np.random.rand(numInputs,numInputs))
            
                    
            listOfSigmoidFuncs.append(weightedSigmoidFuncMaker(self.C, i))
#            listOfSigmoidFuncs.append(weightedSigmoidFuncMaker( \
#                coeffArray, i))     

        checkThatFuncsAreOrthogonal(inputData, listOfSigmoidFuncs)
                
        self.D = np.diag(np.array([invertIfNotZero(e) for e in eigenvalues]))

    def computeFunctionWeights(self, outputData):
        self.yVec = outputData
        self.F = np.dot(self.C, np.transpose(self.X))
#        print self.F, self.yVec
        self.wVec = np.dot(np.dot(self.D, self.F), self.yVec)

#        print self.wVec

        self.weights = np.dot(np.transpose(self.wVec), self.C)
#        print self.weights
        print "Found the final weights"
                
    def evaluate(self, inputs):
        return np.dot(self.weights, inputs)
        
    def test(self, testSet):
        numDataPoints = len(testSet)
        
        yHat = self.getOurPredictions(testSet[0])        
        difference = yHat - np.transpose(testSet[1])
        
        mse = sum(np.multiply(difference, difference))/numDataPoints
        
        print "mse", mse
        
    def getOurPredictions(self, inputData):
        return np.dot(self.weights, np.transpose(inputData))
        
class MatFuncSumClique: 
    def __init__(self, numOptions, convertToIndex):
        self.listOfFuncSums = [MatFuncSum() for i in range(numOptions)]
        self.numOptions = numOptions
        self.convertToIndex = convertToIndex
        
    def getListOfTrainingSets(self, trainingSet):
        listOfTrainingSets = []
        
        outputs = trainingSet[1]
            
        listOfNewOutputs = [[] for _ in range(self.numOptions)]    
        
        for output in outputs:
            for l in listOfNewOutputs:
                l.append(0.0)
                
            listOfNewOutputs[self.convertToIndex(output)][-1] += 1.0       
        
        for i in range(self.numOptions):
            listOfTrainingSets.append((trainingSet[0], np.array(listOfNewOutputs[i])))
            
        return listOfTrainingSets
        
    def train(self, trainingSet):
    
        listOfTrainingSets = self.getListOfTrainingSets(trainingSet)
        
#        print listOfTrainingSets
        
        print "Done with data preprocessing"
        
        [fs.train(ts) for fs, ts in zip(self.listOfFuncSums, listOfTrainingSets)]            
    def test(self, testSet, verbose=False):
        inputData = testSet[0]
        
        allResults = [fs.getOurPredictions(inputData) for fs in \
            self.listOfFuncSums]
        
        print allResults
        
        numCorrect = 0.
        numTotal = 0.
        
        for i in range(len(inputData)):
            results = [allResults[j][i] for j in range(self.numOptions)]
            
            bestResultIndex = argmax(results)
            
            if self.convertToIndex(testSet[1][i]) == bestResultIndex:
                numCorrect += 1.
                numTotal += 1.
                
            else:
                numTotal += 1.
                
                if verbose:
                    print "Image"
#                    displayDigit(inputs)

                    
            print "Program's guess:", bestResultIndex
                    
                
        print "Got", numCorrect, "out of", numTotal, "correct."
        
        return numCorrect/numTotal        