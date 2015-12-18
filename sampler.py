import random

def pront(x):
    print x

def arrayTimesEquals(arr1, arr2):
    for i, val in enumerate(arr1):
        arr1[i] *= arr2[i]
    
def arrayPlusEquals(arr1, arr2):
    for i, val in enumerate(arr1):
        arr1[i] += arr2[i]
            
def notArray(arr1):
    return [1.-x for x in arr1]
    
def softenArray(arr):
    EPS = 0.001
    # I don't think the epsilons actually matter for this purpose
    return [x+(x==0.0)*EPS-(x==1.0)*EPS for x in arr]

class Sampler:
    def __init__(self, f, initialParams):
        # f must be a function of a list of inputs and a list of parameters
        self.f = f
        self.params = initialParams
        
    def roundParams(self):
        return [round(i) for i in self.params]    
        
    def evaluate(self, inputs, params=None):
        if params == None:
            params = self.roundParams()
        
        return self.f(inputs, params)
        
    def getRandomParamAssignment(self):
        return [1.0*(random.random() < i) for i in self.params]        
        
    def testParamAssignment(self, inputs, outputs, outputIndex, paramAssignment):    
        result = self.evaluate(inputs, paramAssignment)
        
        if outputs[outputIndex] == result[outputIndex]:
            return True
            
        return False
        
    def findGoodParamAssignment(self, inputs, outputs, outputIndex):
        paramAssignment = self.getRandomParamAssignment()
                        
        while not self.testParamAssignment(inputs, outputs, outputIndex, paramAssignment):
            paramAssignment = self.getRandomParamAssignment()
            
        return paramAssignment
        
    def getAverageGoodParamAssignment(self, inputs, outputs, outputIndex, numSamples):
        paramSampleArray = [0.0]*len(self.params)
                                
        for _ in range(numSamples):
                                                
            goodParamAssignment = self.findGoodParamAssignment(inputs, outputs, outputIndex)
            arrayPlusEquals(paramSampleArray, goodParamAssignment)
                        
        return [float(i) / numSamples for i in paramSampleArray]
        
    def train1Epoch(self, trainingSet, numSamples, verbose=False):
        numOutputs = len(trainingSet[0][1])
        
        miniEpochCounter = 0
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            outputs = dataPoint[1]
                
            for outputIndex in range(numOutputs):    
                                
                currentParams = self.params[:]
                #        currentParams = [0.5]*len(self.params)
                notCurrentParams = notArray(currentParams)                
                
                averageGoodParamAssignment = \
                    self.getAverageGoodParamAssignment(inputs, outputs, outputIndex, numSamples)    
                    
                arrayTimesEquals(currentParams, averageGoodParamAssignment)
                arrayTimesEquals(notCurrentParams, notArray(averageGoodParamAssignment))
                
                arrayTimesEquals(currentParams, softenArray(notArray(self.params)))
                arrayTimesEquals(notCurrentParams, softenArray(self.params))          

                self.params = [float(x)/(x+y) for x,y in zip(currentParams, notCurrentParams)]
                pront("Mini-epoch " + str(miniEpochCounter) + " complete.")
                
                miniEpochCounter += 1
                
#                pront("Params: " + str(self.params))
                
        
    def train(self, trainingSet, numEpochs, numSamples, verbose=False):
        for i in range(numEpochs):
            if numEpochs == 1 and verbose:
                self.train1Epoch(trainingSet, numSamples, True)
            else:
                self.train1Epoch(trainingSet, numSamples, False)
            
            pront("Epoch " + str(i) + " complete")
            if verbose:
                pront("Params: " + str(self.params))
            
    def test(self, testSet, randomOutcomes=False, verbose=False):
        correctnessCounter = 0.0
        randomCounter = 0.0
        alwaysZeroCounter = 0.0
        overallCounter = 0.0
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            correctOutputs = dataPoint[1]
                        
            myOutputs = self.evaluate(inputs)
                                    
            if verbose:
                pront("Correct: " + str(correctOutputs))
                pront("Observed: " + str(myOutputs))
                pront("")                
                        
            for i in range(len(correctOutputs)):              
                if round(myOutputs[i]) == correctOutputs[i]:
                    correctnessCounter += 1.0
                
                if random.random() < 0.5:
                    randomCounter += 1.0
				
                if not correctOutputs[i]:
                    alwaysZeroCounter += 1.0
                
                overallCounter += 1.0
        
        pront("Got " + str(correctnessCounter) + " out of " + str(overallCounter) + " correct.")
        pront("")
        
        if randomOutcomes:        
            pront("Compare to the random outcome: ")
            pront("Got " + str(randomCounter) + " out of " + str(overallCounter) + " correct.")	
            pront("")
            pront("Compare to the outcome you'd have gotten if you always picked zero: ")
            pront("Got " + str(alwaysZeroCounter) + " out of " + str(overallCounter) + " correct.")
				
        return correctnessCounter / overallCounter              
    