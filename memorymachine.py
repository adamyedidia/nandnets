import random

def pront(x):
    print x

class MemoryMachine:
    def __init__(self):
        self.memory = {}
        
    def evaluate(self, inputs):    
        if tuple(inputs) in self.memory:
            return self.memory[tuple(inputs)]
            
        return [1.0*(self.numOnesSeen[i] > self.numZeroesSeen[i]) \
            for i in range(len(self.numZeroesSeen))]
        
    def train(self, trainingSet):
        numOutputs = len(trainingSet[0][1])
        self.numZeroesSeen = [0]*numOutputs
        self.numOnesSeen = [0]*numOutputs
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            outputs = dataPoint[1]
            
            self.memory[tuple(inputs)] = outputs
            
            for i, output in enumerate(outputs):
                if output == 0:
                    self.numZeroesSeen[i] += 1
                    
                else:
                    self.numOnesSeen[i] += 1
                    
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