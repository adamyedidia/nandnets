def convertToOneMinusOne(x):
    if x == 0:
        return -1.0
        
    return 1.0

def majority(inputs, params):
    assert len(inputs) == len(params)
    
    counter = 0
    
    for inp, param in zip(inputs, params):
        counter += convertToOneMinusOne(inp) * param
        
    return 1*(counter>0)

class Majoritarian:
    def __init__(self, functionList):
        self.functionList = functionList
                
    def evaluate(self, inputs):
        inputsToMajority = [f(inputs) for f in functionList]
        
        return majority(inputsToMajority, self.params)
        
    def train(self, trainingSet):
        # Note: only works on 1-output functions
        
        self.params = [0]*len(functionList)
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1][0]
            
            inputsToMajority = [f(inputs) for f in functionList]
            
            for majInput in inputsToMajority:
                if majInput == output:
                    self.params += 1.0
                else:
                    self.params -= 1.0
                    
        self.params = [i/len(functionList) for i in self.params]
        
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