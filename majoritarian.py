import random

def pront(x):
    print x

def sign(x):
    return 1.0*(x>=0) + -1.0*(x<0)

def convertListToRealNumber(l):
    if len(l) == 0:
        return 0
    
    else:
        return 2*convertListToRealNumber(l[:-1]) + l[-1]

def convertToOneMinusOne(x):
    if x == 0:
        return -1.0
        
    return 1.0

def allListsOfSizeX(x):
    if x == 0:
        return [[]]
        
    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0] for i in oneLess] + [i + [1] for i in oneLess]  

def majority(inputs, params):
    assert len(inputs) == len(params)
    
    counter = 0
    
    for inp, param in zip(inputs, params):
        counter += convertToOneMinusOne(inp) * param
        
    return 1*(counter>0)

class Majoritarian:
    def __init__(self, functionList):
        self.functionList = functionList
                
    def evaluate(self, inputs, params=None):
        if params==None:
            params = self.params
        
        inputsToMajority = [f(inputs) for f in self.functionList]
        
        return [majority(inputsToMajority, params)]
        
    def enumerateAllPossibleParams(self, numInputs):
        
        funcsSeenSoFar = {}
        
        listOfPossibleParams = allListsOfSizeX(len(self.functionList))
        random.shuffle(listOfPossibleParams)
        
        for params in listOfPossibleParams:
            params = [convertToOneMinusOne(i) for i in params]
            
            tt = self.getTruthTable(numInputs, params)        
            
            if not convertListToRealNumber(tt) in funcsSeenSoFar:
                funcsSeenSoFar[convertListToRealNumber(tt)] = params
                
            else:
#                raise
                print "duplicate function:"
                print funcsSeenSoFar[convertListToRealNumber(tt)]
                print params
                print sum(tt)
                    
                raise
                
            print params, tt, sum(tt), convertListToRealNumber(tt)
            
            
        print len(funcsSeenSoFar)    
#            assert sum(tt) % 2 ==1
            
        
    def perfectTrain(self, trainingSet):
        numInputs = len(trainingSet[0][0])
        
        bestPerformanceSoFar = 0.0
        bestParamsSoFar = None
        
        counter = 0
        
        for params in allListsOfSizeX(len(self.functionList)):    
            params = [convertToOneMinusOne(i) for i in params]
            
            performance = self.getPerformance(trainingSet, params)
            
            counter += 1
            if counter % 1000 == 0: 
                print counter
                print bestPerformanceSoFar
                print bestParamsSoFar
            
            if performance > bestPerformanceSoFar:
                bestPerformanceSoFar = performance
                bestParamsSoFar = params
                
        self.params = bestParamsSoFar
    
    def getTruthTable(self, numInputs, params):
        truthTable = []
        
        for inputs in allListsOfSizeX(numInputs):
            truthTable.append(self.evaluate(inputs, params)[0])
            
        return truthTable
            
    # This is a hell of a lot like self.test()        
    def getPerformance(self, trainingSet, params):
        performance = 0
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1][0]
                        
            if self.evaluate(inputs, params)[0] == output:
                performance += 1
                
        return performance
                    
        
    def train(self, trainingSet, binaryParams=True):
        # Note: only works on 1-output functions
        
        self.params = [0]*len(self.functionList)
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1][0]
            
            inputsToMajority = [f(inputs) for f in self.functionList]
            
            for i, majInput in enumerate(inputsToMajority):
                if majInput == output:
                    self.params[i] += 1.0
                else:
                    self.params[i] -= 1.0
        
        if binaryParams:
            self.params = [sign(i) for i in self.params]
        else:            
            self.params = [i/len(self.functionList) for i in self.params]
        
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
                if correctOutputs != myOutputs:
                    print str(inputs)
                majInputs = [f(inputs) for f in self.functionList]
#                print majInputs
                counter=0
                for inp, param in zip(majInputs, self.params):
                    counter += convertToOneMinusOne(inp) * param
#                        print convertToOneMinusOne(inp), param, \
#                            convertToOneMinusOne(inp) == param
                    
                print counter 
                
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