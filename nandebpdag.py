import random
import mnist_loader

from bitcounter import *

# These are the values that are passed into the nands; they represent the probability that the true value is 1

# These are more or less just mutable Floats 
class Value:
    def __init__(self):
        self.value = None
        
    def set(self, value):
        self.value = value

class Nand:
    def __init__(self, weights, ID, inputList, output):
        assert len(weights) == len(inputList)
        
        # UNCHANGING
        self.ID = ID
        self.inputList = inputList
        self.output = output
        
        # MUTABLE
        self.weights = weights
    
    def getInputs(self):
        return [inp.value for inp in self.inputList] 
        
    def setInputs(self, newInputs):
#        print newInputs, self.inputList
        assert len(newInputs) == len(self.inputList)
        
        for i, value in enumerate(newInputs):
            self.inputList[i].set(value)
        
    def feedForward(self):
        inputs = self.getInputs()
        
        numInputs = len(inputs)
        
        assert numInputs == len(self.weights)
        
        isThereAnActive0Product = 1.0
        
        for i in range(numInputs):
            x = inputs[i]
            w = self.weights[i]
            
            # Multiply by the probability that either it's inactive or it's not a 0
            isThereAnActive0Product *= x + 1 - w - x*(1-w)
            
        self.output.set(1.0 - isThereAnActive0Product)
        
    def determineFeedForward(self):
        inputs = self.getInputs()
        
        numInputs = len(inputs)
        
        assert numInputs == len(self.weights)
        
        isThereAnActive0Product = 1.0
        
        for i in range(numInputs):
            x = inputs[i]
            w = 1.0 * (self.weights[i] > random.random())    
     
            isThereAnActive0Product *= x + 1 - w - x*(1-w)
            
        self.output.set(1.0 - isThereAnActive0Product)
    
    def accurateFeedForward(self):
        inputs = self.getInputs()
        conditionalInputs = self.getConditionalInputs()
        
        numInputs = len(inputs)
        
        assert numInputs == len(self.weights)
        assert len(conditionalInputs) == numInputs
        
        isThereAnActive0Product = 1.0
        
        for i in range(numInputs):
            x = inputs[i]
            w = self.weights[i]
            
            pureInputs = self.getPureInputs()
            
            xGivenPrevious = conditionalInputs[i]
            
            isThereNoActive0Product *= orProb(xGivenPrevious, 1-w)
            
        self.output.set(1.0 - isThereAnActive0Product)
        self.core.set()
    
    
            
     
    def updateFromOutput(self):
                
        currentWeights = self.weights
        
        inputs = self.getInputs()
        
#        weightContributionFrom0 = numTimesList(1-self.output.value, updateWeightsFrom0(inputs, currentWeights))
#        weightContributionFrom1 = numTimesList(self.output.value, updateWeightsFrom1(inputs, currentWeights))
        
#        print "before weights", self.weights
#        print "before inputs", inputs
#        print "output", self.output.value
        
        if self.output.value < 1e-10:
            try:
                self.weights = updateWeightsFrom0(inputs, self.weights)
            except:
                print "Impossible that output is 0 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise
            
        elif self.output.value > 1.0 - 1e-10:
            try:
                self.weights = updateWeightsFrom1(inputs, self.weights)
            except:
                print "Impossible that output is 1 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise
            
        else:
            try:
                weightContributionFrom0 = numTimesList(1-self.output.value, updateWeightsFrom0(inputs, self.weights))
            except:
                print "Impossible that output is 0 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise            
            
            try:
                weightContributionFrom1 = numTimesList(self.output.value, updateWeightsFrom1(inputs, self.weights))
            except:
                print "Impossible that output is 1 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise     
        
            self.weights = listPlusList(weightContributionFrom0, weightContributionFrom1)
            
        if self.output.value < 1e-10:
            try:
                self.setInputs(updateInputsFrom0(inputs, self.weights))
            except:
                print "Impossible that output is 0 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise            
                        
        elif self.output.value > 1.0 - 1e-10:
            try:
                self.setInputs(updateInputsFrom1(inputs, self.weights))
            except:
                print "Impossible that output is 1 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise        
                
        else:
            try:
                inputContributionFrom0 = numTimesList(1-self.output.value, updateInputsFrom0(inputs, self.weights))
            except:
                print "Impossible that output is 0 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise   
                
            try:
                inputContributionFrom1 = numTimesList(self.output.value, updateInputsFrom1(inputs, self.weights))
            except:
                print "Impossible that output is 1 on gate", self.ID
                print "My weights are", self.weights
                print "My inputs are", inputs
                raise        
                        
            self.setInputs(listPlusList(inputContributionFrom0, inputContributionFrom1))
        
#        print "after weights", self.weights
#        print "after inputs", self.getInputs()
#        print ""
        
    def roundWeights(self):
        self.weights = [round(w) for w in self.weights] 
        
    def randomizeWeights(self):
        self.weights = [1.0*(w<random.random()) for w in self.weights]
        
    def printWeights(self):
        print self.weights
        
class DAG:
    def __init__(self, numInputs, numInternalNands, numOutputNands):
        self.inputValues = [Value() for i in range(numInputs)]
        
        self.listOfInternalNands = []
        self.listOfOutputNands = []
        
        self.outputValues = []
        
        runningListOfInputs = self.inputValues[:]
        
        for i in range(numInternalNands):
            newValue = Value()
            
#            initialWeights = [1./numInputs]*len(runningListOfInputs)
            initialWeights = [1./(numInputs+numInternalNands+numOutputNands)]*len(runningListOfInputs)
#            initialWeights = [0.5]*len(runningListOfInputs)

#            if i == 0:
#                initialWeights = [0.9, 0.9]
#            elif i == 1:
#                initialWeights = [0.1, 0.1, 0.9]
#            elif i == 2:
#                initialWeights = [0.9, 0.9, 0.9, 0.9]
#            elif i == 3:
#                initialWeights = [0.1, 0.1, 0.1, 0.9, 0.9]
            
            newNand = Nand(initialWeights, i, runningListOfInputs, newValue)
            
            runningListOfInputs = runningListOfInputs + [newValue]
            self.listOfInternalNands.append(newNand)
        
        for i in range(numOutputNands):
            newValue = Value()
            
#            initialWeights = [1.0/numInputs]*len(runningListOfInputs)
            initialWeights = [0.5] * len(runningListOfInputs)
            
            newNand = Nand(initialWeights, numInternalNands+i, runningListOfInputs, newValue)
            
            runningListOfInputs = runningListOfInputs + [newValue]
            
            self.listOfOutputNands.append(newNand)
            self.outputValues.append(newValue)    
                        
        self.listOfNands = self.listOfInternalNands + self.listOfOutputNands                
                        
    def setInputValues(self, newInputs):
        assert len(newInputs) == len(self.inputValues)
        
        for i, value in enumerate(newInputs):
            self.inputValues[i].set(value)      
            
    def setOutputValues(self, newOutputs):
        assert len(newOutputs) == len(self.outputValues)
        
        for i, value in enumerate(newOutputs):
            self.outputValues[i].set(value)  
            
    def getOutputs(self):
        return [v.value for v in self.outputValues]        
            
    def feedForward(self, inputs):
        self.setInputValues(inputs)
        
        for nand in self.listOfNands:        
            nand.feedForward()
            
        return self.getOutputs()
        

    
    def determineFeedForward(self, inputs):
        self.setInputValues(inputs)
        
        for nand in self.listOfNands:
            nand.determineFeedForward()
            
        return self.getOutputs()    
        
    def updateFromOutput(self, correctOutputs):
        self.setOutputValues(correctOutputs)
                
        for i in range(len(self.listOfNands) - 1, -1, -1):
            nand = self.listOfNands[i]
            
            nand.updateFromOutput()    
                    
    def train(self, trainingSet, epochs=1):
        trainingSetCopy = trainingSet[:]
        
        for i in range(epochs):
            random.shuffle(trainingSetCopy)
            
            counter = 0
            
            for dataPoint in trainingSetCopy:
                inputs = dataPoint[0]
                correctOutput = dataPoint[1]
                
                print "Mini-epoch", counter, "complete"
                
                counter += 1
                
                self.feedForward(inputs)
                print correctOutput
                self.updateFromOutput(correctOutput)
                
            print "Epoch", i, "complete."
            
    def testBoolean(self, testSet, randomOutcomes=False, verbose=False):
        correctnessCounter = 0.0
        randomCounter = 0.0
        alwaysZeroCounter = 0.0
        overallCounter = 0.0
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            correctOutputs = dataPoint[1]
            
            myOutputs = self.feedForward(inputs)
            
            assert len(correctOutputs) == len(self.outputValues)
            
            if verbose:
                print "Correct:", correctOutputs
                print "Observed:", myOutputs
                print ""                
                        
            for i in range(len(correctOutputs)):              
                if round(myOutputs[i]) == correctOutputs[i]:
                    correctnessCounter += 1.0
                
                if random.random() < 0.5:
                    randomCounter += 1.0
				
                if not correctOutputs[i]:
                    alwaysZeroCounter += 1.0
                
                overallCounter += 1.0
        
        print "Got", correctnessCounter, "out of", overallCounter, "correct."
        print ""
        
        if randomOutcomes:        
            print "Compare to the random outcome: "	
            print "Got", randomCounter, "out of", overallCounter, "correct."	
            print ""
            print "Compare to the outcome you'd have gotten if you always picked zero: "
            print "Got", alwaysZeroCounter, "out of", overallCounter, "correct."
				
        return correctnessCounter / overallCounter
    
    def testMax(self, testSet, verbose=False):
        correctnessCounter = 0.0
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            correctOutputs = dataPoint[1]
                        
            myOutputs = self.feedForward(inputs)            
                        
            assert len(correctOutputs) == len(myOutputs)
            
            if verbose:
                print "Correct:", correctOutputs
                print "Observed:", myOutputs
                print ""
            
            if argmax(correctOutputs) == argmax(myOutputs):
                correctnessCounter += 1.0
        
        print "Got", correctnessCounter, "out of", len(testSet), "correct."
        
        return correctnessCounter / len(testSet)
    
    def testMaxIndex(self, testSet, verbose=False):
        correctnessCounter = 0.0
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            correctOutput = dataPoint[1]

            myOutputs = self.feedForward(inputs)            
            
            if argmax(myOutputs) == correctOutput:
                correctnessCounter += 1.0
                
            if verbose:
                print "Correct:", correctOutput
                print "Observed:", argmax(myOutputs), myOutputs
                print ""    
                
        print "Got", correctnessCounter, "out of", len(testSet), "correct."
        
        return correctnessCounter / len(testSet)
    
    def roundWeights(self):
        for nand in self.listOfNands:
            nand.roundWeights()      
            
    def randomizeWeights(self):
        for nand in self.listOfNands:
            nand.randomizeWeights()        
            
    def outputWeights(self):
        for nand in self.listOfNands:
            nand.printWeights()        
            
def numTimesList(x, l):
    return [x * i for i in l]

def listPlusList(l1, l2):
    lenL1 = len(l1)
    assert lenL1 == len(l2)

    return [l1[i] + l2[i] for i in range(lenL1)]

def allBooleanTuplesOfLength(x):
    if x == 0:
        return [()]
        
    allSizeXMinus1Tuples = allBooleanTuplesOfLength(x-1)
    
    return [littleTuple + tuple([True]) for littleTuple in allSizeXMinus1Tuples] + \
        [littleTuple + tuple([False]) for littleTuple in allSizeXMinus1Tuples]
        

# This NAND gate outputs a 0. We must update in response!

# ALL active inputs must be 1's. So if a weight is a 1, then the corresponding input MUST be 1!
#def updateInputsFrom0(inputs, weights):
#    assert len(inputs) == len(weights)
    
#    newInputs = []
    
#    for i in range(len(weights)):
#        x = inputs[i]
#        w = weights[i]
        
#        newInputs.append((1-w)*x + w)
        
#    return newInputs

def argmax(l):
    currentMax = float("-Inf")
    currentIndex = None
    
    for i, item in enumerate(l):
        if item > currentMax:
            currentMax = item
            currentIndex = i
            
    return currentIndex
 
def orProb(x, y):
     return x + y - x*y
 
def updateInputsFrom0(inputs, weights):
    assert len(inputs) == len(weights)
    
    newInputs = []
    
    for i in range(len(weights)):
        x = inputs[i]
        w = weights[i]
        
        newInputs.append(x / (x + (1-x)*(1-w)))
        
    return newInputs 
    
# This NAND gate output a 1. We must update in response!

# Some active input must be a 0. 
def updateInputsFrom1Slower(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == len(weights)
    
    listOfOnnessAssignments = allBooleanTuplesOfLength(len(inputs))
    
    onnessDict = {}
    
    sumAllPosteriors = 0.0
    
    for booleanTuple in listOfOnnessAssignments:
        
        # Multiplies together the probabilities that the inputs are this onness level in the first place
        priorProduct = 1.0
        
        # Multiplies together the probabilities that each input that's a 0 is inactive (one of them must be active, so we'll do 1-this at the end)
        likelihoodProduct = 1.0
        
        for i, onness in enumerate(booleanTuple):
            if onness:
                priorProduct *= inputs[i]
            else:
                priorProduct *= 1.0 - inputs[i]
                
            if not onness:
                likelihoodProduct *= 1.0 - weights[i]
                
        likelihoodProduct = 1.0 - likelihoodProduct
        
        onnessDict[booleanTuple] = priorProduct * likelihoodProduct
        
        sumAllPosteriors += onnessDict[booleanTuple]
        
    # Now we need to figure out the probability that each input is on or not
    newInputs = []
    
    for i in range(numWeights):
        
        tuplesBefore = allBooleanTuplesOfLength(i)
        tuplesAfter = allBooleanTuplesOfLength(numWeights - i - 1)
        
        truthLikelihood = 0.0
        
        for beforeTuple in tuplesBefore:
            for afterTuple in tuplesAfter:
                trueTuple = beforeTuple + tuple([True]) + afterTuple
                
#                print trueTuple, onnessDict[trueTuple]
                
                truthLikelihood += onnessDict[trueTuple]
        
#        print truthLikelihood, sumAllPosteriors
        
        # Weird 0/0 case, just take the prior
#        if truthLikelihood == 0 and sumAllPosteriors == 0:
#            newInputs.append(inputs[i])
        
#        else:
        newInputs.append(truthLikelihood / sumAllPosteriors)
        
    return newInputs
    
def updateInputsFrom1(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == numWeights
        
    isThereAnActive0Product = 1
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        isThereAnActive0Product *= orProb(x, 1-w)
        
    newInputs = []
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        if orProb(x, 1-w) == 0.0:
            newInputs.append(0.0)
        else:
            oddsOfEverythingElseAlright = 1 - isThereAnActive0Product/(orProb(x, 1-w))
      
            everythingAlrightDespiteMe = x * oddsOfEverythingElseAlright
            everythingAlrightWithoutMe = (1-x) * orProb(w, oddsOfEverythingElseAlright)
        
            newInputs.append(everythingAlrightDespiteMe / (everythingAlrightDespiteMe + everythingAlrightWithoutMe))
    
#    print newInputs
    
    return newInputs
# This NAND gate outputs a 0. We must update in response!

# ALL active inputs must be 1's. So if an input is a 0, then the corresponding weight MUST be 0!
def updateWeightsFrom0(inputs, weights):
    assert len(inputs) == len(weights)
    
    newWeights = []
    
    for i in range(len(weights)):
        x = inputs[i]
        w = weights[i]
        
        # x is prob input is a 1 (that's the okay probability)
        # In this case we don't know what the weight should be so we use its prior odds

#        print 1-w, w*x, w, x
        
        newWeights.append(w*x / (1-w + w*x))
        
    return newWeights

# The NAND gate outputs a 1. We must update in response!

# Some active input must be a 0.
def updateWeightsFrom1(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == numWeights
    
    isThereAnActive0Product = 1
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        isThereAnActive0Product *= orProb(x, 1-w)
    
    newWeights = []
    
    for i in range(numWeights):
        x = inputs[i]
        w = weights[i]
        
        if orProb(x, 1-w) == 0.0:
            newWeights.append(1.0)
        # Think about it! If the or is 0, then w has to be 1, there's no two ways about it
        else:    
            oddsOfEverythingElseAlright = 1 - isThereAnActive0Product/(orProb(x, 1-w))
        
            everythingAlrightThanksToMe = w * orProb(1-x, oddsOfEverythingElseAlright)
            everythingAlrightWithoutMe = (1-w) * oddsOfEverythingElseAlright
        
#            print everythingAlrightThanksToMe, everythingAlrightWithoutMe
        
            newWeights.append(everythingAlrightThanksToMe / (everythingAlrightThanksToMe + everythingAlrightWithoutMe))

    return newWeights
# The NAND gates outputs a 1. We must update in response!

def updateWeightsFromAnythingSlow(inputs, weights, output):
    numWeights = len(weights)
    
    assert len(inputs) == numWeights
    
    listOfActivityAssignments = allBooleanTuplesOfLength(2*numWeights+1)
    
    activityDict = {}
    
    sumAllPosteriors = 0.0
    
    weightDict = {}
    weightUpdateDict = {}
            
    for i in range(numWeights):
        weightUpdateDict[i] = 0.0
    
    for booleanTuple in listOfActivityAssignments:
        oddsOfThisHappening = 1.0
        isThereAnActive0 = False
        
        for i, activity in enumerate(booleanTuple[:numWeights]):
            weightDict[i] = activity
        
            if activity:    
                oddsOfThisHappening *= weights[i]
            else:
                oddsOfThisHappening *= 1-weights[i]
                                
        for i, activity in enumerate(booleanTuple[numWeights:2*numWeights]):
            if weightDict[i] and not activity:
                isThereAnActive0 = True
            
            if activity:
                oddsOfThisHappening *= inputs[i]
                
            else:
                oddsOfThisHappening *= 1-inputs[i]
               
#        print booleanTuple, isThereAnActive0       
               
        if booleanTuple[2*numWeights] and isThereAnActive0:
            oddsOfThisHappening *= output       
                
        elif booleanTuple[2*numWeights] and not isThereAnActive0:
            oddsOfThisHappening *= 0.0    
            
        elif not booleanTuple[2*numWeights] and isThereAnActive0:
            oddsOfThisHappening *= 0.0
            
        elif not booleanTuple[2*numWeights] and not isThereAnActive0:
            oddsOfThisHappening *= 1-output
        
        else:
            raise    
            
        for i in weightDict:
            activity = weightDict[i]
            
            if activity:
                weightUpdateDict[i] += oddsOfThisHappening                
            
        sumAllPosteriors += oddsOfThisHappening
        
    return [weightUpdateDict[i] / sumAllPosteriors for i in range(numWeights)]
        

# Some active input must be a 0.         
def updateWeightsFrom1Slower(inputs, weights):
    numWeights = len(weights)
    
    assert len(inputs) == len(weights)
    
    listOfActivityAssignments = allBooleanTuplesOfLength(len(inputs))
    
    activityDict = {}
    
    # The sum of all posterior probabilities (used for renormalizing)
    sumAllPosteriors = 0.0
    
    for booleanTuple in listOfActivityAssignments:
        
        # Multiplies together the probabilities that the weights are this activity level in the first place
        priorProduct = 1.0
        
        # Multiplies together the probabilities that each active input is a 1 (one of them must be a 0, so we'll do 1-this at the end)
        likelihoodProduct = 1.0
        
        for i, activity in enumerate(booleanTuple):
            if activity:
                priorProduct *= weights[i]
            else:
                priorProduct *= 1.0 - weights[i]
            
            if activity:
                likelihoodProduct *= inputs[i]
            
        likelihoodProduct = 1.0 - likelihoodProduct
        
#        f priorProduct, likelihoodProduct
        
        activityDict[booleanTuple] = priorProduct * likelihoodProduct
                
        sumAllPosteriors += activityDict[booleanTuple]                 
                        
    # Now we need to figure out the probability that each input is active or not
    newWeights = []
    
    for i in range(numWeights):
        
        tuplesBefore = allBooleanTuplesOfLength(i)
        tuplesAfter = allBooleanTuplesOfLength(numWeights - i - 1)
        
        truthLikelihood = 0.0
        
        for beforeTuple in tuplesBefore:
            for afterTuple in tuplesAfter:
                trueTuple = beforeTuple + tuple([True]) + afterTuple
                
#                print trueTuple, activityDict[trueTuple]
                
                truthLikelihood += activityDict[trueTuple]
                
#        print truthLikelihood, sumAllPosteriors        
                
#        if truthLikelihood == 0 and sumAllPosteriors == 0:
#            newWeights.append(weights[i])        
        
#        else:        
        newWeights.append(truthLikelihood / sumAllPosteriors)
        
    return newWeights

def allSizeXLists(x):
    if x == 0:
        return [[]]
        
    allSizeXMinus1Lists = allSizeXLists(x - 1)
    
    return [littleList + [0.0] for littleList in allSizeXMinus1Lists] + \
        [littleList + [1.0] for littleList in allSizeXMinus1Lists]

patternSize = 50
trainingSetSize = 1000
testSetSize = 1000


dag = DAG(patternSize, 50, 1)

trainingSet = getCompressionData("../../compression/mediumpattern.txt", patternSize)[:1000]
testSet = getCompressionData("../../compression/mediumpattern.txt", patternSize)[1000:2000]

compressionData = getCompressionData("../../compression/declarationbits.txt", patternSize)

trainingSet = compressionData[:trainingSetSize]
testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]

dag.train(trainingSet, 1)
dag.testBoolean(testSet, True, True)

#print "done"

val1 = Value()
val1.set(0.5)

val2 = Value()
val2.set(0.7)

valOut = Value()

nand = Nand([0.2, 0.3], 3, [val1, val2], valOut)

nand.feedForward()
print updateWeightsFrom0([val1.value, val2.value], [0.2, 0.3])
print updateWeightsFrom1([val1.value, val2.value], [0.2, 0.3])

print [x*valOut.value + y*(1-valOut.value) for \
    x, y in zip(updateWeightsFrom0([val1.value, val2.value], [0.2, 0.3]), \
        updateWeightsFrom1([val1.value, val2.value], [0.2, 0.3]))]

print valOut.value
#valOut.value = 0.5
print valOut.value

print "hi", [x*valOut.value + y*(1-valOut.value) for \
    x, y in zip(updateWeightsFrom1([val1.value, val2.value], [0.2, 0.3]), \
        updateWeightsFrom0([val1.value, val2.value], [0.2, 0.3]))]

print updateWeightsFromAnythingSlow([val1.value, val2.value], [0.2, 0.3], valOut.value)

valOut1 = Value()
w1 = 0.6

nand1 = Nand([w1], 1, [val2], valOut1)

nand1.feedForward()

print valOut1.value

print updateWeightsFrom0([val2.value], [w1])
print updateWeightsFrom1([val2.value], [w1])

print [x*valOut1.value + y*(1-valOut1.value) for \
    x,y in zip(updateWeightsFrom0([val2.value], [w1]), \
    updateWeightsFrom1([val2.value], [w1]))]

print updateWeightsFromAnythingSlow([val2.value], [w1], valOut1.value)


#dag = DAG(1, 2, 1)

#print dag.feedForward([0])

#numerator = 0
#denominator = 0

#for i in range(10000):
#    if dag.determineFeedForward([0])[0] == 1.0:
#        numerator += 1.0
#        denominator += 1.0
        
#    else:
#        denominator += 1.0
        
#print numerator / denominator

#dag = DAG(3, 23, 4)

#allSize3Lists = allSizeXLists(3)

#trainingSet = []

#for littleList in allSize3Lists:
#    if sum(littleList) == 0:
#        trainingSet.append([littleList, [1.0, 0.0, 0.0, 0.0]])
#    elif sum(littleList) == 1:
#        trainingSet.append([littleList, [0.0, 1.0, 0.0, 0.0]])
#    elif sum(littleList) == 2:
#        trainingSet.append([littleList, [0.0, 0.0, 1.0, 0.0]])
#    else:
#        trainingSet.append([littleList, [0.0, 0.0, 0.0, 1.0]])

#dag.train(trainingSet, 150)

#dag.testMax(trainingSet, True)

#dag = DAG(2, 4, 3)

#allSize2Lists = allSizeXLists(2)

#trainingSet = []

#for littleList in allSize2Lists:
#    if sum(littleList) == 0:
#        trainingSet.append([littleList, [1.0, 0.0, 0.0]])
#    elif sum(littleList) == 1:
#        trainingSet.append([littleList, [0.0, 1.0, 0.0]])
#    else:
#        trainingSet.append([littleList, [0.0, 0.0, 1.0]])
        
#dag.train(trainingSet, 10)
#dag.roundWeights()

#dag.testMax(trainingSet, True)

#dag.outputWeights()

#dag = DAG(5, 35, 1)

#allSize5Lists = allSizeXLists(5) 

#trainingSet = []

#for littleList in allSize5Lists:
#    if sum(littleList) == 2:
#        trainingSet.append([littleList, [1.0]])
#    else:
#        trainingSet.append([littleList, [0.0]])

#dag.train(trainingSet, 300)

#dag.testBoolean(trainingSet, True)

#print updateInputsFrom0([0, 0.1], [0.2, 0.4]), updateWeightsFrom0([0, 0.1], [0.2, 0.4])        

#trainingSet = [[[0.0, 0.0], [0.0]], [[0.0, 1.0], [1.0]], [[1.0, 0.0], [1.0]], [[1.0, 1.0], [0.0]]]

#trainingSet = [[[0.0, 0.0], [1.0]], [[0.0, 1.0], [1.0]], [[1.0, 0.0], [1.0]], [[1.0, 1.0], [1.0]]] 

#dag = DAG(2, 3, 1)

#print updateWeightsFrom1([0.2, 0.3], [0.1, 0.4])
#print updateWeightsFrom1Faster([0.2, 0.3], [0.1, 0.4])

#f updateInputsFrom1([0.2, 0.3], [0.1, 0.4])
#print updateInputsFrom1Faster([0.2, 0.3], [0.1, 0.4])

#dag.train(trainingSet, 10)

#sdag.testBoolean(trainingSet, True)
#print dag.feedForward([0.0, 0.0])                
#print dag.feedForward([0.0, 1.0])                
#print dag.feedForward([1.0, 0.0])                
#print dag.feedForward([1.0, 1.0])                
                
#trainingSet = []

#allSize3Lists = allSizeXLists(3)

#for littleList in allSize3Lists:
#    if sum(littleList) > 1:
#        trainingSet.append([littleList, [1.0, 0.0]])
#    else:
#        trainingSet.append([littleList, [0.0, 1.0]])

#dag = DAG(3, 3, 2)

#dag.train(trainingSet, 100)
#dag.roundWeights()

#for littleList in allSize3Lists:
#    print littleList, dag.feedForward(littleList)

#print dag.testBoolean(trainingSet, True)    
    
#trainingSet = [[[0.0, 0.0, 0.0], [1.0]], [[0.0, 0.0, 1.0], [0.0]], [[0.0, 1.0, 0.0], [1.0]], [[0.0, 1.0, 1.0], [1.0]],
#    [[1.0, 0.0, 0.0], [0.0]], [[1.0, 0.0, 1.0], [1.0]], [[1.0, 1.0, 0.0], [0.0]], [[1.0, 1.0, 1.0], [0.0]]]     

#dag = DAG(3, 7, 1)

#dag.train(trainingSet, 30)

#print dag.testBoolean(trainingSet, True)
    
#print updateInputsFrom0([0.3, 0.01], [0.4, 0.99])
#print updateInputsFrom1([0.3, 0.01], [0.4, 0.99])                
#print updateWeightsFrom0([0.3, 0.01], [0.4, 0.99])
#print updateWeightsFrom1([0.3, 0.01], [0.4, 0.99])
#print ""
#print updateInputsFrom0([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print updateInputsFrom1([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print updateWeightsFrom0([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print updateWeightsFrom1([0.8, 0.7, 0.8, 0.6, 0.9], [0.3, 0.2, 0.2, 0.1, 0.5])
#print ""
#print updateInputsFrom0([0.99, 0.99], [0.99, 0.99])
#print updateInputsFrom1([0.99, 0.99], [0.99, 0.99])
#print updateWeightsFrom0([0.99, 0.99], [0.99, 0.99])
#print updateWeightsFrom1([0.99, 0.99], [0.99, 0.99])

#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Takes a data set and makes the only thing that matters the x'th entry of the test (so for example if x was 2 we
# would only try to tell if a digit was a 2 or not)
def process(data, x, is_scalar=False):
    returnList = []
    
    random.shuffle(data)
    
    for i, inp in enumerate(data[:10]):
        
#       print inp
        
        # convert to list
        inputList = []
        for value in inp[0]:
            inputList.append(value)
        
#       print len(inputList)            
            
        if is_scalar:
            returnList.append([inputList, 1.0*(inp[1] == x)])    
        else:
            returnList.append([inputList, inp[1][x]])
        
    return returnList    

#processedTrainingData = process(training_data, 0, False)
#print [len(x) for x in test_data]
#processedTestData = process(test_data, 0, True)
#print [len(x) for x in training_data]

#dag = DAG(784, 60, 10)
#dag.train(training_data[:50])
#print dag.testMaxIndex(test_data[:50], True)



#print [x[0] for x in training_data]

#dag.SGD(training_data)