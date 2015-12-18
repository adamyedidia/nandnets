import random

def orProb(x, y):
     return x + y - (x*y)

class Value:
    def __init__(self):
        self.value = None
    
    def set(self, value):
        self.value = value
        
class ValueSet:
    def __init__(self, setSize):
        self.listOfValues = [Value() for i in range(setSize)]
        self.listOfLikelihoods = []
        
    def set(self, newValues):
        assert len(newValues) == len(self.listOfValues)
        
        for i, val in enumerate(newValues):
            self.listOfValues[i].set(val)
            
    def addLikelihood(self, newLikelihood):
        self.listOfLikelihoods.append(newLikelihood)
        
    def collapseLikelihoods(self):
        for i in range(len(self.listOfValues)):
            yesProduct = self.listOfValues[i].value
            noProduct = 1 - yesProduct
            
            for likelihood in self.listOfLikelihoods:
#                print likelihood.listOfOnValues, i
                yesProduct *= likelihood.listOfOnValues[i]
                noProduct *= likelihood.listOfOffValues[i]
            
            self.listOfValues[i].set(yesProduct / (yesProduct + noProduct))
            
        self.listOfLikelihoods = []
            
class Likelihood:
    def __init__(self, listOfOnValues, listOfOffValues):
        self.listOfOnValues = listOfOnValues
        self.listOfOffValues = listOfOffValues
            
class Nand:
    def __init__(self, weights, ID, inputList, output, setSize):
        assert len(weights) == len(inputList)
                
        # UNCHANGING
        self.ID = ID
        self.inputList = inputList # Type: list of ValueSets
        self.output = output # Type: ValueSet
        self.setSize = setSize
        
        # MUTABLE
        self.weights = weights # Type: List of floats
        self.oddsOfNoActive0 = ValueSet(setSize) # Type: list of ValueSets
        
        assert len(self.weights) == len(self.inputList)
        
    def setInputs(self, newInputs):
        assert len(newInputs) == len(self.inputList)
        assert len(newInputs[0]) == len(self.inputList[0].listOfValues)
        
        for i, valueSet in enumerate(newInputs):
            self.inputList[i].set(valueSet)
    
    # Gets the i'th input from each input-set
    def getInputs(self, i):
        return [inputSet.listOfValues[i].value for inputSet in self.inputList]
            
    def computeOddsOfNoActive0ForI(self, i):
        inputs = self.getInputs(i)
                
        oddsOfNoActive0Product = 1.0
        
        numInputs = len(inputs)
        
        for j in range(numInputs):
            x = inputs[j]        
            w = self.weights[j]

            oddsOfNoActive0Product *= orProb(x, 1-w)

        return oddsOfNoActive0Product
            
    def computeOddsOfNoActive0(self):
        oddsOfNoActive0Set = []
        
        for i in range(self.setSize):
            
            oddsOfNoActive0Set.append(self.computeOddsOfNoActive0ForI(i))        
        
        self.oddsOfNoActive0.set(oddsOfNoActive0Set)
            
    def feedForward(self):
        outputSet = []
        oddsOfNoActive0Set = []
        
        for i in range(self.setSize):
            oddsOfNoActive0 = self.computeOddsOfNoActive0ForI(i)
            
            outputSet.append(1.0 - oddsOfNoActive0)
            oddsOfNoActive0Set.append(oddsOfNoActive0)
            
#            if self.ID == 1 and i == 3:
#                print "whatchawhatchawhatcha", 1.0 - oddsOfNoActive0
#                print outputSet    
#                print self.weights
            
        self.output.set(outputSet)
        self.oddsOfNoActive0.set(oddsOfNoActive0Set)
        
        print "weights", self.weights
        print "outset", outputSet
        
#        self.oldWeights = self.weights[:]
            
    def updateWeights(self):      
        
        newWeights = []
          
        numWeights = len(self.weights) 
        
        for i in range(numWeights):
            
            weightOnProduct = 1.0
            weightOffProduct = 1.0
            
            w = self.weights[i]
            
            if w == 1.0:
                newWeights.append(1.0)
            elif w == 0.0:
                newWeights.append(0.0)  
                
            else:    
                for j in range(self.setSize):
                    inputs = self.getInputs(j)
                    out = self.output.listOfValues[j].value
                    
                    numInputs = len(inputs)
                    
                    assert numInputs == len(self.weights)
                    
                    x = inputs[i] # intentionally i
                                        
                    oddsOfActive0WithoutMe = 1 - self.oddsOfNoActive0.listOfValues[j].value / orProb(x, 1-w)
                    
                    likelihoodForOutput1WeightOn = out * orProb(oddsOfActive0WithoutMe, 1-x)
                    likelihoodForOutput0WeightOn = (1-out) * x * (1-oddsOfActive0WithoutMe)
                    
                    likelihoodForOutput1WeightOff = out * oddsOfActive0WithoutMe
                    likelihoodForOutput0WeightOff = (1-out) * (1-oddsOfActive0WithoutMe)
                    
                    weightOnProduct *= likelihoodForOutput1WeightOn + likelihoodForOutput0WeightOn
                    weightOffProduct *= likelihoodForOutput1WeightOff + likelihoodForOutput0WeightOff
                    
#                    if i == 3:
#                        print "out", out
#                        print "oddsOfActive0woMe", oddsOfActive0WithoutMe
#                        print "x", x
#                        print "w", w
#                        print "x or 1-w", orProb(x, 1-w)
#                        print "1on", likelihoodForOutput1WeightOn
#                        print "0on", likelihoodForOutput0WeightOn
#                        print "1off", likelihoodForOutput1WeightOff
#                        print "0off", likelihoodForOutput0WeightOff
#                        print "wOnProd", weightOnProduct
#                        print "wOffProd", weightOffProduct
 #                   if i == 2:
                        
#                        print out, oddsOfActive0WithoutMe, 1-x, orProb(oddsOfActive0WithoutMe, 1-x)
#                        print "weightOn", likelihoodForOutput1WeightOn, likelihoodForOutput0WeightOn
                        
#                        assert likelihoodForOutput1WeightOn >= 0.0
#                        assert likelihoodForOutput0WeightOn >= 0.0
                        
#                        print "weightOn", likelihoodForOutput1WeightOn + likelihoodForOutput0WeightOn
#                        print "weightOff", likelihoodForOutput1WeightOff + likelihoodForOutput0WeightOff
#                print "wOnProd", "wOffProd", weightOnProduct, weightOffProduct    
                newWeights.append(w * weightOnProduct / (w * weightOnProduct + (1-w) * weightOffProduct))
            
        assert len(newWeights) == len(self.weights)    
        self.weights = newWeights    
        
    def updateInputs(self):
        numWeights = len(self.weights)
        
        assert len(self.inputList) == numWeights
        
        newInputs = []
        
        self.computeOddsOfNoActive0()
        
        for i in range(numWeights):
            newInputs.append([])
        
            w = self.weights[i]
            
            onLikelihood = []
            offLikelihood = []
            
            for j in range(self.setSize):
                inputs = self.getInputs(j)
                out = self.output.listOfValues[j].value
                
                x = inputs[i] # intentionally i
                
                if x == 0.0:
                    onLikelihood.append(0.0)
                    offLikelihood.append(1.0)
                    
                elif x == 1.0:
                    onLikelihood.append(1.0)
                    offLikelihood.append(0.0)
                    
                else:
                    oddsOfActive0WithoutMe = 1 - self.oddsOfNoActive0.listOfValues[j].value / orProb(x, 1-w)
                    
                    likelihoodForOutput1InputOn = out * oddsOfActive0WithoutMe
                    likelihoodForOutput0InputOn = (1-out) * (1-oddsOfActive0WithoutMe)
                    
                    likelihoodForOutput1InputOff = out * orProb(oddsOfActive0WithoutMe, w)
                    likelihoodForOutput0InputOff = (1-out) * (1-w) * (1-oddsOfActive0WithoutMe)
                    
                    likelihoodForInputOn = likelihoodForOutput1InputOn + likelihoodForOutput0InputOn
                    likelihoodForInputOff = likelihoodForOutput1InputOff + likelihoodForOutput0InputOff               
                                        
                    onLikelihood.append(likelihoodForInputOn)
                    offLikelihood.append(likelihoodForInputOff)
                    
            self.inputList[i].addLikelihood(Likelihood(onLikelihood, offLikelihood))     
            
            self.inputList[i].collapseLikelihoods()              
                        
    def updateFromOutput(self):
        print "weights before", self.weights
        print "inputs before", self.getInputs(0)
        print "inputs before", self.getInputs(1)
        print "inputs before", self.getInputs(2)
        print "inputs before", self.getInputs(3)
        self.output.collapseLikelihoods()
        self.updateWeights()
        print "output", [i.value for i in self.output.listOfValues]
        print "weights after", self.weights
        self.updateInputs()                
        print "inputs after", self.getInputs(0)
        print "inputs after", self.getInputs(1)
        print "inputs after", self.getInputs(2)
        print "inputs after", self.getInputs(3)
                    
class DAG:
    def __init__(self, setSize, numInputs, numInternalNands, numOutputNands, edgesBetweenOutputs=False):
        self.inputValueSets = [ValueSet(setSize) for i in range(numInputs)]
        
        self.listOfInternalNands = []
        self.listOfOutputNands = []
        
        self.outputValueSets = []
        self.setSize = setSize
        
        runningListOfInputs = self.inputValueSets[:]
        
        for i in range(numInternalNands):
            newValueSet = ValueSet(setSize)                
        
#            initialWeights = [1./numInputs]*len(runningListOfInputs)

            if i == 0:
                initialWeights = [0.9, 0.9]
            elif i == 1:
                initialWeights = [0.1, 0.1, 0.9]
            elif i == 2:
                initialWeights = [0.9, 0.9, 0.9, 0.9]
            elif i == 3:
                initialWeights = [0.1, 0.1, 0.1, 0.9, 0.9]
                
            newNand = Nand(initialWeights, i, runningListOfInputs, newValueSet, setSize)
            
            runningListOfInputs = runningListOfInputs + [newValueSet]
            
            self.listOfInternalNands.append(newNand)
            
        for i in range(numOutputNands):
            newValueSet = ValueSet(setSize)
            
            initialWeights = [1./numInputs]*len(runningListOfInputs)
            
            newNand = Nand(initialWeights, numInternalNands+i, runningListOfInputs, newValueSet, setSize)
            
            if edgesBetweenOutputs:
                runningListOfInputs = runningListOfInputs + [newValueSet]
                
            self.listOfOutputNands.append(newNand)
            self.outputValueSets.append(newValueSet)
            
        self.listOfNands = self.listOfInternalNands + self.listOfOutputNands
        
    def setInputValues(self, newInputs):
        assert len(newInputs) == len(self.inputValueSets)
        
        for i, valueSet in enumerate(newInputs):
            self.inputValueSets[i].set(valueSet)
            
    def setOutputValues(self, newOutputs):
        assert len(newOutputs) == len(self.outputValueSets)
        
        for i, valueSet in enumerate(newOutputs):
            self.outputValueSets[i].set(valueSet)
            
    def getOutputs(self):
        return [valueSet.listOfValues for valueSet in self.outputValueSets]
        
    def feedForward(self, inputs):
        self.setInputValues(inputs)
        
        for nand in self.listOfNands:
            nand.feedForward()
            
        return self.getOutputs()
        
    def updateFromOutput(self, correctOutputs):
        self.setOutputValues(correctOutputs)
        
        for i in range(len(self.listOfNands) - 1, -1, -1):
            nand = self.listOfNands[i]
            
            nand.updateFromOutput()
            
    def train(self, trainingSet, epochs=1):
    
        inputs = []
        outputs = []
        
        for i in range(len(self.inputValueSets)):
            inputs.append([])
            
        for i in range(len(self.outputValueSets)):
            outputs.append([])
        
        for dataPoint in trainingSet:
            newInputs = dataPoint[0]
            newOutputs = dataPoint[1]
            
            for i, value in enumerate(newInputs):
                inputs[i].append(value)
            
            for i, value in enumerate(newOutputs):
                outputs[i].append(value)
                                
        for i in range(epochs):
            self.feedForward(inputs)
            self.updateFromOutput(outputs)
            
            print "Epoch", i, "complete."
        
    def testBoolean(self, testSet, verbose=False):
        correctnessCounter = 0.0
        overallCounter = 0.0
        
        inputs = []
        
        for i in range(len(self.inputValueSets)):
            inputs.append([])
        
        for dataPoint in testSet:
            newInputs = dataPoint[0]
            
            for i, value in enumerate(newInputs):
                inputs[i].append(value)
                
        myOutputs = self.feedForward(inputs)
            
        for i, dataPoint in enumerate(testSet):
            correctOutput = dataPoint[1]
            
            for j in range(len(correctOutput)):
                
                if verbose:
                    print "Correct:", correctOutput[j]
                    print "Observed:", myOutputs[j][i].value
                
                if correctOutput[j] == round(myOutputs[j][i].value):
                    correctnessCounter += 1.0
                    
                overallCounter += 1.0
                
        print "Got", correctnessCounter, "out of", overallCounter, "correct."
        
        return correctnessCounter / overallCounter        
        
def allSizeXLists(x):
    if x == 0:
        return [[]]
        
    allSizeXMinus1Lists = allSizeXLists(x - 1)
    
    return [littleList + [0.0] for littleList in allSizeXMinus1Lists] + \
        [littleList + [1.0] for littleList in allSizeXMinus1Lists]
        
trainingSet = [[[0.0, 0.0], [0.0]], [[0.0, 1.0], [1.0]], [[1.0, 0.0], [1.0]], [[1.0, 1.0], [0.0]]]

dag = DAG(4, 2, 3, 1)    

dag.train(trainingSet, 4)

dag.testBoolean(trainingSet, True)