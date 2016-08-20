import random
from cubewalker import CubeWalker

def argmax(l):
    bestValue = float("-Inf")
    bestIndex = None
    
    for i, val in enumerate(l):
        if val > bestValue:
            bestValue = val
            bestIndex = i
            
    return bestIndex, bestValue
    

def gateWithInvMaker(gate, invs):
    def computeResult(inputs, params):
        afterInvs = [inp ^ inv for inp, inv in zip(inputs, invs)]
        inputsToGate = listMap(afterInvs, params)
                
        return gate(inputsToGate)
        
    return computeResult
    
def pront(x):
    print x    
    
def listMap(l, itemsToKeep):
    returnList = []
    
    for i, val in enumerate(l):
        if itemsToKeep[i]:
            returnList.append(val)
            
    return returnList

def pand(inputs):
    if 0.0 in inputs:
        return [0.0]
        
    return [1.0]
    
def por(inputs):
    if 1.0 in inputs:
        return [1.0]

    return [0.0]
        
def toOneMinusOne(bit, andNotOr):
    if bit == andNotOr:
        return 1.0
        
    return -1.0        
        
def xor(x, y):
    if x == y:
        return 0.
    return 1.
        
class Intersector:
    def __init__(self, andNotOr, numInputs, inverters):
        self.inputsContaining = {}
        
        self.inverters = inverters
        
        self.tendrils = {}
        
        self.wiresSoFar = [0]*numInputs
        
        GATE_COST = 1.0
        
        # This initialization value describes the 
        # value of the gate relative to a wire.
        # i.e. 2 would mean we think that the cost of
        # the gate is twice the cost of the wire.
        self.totalGateCost = GATE_COST
        
        for i in range(numInputs):
            self.inputsContaining[i] = {}
            self.tendrils[i] = {}
        
        # 1 is and, 0 is or
        self.andNotOr = andNotOr
        
        self.currentSet = {}    
        
        # Tells us if we've done walkOneStep already.
        # For optimization purposes.
        # self.first = True
        # Deprecated
        
            
    # Partitions it into each of the n satisfying-it-alone categories    
    def partitionTrainingSet(self, trainingSet):        
        # The ratio of how well we're doing to how much it costs
        self.currentQuality = 0
        
        for dataPointID, dataPoint in enumerate(trainingSet):
            inputs = dataPoint[0]
            output = dataPoint[1][0]
                                                
            outputQuality = toOneMinusOne(output, self.andNotOr)
                                                
            for i, inp in enumerate(inputs):
                if inp == xor(self.andNotOr, self.inverters[i]):
                    self.inputsContaining[i][dataPointID] = outputQuality
                    self.tendrils[i][dataPointID] = outputQuality
                            
            self.currentSet[dataPointID] = outputQuality   
            self.currentQuality += outputQuality
            
        self.currentQuality /= self.totalGateCost   
                    
#    def computeGateQuality(self, relevantSet):
#        return computeSum(relevantSet) / self.totalGateCost
    
    def computeSum(self, relevantSet):
        return sum(relevantSet.values())
    
    def intersection(self, smallerSet, biggerSet):
        newSet = {}
        for elt, goodness in smallerSet.items():
            if elt in biggerSet:
                newSet[elt] = goodness
                
        return newSet
        
    # Returns whether or not the search is complete    
    def walkOneStep(self):
        
        print self.currentQuality
        
        # See which tendril is the best
        tendrilValues = [(i, self.computeSum(t)) for i, t in self.tendrils.items()]        
        
        bestTendrilIndex, nextValue = argmax([i[1] for i in tendrilValues])
        
        if bestTendrilIndex == None:
            # Then all wires must be 1
            print "---------------"
            return True
        
        bestTendrilIndex = tendrilValues[bestTendrilIndex][0]        
                
        # Adjust for the gate cost
        self.totalGateCost += 1.0
        nextValue /= self.totalGateCost
                
        if nextValue >= self.currentQuality:
            
            self.currentQuality = nextValue
        
            # Pursue that tendril
            self.wiresSoFar[bestTendrilIndex] = 1
            self.currentSet = self.tendrils[bestTendrilIndex]
            del self.inputsContaining[bestTendrilIndex]        
        
            # Make a new set of tendrils
            self.tendrils = {}
            for i, val in self.inputsContaining.items():
                self.tendrils[i] = self.intersection(self.currentSet, val)
            
            return False
            
        else:
            print "---------------"
            return True
            
    def train(self, trainingSet):
        self.partitionTrainingSet(trainingSet)
        
        while not self.walkOneStep():
            pass        
            
    def evaluate(self, inputs):      
        if self.andNotOr:
            return pand(listMap([xor(i, j) for i, j in zip(inputs, self.inverters)], self.wiresSoFar))
            
        else:
            return por(listMap([xor(i, j) for i, j in zip(inputs, self.inverters)], self.wiresSoFar))    

class ConstantReturner:
    def __init__(self, constant):
        self.constant = constant
        
    def evaluate(self, inputs):
        return [self.constant]

class PMAssembler:
    def __init__(self):
        self.totalCost = 0
        
    def getFalsePositives(self, trainingSet):
        positiveMistakes = []
        positiveZeroes = []
        positiveOnes = []
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1][0]
                        
            pmOutput = self.mainGatePM.evaluate(inputs)[0]
            
 #           print dataPoint, output, pmOutput
            
            if (pmOutput == 1) and (output == 0):
                positiveMistakes.append([inputs, [0]])
                positiveZeroes.append([inputs, [0]])
            
 #               print "False positive:", dataPoint
            
            else:
                positiveMistakes.append([inputs, [1]])
                positiveOnes.append([inputs, [1]])
                
        return positiveMistakes, positiveZeroes, positiveOnes
        
    def getFalseNegatives(self, trainingSet):
        negativeMistakes = []
        negativeZeroes = []
        negativeOnes = []
                        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1][0]
            
            pmOutput = self.mainGatePM.evaluate(inputs)[0]
            
#            print dataPoint, output, pmOutput
            
            if (pmOutput == 0) and (output == 1):
                negativeMistakes.append([inputs, [1]])
                negativeOnes.append([inputs, [1]])
                
#                print "False negative:", dataPoint
                
            else:
                negativeMistakes.append([inputs, [0]])
                negativeZeroes.append([inputs, [0]])
                
        return negativeMistakes, negativeZeroes, negativeOnes
        
    # moreOnes encodes whether the result is more often a 1 than a 0.
    # Redundant with trainingSet but saves on computation   
    # Returns whether or not it can fit the entire training set trivially

#    def createMainGate(self, trainingSet, zeroExamples, oneExamples, moreOnes):
#        if moreOnes:
#            if zeroExamples == []:
#                self.mainGatePM = ConstantReturner(1.0)
#                return True
#                
#            canonicalExample = random.choice(zeroExamples)    
#            mainGate = por    
#            
#            inputs = canonicalExample[0]
#            numInputs = len(inputs)
#        
#            f = gateWithInvMaker(mainGate, inputs)
#            
#        else:    
#            if oneExamples == []:
#                self.mainGatePM = ConstantReturner(0.0)    
#                return True
#            
#            canonicalExample = random.choice(oneExamples)    
#            mainGate = pand 
#            
#            inputs = canonicalExample[0]
#            numInputs = len(inputs)
#        
#            f = gateWithInvMaker(mainGate, [1-x for x in inputs])
#            
#        pm = CubeWalker(f, [1]*numInputs)
#        
#        cw.train(trainingSet)
#        
#        self.mainGatePM = cw
#        
#        return False

    def createMainGate(self, trainingSet, zeroExamples, oneExamples, moreOnes):
        if moreOnes:
            if zeroExamples == []:
                self.mainGatePM = ConstantReturner(1.0)
                return True
                
            andNotOr = False
            canonicalExample = random.choice(zeroExamples)[0]    
            
        else:
            if oneExamples == []:
                self.mainGatePM = ConstantReturner(0.0)    
                return True
                
            andNotOr = True
            canonicalExample = [xor(i, 1) for i in random.choice(oneExamples)[0]]
                        
        numInputs = len(trainingSet[0][0])
        
        pm = Intersector(andNotOr, numInputs, canonicalExample)
        pm.train(trainingSet)
        
        self.mainGatePM = pm
        
        self.totalCost += pm.totalGateCost
        
        return False
        
    def trainFixers(self, trainingSet, depth):
        numGates = 0
        
        falsePositives, positiveZeroes, positiveOnes = self.getFalsePositives(trainingSet)
        numGates += self.falsePositiveFixer.train(falsePositives, depth-1, positiveZeroes, positiveOnes, True)
        
        self.totalCost += self.falsePositiveFixer.totalCost
        
        falseNegatives, negativeZeroes, negativeOnes = self.getFalseNegatives(trainingSet)
        numGates += self.falseNegativeFixer.train(falseNegatives, depth-1, negativeZeroes, negativeOnes, False) 
        
        self.totalCost += self.falseNegativeFixer.totalCost
        
        return numGates
        
    def getExamples(self, trainingSet):
        zeroExamples = []
        oneExamples = []
        numZeroes = 0
        numOnes = 0
        
        for dataPoint in trainingSet:
            output = dataPoint[1][0]
            
            if output == 0:
                zeroExamples.append(dataPoint)
                numZeroes += 1
                
            if output == 1:
                oneExamples.append(dataPoint)
                numOnes += 1
                
        return zeroExamples, oneExamples, (numOnes > numZeroes)
        
    def train(self, trainingSet, depth=float("inf"), zeroExamples=None, oneExamples=None, moreOnes=None):
        if zeroExamples == None:
            zeroExamples, oneExamples, moreOnes = self.getExamples(trainingSet)            
            
        trivial = self.createMainGate(trainingSet, zeroExamples, oneExamples, moreOnes)
                
        if trivial:
            numGates = 0
            self.leaf = True
            
            return 0
            
        elif depth == 0:
            numGates = 1
            self.leaf = True
            
            return 1
            
        else:
            self.falsePositiveFixer = PMAssembler()
            self.falseNegativeFixer = PMAssembler()
            self.leaf = False
        
            numGates = self.trainFixers(trainingSet, depth)
            
            return numGates + 1
        
    def evaluate(self, inputs):
        mainGateResult = self.mainGatePM.evaluate(inputs)
        
        if self.leaf:
            return mainGateResult
                
        falsePositiveFixerResult = self.falsePositiveFixer.evaluate(inputs)
        falseNegativeFixerResult = self.falseNegativeFixer.evaluate(inputs)

        return [1*((mainGateResult[0] and falsePositiveFixerResult[0]) \
             or falseNegativeFixerResult[0])]
            
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
                    
            