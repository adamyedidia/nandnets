import random
from bitcounter import *

# Makes an array of None's that is the same dimension as the
# input array
def noneDeepCopy(l, init=None):
    returnList = []
    
    for i in l:
        returnList.append([])
        for j in i:
            returnList[-1].append(init)
            
    return returnList
    
def deepCopy(l):
    returnList = []
    
    for i in l:
        returnList.append(i[:])
        
    return returnList    
    
def listEquals(list1, list2):
    assert len(list1) == len(list2)
        
    for x, y in zip(list1, list2):
        if x != y:
            return False
            
    return True
    
def arrayPlusEquals(array1, array2):
    for i, subArray in enumerate(array2):
        for j, x in enumerate(subArray):
            array1[i][j] += x
            
def arrayDivideEquals(array, scalar):
    for i, subArray in enumerate(array):
        for j in range(len(subArray)):
            array[i][j] /= scalar
    
# Transforms a 2D array into its 1D version. 
def flattenWithZeroes(l): 
    flattenedArray = []
    for miniL in l:
        flattenedArray += miniL
    
    return flattenedArray
 
def orProb(x, y):
     return x + y - x*y

class DAG:
    def __init__(self, weightArray, gateTypeArray):
        assert len(weightArray) == len(gateTypeArray)
        
        self.wArray = weightArray # 2D array 
        self.gateTypeArray = gateTypeArray
        
        # Deprecated, I think?
        self.gArray = noneDeepCopy(weightArray)
        self.condXArray = noneDeepCopy(weightArray)
        
        self.xArray = [None]*len(weightArray)
        self.flawlessXArray = [None]*len(weightArray)
        
        
    def setInputs(self, inputs):
        for i, inp in enumerate(inputs):
            self.xArray[i] = inp
            
    def roundWeights(self):
        for subArray in self.wArray:
            for i, val in enumerate(subArray):
                subArray[i] = round(val)
            
    def extractOutputs(self, numOutputs):
        return self.xArray[-numOutputs:]    
        
    def determineFeedForward(self, inputs):
        dag.setInputs(inputs)
        
        # tracks which way the w's went
        wDecisions = noneDeepCopy(self.wArray, 0.0)
        
        for i in range(len(self.xArray)):
            
            # Later want to change to "if i >= numInputs"    
            if i >= len(inputs): # Don't want to reset the input guys
#            if True:
            
                noActive0Product = 1.0
            
                for j in range(i):
                    whatWWillBe = random.random()
                    
                    x = self.xArray[j]
                    
                    wDecisions[i][j] = 1.0*(self.wArray[i][j] > whatWWillBe)
                    
                    w = 1.0 * wDecisions[i][j]
                
                    noActive0Product *= orProb(x, 1-w)
                
                self.xArray[i] = 1.0 - noActive0Product
                
        return wDecisions
                
    def getRandomWArray(self):
        wDecisions = noneDeepCopy(self.wArray, 0.0)
        
        for i in range(len(wDecisions)):
            for j in range(len(wDecisions[i])):
                
                wDecisions[i][j] = 1.0*(self.wArray[i][j] > random.random())
                                
        return wDecisions 
        
    def getRandomGateTypeArray(self):
        gateTypeDecisions = [0.0] * len(self.gateTypeArray)
        
        for i in range(len(gateTypeDecisions)):
            gateTypeDecisions[i] = 1.0*(self.gateTypeArray[i] > random.random())
            
        return gateTypeDecisions
                
    def flawlessFeedForwardOneStep(self, xArraySoFar, likelihoodSoFar, i, j, noActive0Product, wArraySoFar=[]):      
        assert len(xArraySoFar) == i
          
        if i < len(self.xArray):
#            print "xarr", xArraySoFar, likelihoodSoFar
            x = xArraySoFar[j]
            
#            print likelihoodSoFar
#            print "noActive0", noActive0Product
#            print self.wArray[i][j]
#            print x
            
            noActive0ProductIf1 = noActive0Product * x
            likelihoodSoFarIf1 = likelihoodSoFar * self.wArray[i][j]
            
            noActive0ProductIf0 = noActive0Product
            likelihoodSoFarIf0 = likelihoodSoFar * (1.0 - self.wArray[i][j])
            
#            print noActive0ProductIf1
#            print noActive0ProductIf0
            
            if j+1 == i:
#                if i == 2:
#                    print likelihoodSoFarIf1 * (1.0 - noActive0ProductIf1)
#                    print likelihoodSoFarIf1 * (1.0 - noActive0ProductIf0)
                
                self.flawlessXArray[i] += likelihoodSoFarIf1 * (1.0 - noActive0ProductIf1)
                self.flawlessXArray[i] += likelihoodSoFarIf0 * (1.0 - noActive0ProductIf0)
                
                if i == 3:
                    print xArraySoFar + [noActive0ProductIf1], wArraySoFar + [1.0], likelihoodSoFarIf0 * (1.0 - noActive0ProductIf0)
                    print xArraySoFar + [noActive0ProductIf0], wArraySoFar + [0.0], likelihoodSoFarIf1 * (1.0 - noActive0ProductIf1)
                
                self.flawlessFeedForwardOneStep(xArraySoFar + [noActive0ProductIf1], likelihoodSoFarIf1, 
                    i+1, 0, 1.0, wArraySoFar + [1.0])
                self.flawlessFeedForwardOneStep(xArraySoFar + [noActive0ProductIf0], likelihoodSoFarIf0,
                    i+1, 0, 1.0, wArraySoFar + [0.0])
            
            else:
                self.flawlessFeedForwardOneStep(xArraySoFar, likelihoodSoFarIf1,
                    i, j+1, noActive0ProductIf1, wArraySoFar + [1.0])    
                self.flawlessFeedForwardOneStep(xArraySoFar, likelihoodSoFarIf0,
                    i, j+1, noActive0ProductIf0, wArraySoFar + [0.0])     


    def flawlessFeedForward(self, inputs):       
        print "xArray", self.xArray
         
        dag.setInputs(inputs)
        
        self.flawlessXArray = [0]*len(weightArray)
        
#        self.flawlessFeedForwardOneStep(1.0, len(inputs), 0, 1.0, )
                    
        self.flawlessFeedForwardOneStep(inputs, 1.0, len(inputs), 0, 1.0)
            
  #          print i, self.xArray, self.flawlessXArray
            
        for j in range(len(inputs), len(self.xArray)): 
            self.xArray[j] = self.flawlessXArray[j]
                        
                
    def stochasticFeedForward(self, inputs, numIter=10000):
        xCountArray = [0] * len(self.xArray)
        
        for i in range(numIter):
            self.determineFeedForward(inputs)
            
            for j, x in enumerate(self.xArray):
                                
                xCountArray[j] += x
            
        self.xArray = [i / numIter for i in xCountArray]
                
    def feedForward(self, inputs, auxWArray=None, auxGateTypeArray=None):
        dag.setInputs(inputs)
        
        if auxWArray == None:
            auxWArray = self.wArray
            
        if auxGateTypeArray == None:
            auxGateTypeArray = self.gateTypeArray
        
        nandArray = [None]*len(self.xArray)
        norArray = [None]*len(self.xArray)
        
        for i in range(len(self.xArray)):
            
            if i >= len(inputs):
        
                # It's a NAND!
                xProduct = 1.0
    
                for j in range(i):
        
                    xProduct *= orProb(self.xArray[j], 1-auxWArray[i][j])
        
                nandArray[i] = 1.0 - xProduct    
                
                # It's a NOR!
                xProduct = 1.0
                
                for j in range(i):
                    
                    xProduct *= orProb(1-self.xArray[j], 1-auxWArray[i][j])
                    
                norArray[i] = xProduct
                
                self.xArray[i] = auxGateTypeArray[i] * nandArray[i] + \
                                (1-auxGateTypeArray[i]) * norArray[i]
                
    def evaluate(self, inputs, numOutputs):
        self.feedForward(inputs)
        return self.extractOutputs(numOutputs)
                
    def accurateFeedForward(self, inputs):
        
        dag.setInputs(inputs)
        
        for i in range(len(self.xArray)):
            
            if i >= len(inputs):
            
                # Again, later change to "if i >= numInputs"
    #            if i >= 1:
            
                # Compute x_j | x_{j-1}, x_{j-2}, ...
                # for all j < i
                # for a given set of w_ij
                for j in range(i):
                    if j >= len(inputs):
                
                        condXProduct = 1.0
                        for k in range(j):
                            condXProduct *= (1-self.wArray[i][k]) * self.gArray[j][k] + self.wArray[i][k]
                        
                        self.condXArray[i][j] = 1.0 - condXProduct
                
                    else:
                        self.condXArray[i][j] = self.xArray[j]    
                # Compute g_{ij}
                for j in range(i):
                    self.gArray[i][j] = orProb(self.condXArray[i][j], 1-self.wArray[i][j])
                
                xProduct = 1.0
                
                for j in range(i):
                    xProduct *= self.gArray[i][j]
                
                self.xArray[i] = 1.0 - xProduct
    
    def getAGoodDecision(self, subList):
                
        goodSoFar = False
        
        numTries = 0
        
        while not goodSoFar:
            
            goodSoFar = True
            
            wDecisions = self.getRandomWArray()
            gateTypeDecisions = self.getRandomGateTypeArray()            
                        
            for dataPoint in subList:
                inputs = dataPoint[0]
                correctOutputs = dataPoint[1]
                                
                numOutputs = len(correctOutputs)
                
                self.feedForward(inputs, wDecisions, gateTypeDecisions)
                myOutputs = self.extractOutputs(numOutputs)
            
                goodSoFar = goodSoFar and listEquals(myOutputs, correctOutputs)
            
#            print self.xArray
            
#            print inputs, myOutputs, correctOutputs

            numTries += 1
        
        return wDecisions, gateTypeDecisions, numTries
    
    def train1Epoch(self, trainingSet, trainTier=1, numSamples=1, verbose=False):
        
        # The trainTier refers to how many points must be made to fit at once
        
#        overallWeightArray = noneDeepCopy(self.wArray, 0.0)
#        overallWeightArray = noneDeepCopy(self.wArray, 1.0)

        # This is used only for debugging
        overallXArray = [0.0]*len(self.xArray)
        
#        overallGateTypeArray = [0.0]*len(self.gateTypeArray)
        
        weightIs1LikelihoodArray = deepCopy(self.wArray)
        weightIs0LikelihoodArray = deepCopy(self.wArray)
        
        arrayDivideEquals(weightIs0LikelihoodArray, -1)
        arrayPlusEquals(weightIs0LikelihoodArray, noneDeepCopy(self.wArray, 1.0))
        
        numPoints = 0.0
#        numPoints = 2.0
        
        numTries = 0
        
        for i in range(numSamples):
            allSubListsOfSize = getAllUniqueSortedSubsetsOfSizeX(trainingSet, trainTier)                        
            
            instancesOfWArray = noneDeepCopy(self.wArray, 0.0)
            instancesOfGateTypeArray = [0.0]*len(self.gateTypeArray)
        
#            print allSubListsOfSize            
                                                
            for j, subList in enumerate(allSubListsOfSize):
                                                
                goodDecision, gateTypeDecisions, numTriesThisTime = \
                    self.getAGoodDecision(subList)                
                                        
#                print "Good decision!", goodDecision                
                
                numTries += numTriesThisTime
                                   
 #               print subList
 #               print ""                                                
 #               print goodDecision
 #               print ""
 #               print self.xArray
 #               print "" 
 #               print "-----------"
 #               print ""                                               
                                                                
                arrayPlusEquals(instancesOfWArray, goodDecision)
                
                
                
                overallXArray = [x+y for x, y in zip(overallXArray, self.xArray)]     
                
                overallGateTypeArray = [x+y for x, y in zip(instancesOfGateTypeArray,
                    gateTypeDecisions)]
                
                numPoints += 1.0
                
                if numSamples <= 3 and verbose:
                    print "Mini-mini-epoch", j, "complete."
                
                
                
            if verbose:
                print "Mini-epoch", i, "complete."
            
        arrayDivideEquals(overallWeightArray, numPoints)
                
        overallXArray = [x/numPoints for x in overallXArray]        
                                
        self.gateTypeArray = [x/numPoints for x in overallGateTypeArray]        
                
        self.wArray = overallWeightArray
#        print self.wArray
#        print self.gateTypeArray
#        print overallXArray
        print numTries
            
            
    def train(self, trainingSet, trainTier=1, numEpochs=1, numSamples=1):
        for i in range(numEpochs):
            if numEpochs > 3:
                self.train1Epoch(trainingSet, trainTier, numSamples)  
            else:
                self.train1Epoch(trainingSet, trainTier, numSamples, True)
            
            print "Epoch", i, "complete."     
            
    def testBoolean(self, testSet, randomOutcomes=False, verbose=False):
        correctnessCounter = 0.0
        randomCounter = 0.0
        alwaysZeroCounter = 0.0
        overallCounter = 0.0
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            correctOutputs = dataPoint[1]
            
            numOutputs = len(correctOutputs)
            
            myOutputs = self.evaluate(inputs, numOutputs)
            
            print self.xArray
                        
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
         

def createWeightArray(size, value=-1):
    if value == -1:
        value = 2./size
    
    weightArray = []
    
    for i in range(size):
        if value == "rand":
            weightArray.append([random.random() for _ in range(i)])
        else:
            weightArray.append([value]*i)
    
    return weightArray

def createGateTypeArray(size, value=0.5):
    gateTypeArray = []
    
    for i in range(size):
        if value == "rand":
            gateTypeArray.append(random.random())
        else:
            gateTypeArray.append(value)
            
    return gateTypeArray

def allSizeXLists(x):
    if x == 0:
        return [[]]
        
    allSizeXMinus1Lists = allSizeXLists(x - 1)
    
    return [littleList + [0.0] for littleList in allSizeXMinus1Lists] + \
        [littleList + [1.0] for littleList in allSizeXMinus1Lists]
        
def generateRandomFunc(numInputs, numOutputs):
    inputs = allSizeXLists(numInputs)
    
    trainingSet = []
    
    for inp in inputs:
        output = [random.randint(0, 1) for _ in range(numOutputs)]
        trainingSet.append([inp, output])
        
    testSet = trainingSet[:]
    
    return trainingSet, testSet
    
def generateMajorityFunc(numInputs, threshold):
    inputs = allSizeXLists(numInputs)
    
    trainingSet = []
    
    for inp in inputs:
        output = [sum(inp) >= threshold]
        trainingSet.append([inp, output])
        
    testSet = trainingSet[:]
    
    return trainingSet, testSet
        
def incrementList(l, maxValue):
    assert maxValue > 0
    
    return incrementListIndex(l, maxValue, 0)
    
# returns overflow    
def incrementListIndex(l, maxValue, index):
    if index == len(l):
        return True
        
    l[index] += 1
    
    if l[index] == maxValue:        
        l[index] = 0
        return incrementListIndex(l, maxValue, index+1)
        
    return False    
                        
def getAllSublistsOfSizeX(l, x):
    returnList = []
    
    indexList = [0]*x
    overflow = False
    
    while not overflow:
        returnList.append([l[i] for i in indexList])
        overflow = incrementList(indexList, len(l))
        
    return returnList
    
def getAllUniqueSortedSubsetsOfSizeX(l, x):    
    if x > len(l):
        return None
    
    if x == 0:
        return [[]]    
        
    else:
        returnList = []
        
        for i, val in enumerate(l):
            
            smallerLists = getAllUniqueSortedSubsetsOfSizeX(l[i+1:], x-1)
            
            if not smallerLists == None:
            
                returnList.extend([[val] + smallerList for smallerList in \
                    smallerLists])
        
        return returnList        
        
#numInputs = 10
#numOutputs = 1

#trainingSet, testSet = generateRandomFunc(numInputs, numOutputs)

#numInternals = 30

#weightArray = createWeightArray(numInputs+numInternals+numOutputs)

#print getAllUniqueSortedSubsetsOfSizeX([1,2,3,4], 0)

#dag = DAG(weightArray)

#dag.train(trainingSet, 1000, 100)

#dag.roundWeights()
#dag.testBoolean(testSet, True, True)        

majorityTest = False

if majorityTest:
    
    numInputs = 5
    numInternals = 10
    
    numGates = numInputs+numInternals+1
    
    weightArray = createWeightArray(numGates, 0.5)
    gateTypeArray = createGateTypeArray(numGates, 0.0)
 
    dag = DAG(weightArray, gateTypeArray)
    
    trainingSet, testSet = generateMajorityFunc(numInputs, numInputs/2)
        
    dag.train(trainingSet, 1, 1000, 10)
    
    dag.testBoolean(testSet, True, True)
    
 
simpleGateTest = False 
 
if simpleGateTest: 
        
    weightArray = createWeightArray(6, 0.5)     

    gateTypeArray = createGateTypeArray(6, 1.0)
          
    dag = DAG(weightArray, gateTypeArray)

    trainingSet = [[[0.0, 0.0], [1.0]],
                    [[0.0, 1.0], [1.0]], 
                    [[1.0, 0.0], [1.0]],
                    [[1.0, 1.0], [0.0]]]
                
    dag.train(trainingSet, 1, 100, 100)
    #dag.roundWeights()
    print dag.evaluate([0.0, 0.0], 1)
    print dag.evaluate([0.0, 1.0], 1)
    print dag.evaluate([1.0, 0.0], 1)
    print dag.evaluate([1.0, 1.0], 1)
 
trivialGateTest = True 

if trivialGateTest:
    
    weightArray = createWeightArray(3, 0.43)
    gateTypeArray = createGateTypeArray(3, 0.43)
    
    dag = DAG(weightArray, gateTypeArray)
    
    trainingSet = [[[0.0, 0.0], [1.0]],
                    [[0.0, 1.0], [1.0]], 
                    [[1.0, 0.0], [1.0]],
                    [[1.0, 1.0], [0.0]]]
                    
    dag.train(trainingSet, 1, 1000, 100)
    #dag.roundWeights()
    print dag.evaluate([0.0, 0.0], 1)
    print dag.evaluate([0.0, 1.0], 1)
    print dag.evaluate([1.0, 0.0], 1)
    print dag.evaluate([1.0, 1.0], 1)                     
 
compressionTest = False 
 
if compressionTest:

    patternSize = 20
    numInternals = 50

    trainingSetSize = 1000
    testSetSize = 1000

    weightArray = createWeightArray(patternSize+numInternals+1)
    gateTypeArray = createGateTypeArray(patternSize+numInternals+1, 0.5)

    dag = DAG(weightArray, gateTypeArray)

    compressionData = getCompressionData("../../compression/declarationbits.txt", patternSize)

    trainingSet = compressionData[:trainingSetSize]
    testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]

    dag.train(trainingSet, 1, 4, 1)

    #dag.roundWeights()

    dag.testBoolean(testSet, True, True)

    print dag.wArray
    print dag.gateTypeArray
    
    print dag.xArray
    

#inp = [0.0]

#dag.accurateFeedForward(inp)
#print dag.extractOutputs(1)
#print dag.xArray

#dag.feedForward(inp)
#print dag.extractOutputs(1)      
#print dag.xArray          
                
#dag.stochasticFeedForward(inp, 100000)
#print dag.extractOutputs(1)     
#print dag.xArray           

#dag.flawlessFeedForward(inp)
#print dag.xArray                
				
#print 45./64                
#            else:
#                self.condXArray[i] = self.xArray[i]