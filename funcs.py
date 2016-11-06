from cubewalker import CubeWalker
from cubeanalyzer import CubeAnalyzer
from sampler import Sampler
from softsampler import SoftSampler
from bitcounter import getCompressionData
from randominstancegen import randomInstanceGen, generateFullTruthTable, \
    generateRandomPartialTruthTable
from memorymachine import MemoryMachine
from majoritarian import Majoritarian
from funcsum import FuncSum, FuncSumClique
from matfuncsum import MatFuncSum, MatFuncSumClique
from pmassembler import PMAssembler
import numpy as np
import scipy.optimize as opt
import math
import random
import time
import gzip
import cPickle

def product(l):
    val = 1.0

    for i in l:
        val *= i

    return val

def allListsOfSizeX(x):
    if x == 0:
        return [[]]

    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0] for i in oneLess] + [i + [1] for i in oneLess]


def convertToOneMinusOne(x):
    if x == 0:
        return -1.0

    return 1.0


def allSubsetsOfSize(x, numOnesLeft):
    if numOnesLeft == 0:
        return [[0]*x]

    elif numOnesLeft == x:
        return [[1]*x]

    else:
        oneLessWithOne = allSubsetsOfSize(x-1, numOnesLeft-1)
        oneLessWithZero = allSubsetsOfSize(x-1, numOnesLeft)

        return [i + [0] for i in oneLessWithZero] + [i + [1] for i in oneLessWithOne]

def pand(inputs):
    if 0.0 in inputs:
        return 0.0

    return 1.0

def por(inputs):
    if 1.0 in inputs:
        return 1.0

    return 0.0

def allTuplesOfSizeX(x):
    if x == 0:
        return [tuple([])]

    else:
        oneLess = allTuplesOfSizeX(x-1)
        return [i + tuple([0]) for i in oneLess] + [i + tuple([1]) for i in oneLess]

def xor(x, y):
    return (x+y)%2

def bitwiseXor(arr1, arr2):
    return [xor(i, j) for i, j in zip(arr1, arr2)]

def bitwiseNot(arr):
    return [1.0 - i for i in arr]

def makeOrthogonalSet(logSetSize):
    if logSetSize == 0:
        return [[0.0]]

    else:
        returnList = []
        for i in makeOrthogonalSet(logSetSize-1):
            returnList.append(i+i)
            returnList.append(i+bitwiseNot(i))

        return returnList

def generateRandomTruthTable(numInputs, numOutputs, oneMinusOne=False, itsAllReal=False):
    trainingSet = []

    allInputs = allListsOfSizeX(numInputs)

    for i in allInputs:
        if oneMinusOne:
            trainingSet.append([i, [convertToOneMinusOne(random.randint(0, 1)) for _ in range(numOutputs)]])
        else:
            if itsAllReal:
                trainingSet.append([i, [random.random() for _ in range(numOutputs)]])
            else:
                trainingSet.append([i, [random.randint(0, 1) for _ in range(numOutputs)]])


    return trainingSet

def nand(inputs):
    if 0.0 in inputs:
        return 1.0

    return 0.0

def nor(inputs):
    if 1.0 in inputs:
        return 0.0

    return 1.0

def xorList(inputs):
    return 1.0*((sum(inputs) % 2) == 0)

def listMap(l, itemsToKeep):
    returnList = []

    for i, val in enumerate(l):
        if itemsToKeep[i]:
            returnList.append(val)

    return returnList

def tinyNandNor(inputs, params):
    # First param is whether first input is on
    # Second param is whether second input is on
    # Third param is whether the entire gate is a NAND or a NOR

    if params[2]:
        return [nand(listMap(inputs, params[:2]))]

    else:
        return [nor(listMap(inputs, params[:2]))]

def andWithNot(inputs, params):
    # First param is whether first input is on
    # Second param is whether second input is on
    # Third param is whether there is a NOT in front of the AND

    andResult = pand(listMap(inputs, params[:2]))

    if params[2]:
        return 1-andResult
    else:
        return andResult

def tinyNandNorBinOutput(inputs, params):
    # First param is whether first input is on
    # Second param is whether second input is on
    # Third param is whether the entire gate is a NAND or a NOR

    if params[2]:
        return nand(listMap(inputs, params[:2]))

    else:
        return nor(listMap(inputs, params[:2]))

def tinyNandNorImproved(inputs, params):
    # This is the same as tinyNandNor, but with the added benefit that when the
    # gate is a NOR that everything is reversed

    if params[2]:
        return [nand(listMap(inputs, params[:2]))]

    else:
        # Reverse all wires' onness
        return [nor(listMap(inputs, [1.0-i for i in params[:2]]))]

def andOrHybrid(inputs, paramSetAND, paramSetOR, isNand):
    # A nand or a nor of an AND of a bunch of things, and an OR of a bunch of things
    andResult = pand(listMap(inputs, paramSetAND))
    orResult = por(listMap(inputs, paramSetOR))

    if isNand:
        return nand([andResult, orResult])

    else:
        return nor([andResult, orResult])

def bigAnd(inputs, params):
    return [pand(listMap(inputs, params))]

def nandDagOneOutputFunctionMaker(numInputs, numGates):
    nandDag = nandDagFunctionMaker(numInputs, numGates-1, 1)

    def nandDagOneOutput(inputs, params):
        return nandDag(inputs, params)[0]

    return nandDagOneOutput

def nandDagFunctionMaker(numInputs, numInternals, numOutputs):
    def nandDag(inputs, params):
#        t=time.time()
        assert len(inputs) == numInputs

        valuesSoFar = list(inputs[:])
        numValuesSoFar = numInputs

        paramIndex = 0

        for i in range(numInternals):
            result = nand(listMap(valuesSoFar, params[paramIndex:paramIndex+numValuesSoFar]))

            paramIndex += numValuesSoFar
            valuesSoFar += [result]
            numValuesSoFar += 1

        output = []

        for i in range(numOutputs):
            result = nand(listMap(valuesSoFar, params[paramIndex:paramIndex+numValuesSoFar]))

            paramIndex += numValuesSoFar
            output += [result]

        return output

    return nandDag

def hybridFunctionMaker(numInputs, numInternals, numOutputs):
    # Builds mega-gates out of a NAND or a NOR of ANDs and ORs
    def hybridFunc(inputs, params):
        assert len(inputs) == numInputs

        valuesSoFar = inputs[:]
        numValuesSoFar = numInputs

        paramIndex = 0

        for i in range(numInternals):
            paramSetAND = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar
            paramSetOR = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar

            result = andOrHybrid(valuesSoFar, paramSetAND, paramSetOR, True)

            valuesSoFar += [result]
            numValuesSoFar += 1

        output = []

        for i in range(numOutputs):
            paramSetAND = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar
            paramSetOR = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar

            result = andOrHybrid(valuesSoFar, paramSetAND, paramSetOR, True)
            output += [result]

        return output

    return hybridFunc

def andOrHybridClever(inputSet1, inputSet2, paramSet1, paramSet2):
    # Returns two things; one is where it goes (andset, orset, nand) and the other where
    # it goes (orset, andset, nor)

    andResult1 = pand(listMap(inputSet2, paramSet1))
    orResult1 = por(listMap(inputSet1, paramSet1))

    andResult2 = pand(listMap(inputSet1, paramSet2))
    orResult2 = por(listMap(inputSet2, paramSet2))

    return nand([andResult1, orResult2]), nor([andResult2, orResult1])

def hybridFunctionMakerClever(numInputs, numInternals, numOutputs):
    # Builds mega-gates out of a NAND or a NOR of ANDs and ORs
    def hybridFuncClever(inputs, params):
        assert len(inputs) == numInputs

        valuesSoFar1 = inputs[:]
        valuesSoFar2 = inputs[:]
        numValuesSoFar = numInputs

        paramIndex = 0

        for i in range(numInternals):
            paramSet1 = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar
            paramSet2 = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar

            result1, result2 = andOrHybridClever(valuesSoFar1, valuesSoFar2, paramSet1, paramSet2)

            valuesSoFar1 += [result1]
            valuesSoFar2 += [result2]

            numValuesSoFar += 1

        output = []

        for i in range(numOutputs):
            paramSet1 = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar
            paramSet2 = params[paramIndex:paramIndex+numValuesSoFar]
            paramIndex += numValuesSoFar

            result1, result2 = andOrHybridClever(valuesSoFar1, valuesSoFar2, paramSet1, paramSet2)

            output += [result1]

        return output

    return hybridFuncClever

def agreementMaker(numInputs, threshold, standAlone=True):
    def agreement(inputs, params, tiebreakerBitIndex=None):
        assert len(inputs) == numInputs
        assert len(params) == numInputs

        counter = 0

        for i in range(numInputs):
            if inputs[i] == params[i]:
                counter += 1

        if counter > threshold:
            if standAlone:
                return [1.0]
            else:
                return 1.0

        elif counter == threshold:
            if standAlone:
                return [1.0]
            else:
                if tiebreakerBitIndex == None:
                    return 1.0
                else:
                    return 1.0*(inputs[tiebreakerBitIndex] == params[tiebreakerBitIndex])

        else:
            if standAlone:
                return [0.0]
            else:
                return 0.0

    return agreement

def agreementSquareMaker(numInputs, threshold, pad=None):
    if pad == None:
        pad = [[random.randint(0, 1) for _ in range(numInputs)] for _ in range(numInputs)]


    def agreementSquare(inputs, params):

        assert len(inputs) == numInputs
        assert len(params) == 2*numInputs

        neutralAgreement = agreementMaker(numInputs, numInputs/2+1*(numInputs%2), False)

        inputsToFinal = [neutralAgreement(inputs, bitwiseXor(bit, params[:numInputs]), i) for i, bit in enumerate(pad)]


        finalAgreement = agreementMaker(numInputs, threshold, True)

        return finalAgreement(inputsToFinal, params[numInputs:])

    return agreementSquare

def numParamCalculator(numInputs, numInternals, numOutputs):
    numParams = 0
    numValuesSoFar = numInputs

    for i in range(numInternals):
        numParams += numValuesSoFar
        numValuesSoFar += 1

    for i in range(numOutputs):
        numParams += numValuesSoFar

    return numParams

#def randomFunctionMaker(numInputs, numOutputs, seedSpaceSize=1000000):
#    allTuples = allTuplesOfSizeX(numInputs)

#    tt = {}

#    for tup in allTuples:
#        tt[tup] = [random.randint(0, 1) for _ in range(numOutputs)]

#    def randomFunc(inputs):
#        return tt[tuple(inputs)]

#    return randomFunc

def randomFunctionMaker(numInputs, numOutputs):
    functionSeed = random.random()

    def randomFunc(inputs):
        entrySeed = (tuple(inputs), functionSeed)
        random.seed(entrySeed)
        entry = random.randint(0, 1)
        random.seed()
        return entry

    return randomFunc

def randomAgreementMaker(numInputs, numParams, threshold):
    funcs = [randomFunctionMaker(numInputs, 1) for _ in range(numParams)]

    def randomAgreement(inputs, params):
        agreement = agreementMaker(numParams, threshold, True)

        inputsToAgreementGate = [f(inputs) for f in funcs]
        return agreement(inputsToAgreementGate, params)

    return randomAgreement

def xorMap(inputs, xorMapList):
    return xorList(listMap(inputs, xorMapList))

def randomXORAgreementMaker(numInputs, numParams, threshold):
    alreadyDone = {}

    numTupsFound = 0

    while numTupsFound < numParams:
        randomTup = tuple(float(random.randint(0, 1)) for _ in range(numInputs))

        if (not (randomTup in alreadyDone)) and (sum(randomTup) != 0):
            alreadyDone[randomTup] = None
            numTupsFound += 1

    xorMapList = alreadyDone.keys()

    agreement = agreementMaker(numParams, threshold, True)

    def randomXORAgreement(inputs, params):
#        t = time.time()
        inputsToAgreementGate = [xorMap(inputs, i) for i in xorMapList]
        result = agreement(inputsToAgreementGate, params)
        return result

    return randomXORAgreement

def xorSubsetFuncMaker(subset):
    def xorSubset(inputs):
        return xorMap(inputs, subset)

    return xorSubset

def allXORSubsetFuncMaker(numInputs, setOfSubsets):
    returnList = []

    for subset in setOfSubsets:

        returnList.append(xorSubsetFuncMaker(subset))

    return returnList

def subsetMultiplyFuncMaker(subset):
    def multSubset(inputs):
        return product(listMap(inputs, subset))

    return multSubset

def allMultSubsetFuncMaker(numInputs, setOfSubsets):
    returnList = []

    for subset in setOfSubsets:

        returnList.append(subsetMultiplyFuncMaker(subset))

    return returnList

def getRawMnistData():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    rawTrainingData, rawValidationData, rawTestData = cPickle.load(f)
    f.close()

    return (rawTrainingData, rawValidationData, rawTestData)

def makeTruthTable(numInputs, listOfOutputs):
    assert len(listOfOutputs) == 2**numInputs

    trainingSet = []

    allInputs = allListsOfSizeX(numInputs)

    for i, inp in enumerate(allInputs):
        trainingSet.append([inp, [listOfOutputs[i]]])

    return trainingSet

def processMnistData(rawTrainingData, rawValidationData, rawTestData):

    trainingData = []
    for i in range(len(rawTrainingData[0])):
        trainingData.append([rawTrainingData[0][i], rawTrainingData[1][i]])

    validationData = []
    for i in range(len(rawValidationData[0])):
        validationData.append([rawValidationData[0][i], rawValidationData[1][i]])

    testData = []
    for i in range(len(rawTestData[0])):
        testData.append([rawTestData[0][i], rawTestData[1][i]])

    return (trainingData, validationData, testData)

def basicFuncMaker(numInputs, index):
    def basicFunc(l):
        return l[index]

    return basicFunc

# Makes a family of functions that are just f_i(\vec{x}) = x_i
# In other words, just take the i^th input. The simplest family of functions
# you could ask for!
def basicFuncFamilyMaker(numInputs):
    return [basicFuncMaker(numInputs, i) for i in range(numInputs)]

# Assume the function is linear. As a function of the coeffs, what would the
# sum-product for two different functions be?
def linearFuncSumTSMakerLargeTS(trainingSet, numInputs):
    runningSum = np.zeros((numInputs, numInputs))

    for dataPoint in trainingSet:
        inputs = dataPoint[0]

        runningSum += np.outer(np.array(inputs), np.array(inputs))

    for i in range(numInputs):
        for j in range(numInputs):
            print runningSum[i][j]

    def computeSumAnswersOverTS(coeffs1, coeffs2):
        return sum(sum(np.multiply(np.outer(coeffs1, coeffs2), runningSum)))

    return computeSumAnswersOverTS

# Find the best set of linear orthogonal functions you can!
def findBestLinearOrthogFuncsLargeTSOld(trainingSet):
    numInputs = len(trainingSet[0][0])

    computeSumAnswersOverTS = linearFuncSumTSMaker(trainingSet, numInputs)

    def computeFitnessOfCoeffs(coeffs):
        runningSum = 0

        coeffArray = np.reshape(coeffs, (numInputs, numInputs))

        for i in range(len(coeffArray)):
            for j in range(len(coeffArray)):
                if i != j:
                    runningSum += computeSumAnswersOverTS(coeffArray[i],
                        coeffArray[j]) ** 2

            print i

        print runningSum

        return runningSum

    # matrix of random values between -1 and 1
    initMatrix = 2*(0.5*np.ones((numInputs,numInputs))-np.random.rand(numInputs,numInputs))

    # gradient descent
    print opt.minimize(computeFitnessOfCoeffs, initMatrix, method="BFGS",
        options={'disp':True})

def findBestLinearOrthogFuncs(rawTrainingData):
    numInputs = len(rawTrainingData[0])
    # We need to compute CX^T XC^T

#    print rawTrainingData
#    print np.transpose(rawTrainingData)

    middleMatrix = np.dot(np.transpose(rawTrainingData), rawTrainingData)
    # middleMatrix is X^T X

    def computeFitnessOfCoeffs(coeffs):
        coeffArray = np.reshape(coeffs, (numInputs, numInputs))

        result = sum(sum((np.dot(np.dot(coeffArray, middleMatrix),
            np.transpose(coeffArray)) - np.identity(numInputs))))

        print result

        return result

    # matrix of random values between -1 and 1
    initMatrix = 2*(0.5*np.ones((numInputs,numInputs))-np.random.rand(numInputs,numInputs))

    # gradient descent
    print opt.minimize(computeFitnessOfCoeffs, initMatrix, method="BFGS",
        options={'disp':True})

# Gets rid of all pixels that only ever have 0.0 in the training set
# (this avoids a singular matrix later)
def removeDudsFromTrainingData(rawTrainingData):
    # This could probably be made faster...
    rawTrainingData = np.transpose(rawTrainingData)

    prunedTrainingData = []

    for pixelHistory in rawTrainingData:

        if sum(pixelHistory):
            prunedTrainingData.append(pixelHistory)

    return np.transpose(np.array(prunedTrainingData))

# TRY ITERATIVE SOLUTION!!
def findBestLinearOrthogFuncsIter(prunedTrainingData):

    numInputs = len(prunedTrainingData[0])

#    shiftedTrainingData = rawTrainingData - 0.5*np.ones((len(rawTrainingData), numInputs))


    middleMatrix = np.dot(np.transpose(prunedTrainingData), prunedTrainingData)
    # middleMatrix is X^T X

    w, v = np.linalg.eig(middleMatrix)

    print w
    print v

    print np.dot(np.transpose(v), v)

    print "o", middleMatrix

    print "p", np.dot(np.dot(v, np.diag(w)), np.transpose(v))

    print "r", np.allclose(np.dot(np.dot(v, np.diag(w)), np.transpose(v)), middleMatrix)

    print "q", np.dot(np.dot(np.transpose(v), middleMatrix), v)

    checkThatFuncsAreOrthogonal(prunedTrainingData, np.transpose(v))

    raise

    print "middleMatrix computed"

    # C <- C^T^-1 (X^T X)^-1
#    for i in range(784):
#        printString = ""
#        for j in range(784):
#            printString += "|" + str(middleMatrix[i][j])
#        print printString

    middleMatrixInv = np.linalg.inv(middleMatrix)

    # matrix of random values between -1 and 1
    coeffArray = 2*(0.5*np.ones((numInputs,numInputs))-np.random.rand(numInputs,numInputs))

    notDoneYet = True
    while notDoneYet:
        for _ in range(1):
            coeffArray = np.dot(np.linalg.inv(coeffArray.transpose()), middleMatrixInv)
#            coeffArray = np.dot(middleMatrix, coeffArray.transpose())

#            for i in range(784):
#                printString = ""
#                for j in range(784):
#                    printString += "|" + str(coeffArray[i][j])
#                print printString

#            coeffArray = np.linalg.inv(coeffArray)


        currentApproxOfId = np.dot(np.dot(coeffArray, middleMatrix),
            coeffArray.transpose())

        distanceFromId = sum(sum(np.multiply(currentApproxOfId, currentApproxOfId)))
        print distanceFromId

        if distanceFromId < 0.01:
            notDoneYet = False

    return coeffArray

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

def justThatInputMaker(i):
    def justThatInputs(inputs):
        return inputs[i]
    return justThatInputs

def perceptronMaker(N):
    def perceptron(inputs, params):
        assert len(inputs) == N

        return 1*(sum([i==j for i,j in zip(inputs, params)]) >= (N/2.))

    return perceptron

def gateFuncMaker(n, metaParamMatrix):
    def gate(inputs, params):
        positiveOnes = sum([i*j for i, j in zip(inputs, params)])
        negativeOnes = sum([i*(1-j) for i, j in zip(inputs, params)])

        return metaParamMatrix[sum(params)][positiveOnes][negativeOnes]

    return gate

# a pile of n gates
def gatePileMakerMaker(n, gate):
    def gatePileMaker(paramMatrix):
        def gatePile(inputs):
            nextInputsToGate = inputs[:]

            for i in range(n):
                nextInputsToGate = nextInputsToGate[1:] + [gate(nextInputsToGate, paramMatrix[i])]

            return nextInputsToGate[-1]
        return gatePile
    return gatePileMaker

def gatePileMetaMakerMaker(n):
    def gatePileMetaMaker(metaParamMatrix):
        gateFunc = gateFuncMaker(n, metaParamMatrix)
        gatePileMaker = gatePileMakerMaker(n, gateFunc)
        return gatePileMaker

    return gatePileMetaMaker

# WARNING: param3dArray must break dimension on the last layer!
# currently deprecated
def gateGridMaker(n, gate, numLayers):
    def gateGrid(inputs, param3dArray):
        inputsToNextLayer = inputs

        for layer in range(numLayers):
            outputsFromThisLayer = [gate(inputsToNextLayer, param3dArray[layer][i]) for i in range(n)]
            inputsToNextLayer = outputsFromThisLayer

        return gate(inputsToNextLayer, param3dArray[-1])

def smallPerceptronStack(inputs, params):
    # two threes feeding into a two

    firstResult = perceptronMaker(3)(inputs[0:3], params[0:3])
    secondResult = perceptronMaker(3)(inputs[3:6], params[3:6])
    thirdResult = perceptronMaker(2)([firstResult, secondResult], params[6:8])

    return thirdResult
