from cubewalker import CubeWalker
from cubeanalyzer import CubeAnalyzer
from sampler import Sampler
from softsampler import SoftSampler
from bitcounter import getCompressionData
from randominstancegen import randomInstanceGen, generateFullTruthTable, \
    generateRandomPartialTruthTable
from memorymachine import MemoryMachine    
from majoritarian import Majoritarian
import math
import random    
import time

def allListsOfSizeX(x):
    if x == 0:
        return [[]]
        
    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0] for i in oneLess] + [i + [1] for i in oneLess]  

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

def generateRandomTruthTable(numInputs, numOutputs):
    trainingSet = []
    
    allInputs = allListsOfSizeX(numInputs)
    
    for i in allInputs:
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
    
def pand(inputs):
    if 0.0 in inputs:
        return 0.0
        
    return 1.0
    
def por(inputs):
    if 1.0 in inputs:
        return 1.0

    return 0.0
    
def xorList(inputs):
    return (sum(inputs) % 2) == 0
    
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

def nandDagFunctionMaker(numInputs, numInternals, numOutputs):
    def nandDag(inputs, params):
#        t=time.time()
        assert len(inputs) == numInputs
        
        valuesSoFar = inputs[:]
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
                
#        print time.time()-t        
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
    
#        print inputs, params[:numInputs], pad
    
        assert len(inputs) == numInputs
        assert len(params) == 2*numInputs
    
        neutralAgreement = agreementMaker(numInputs, numInputs/2+1*(numInputs%2), False)
    
        inputsToFinal = [neutralAgreement(inputs, bitwiseXor(bit, params[:numInputs]), i) for i, bit in enumerate(pad)]
        
#        print [(inputs, bitwiseXor(bit, params[:numInputs]), i) for i, bit in enumerate(pad)] 
#        print inputsToFinal
        
#        print inputsToFinal, params[numInputs:]
    
        finalAgreement = agreementMaker(numInputs, threshold, True)
    
#        print finalAgreement(inputsToFinal, params[numInputs:])
    
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
#        print time.time()-t
        return result
        
    return randomXORAgreement