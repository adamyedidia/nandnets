import random

def allListsOfSizeX(x):
    if x == 0:
        return [[]]
    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0.] for i in oneLess] + [i + [1.] for i in oneLess]

def randomInstanceGen(func, numParams, p=0.5):
    randomParams = [1.0*(random.random()<p) for i in range(numParams)]
        
    return lambda x : func(x, randomParams), randomParams
    
def generateFullTruthTable(func, numInputs):
    allInputs = allListsOfSizeX(numInputs)
    
    trainingSet = []
    
    for inputs in allInputs:
                
        trainingSet.append([inputs, func(inputs)])
            
    return trainingSet
    
def generateRandomPartialTruthTable(func, numInputs, numDataPoints):
    trainingSet = []
    
    for _ in numDataPoints:
        inputs = [1.0*(random.random()>0.5) for i in range(numInputs)]
                
        trainingSet.append([inputs, func(inputs)])
    
    return inputs
    
