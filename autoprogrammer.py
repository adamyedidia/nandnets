import random
import sys
import traceback

def zero(l):
    return 0

def c20(x):
    return (x == 0)
    
def incr(l):
    return l[0] + 1
    
def decr(l):
    return l[0] - 1
    
def listSum(listOfLists):
    returnList = []
    
    for l in listOfLists:
        returnList += l
        
    return returnList    
    
    
def stringSum(listOfStrings):
    returnString = ""
    
    for s in listOfStrings:
        returnString += s
        
    return returnString  
    
    
MAX_INPUTS = 2

def makeInputTransformer(i):
    def inputTransformer(inps):
        return inps[i]
    return inputTransformer

inputTransformers = [(makeInputTransformer(i), "inp" + str(i), 0) for i in range(MAX_INPUTS)] 

listOfFuncs = [(zero, "zero", 0), (incr, "incr", 1), (decr, "decr", 1)]   
    
    
def generateRandomExpression(primitives, numInputs, maxDepth):
    relevantTransformers = inputTransformers[:numInputs]
    
    if maxDepth == 0:
        randomChoice = random.choice([(zero, "zero", 0)] + relevantTransformers)
        
    else:            
        randomChoice = random.choice(primitives + relevantTransformers)    
    
    if randomChoice[1][:3] == "inp":            
        return (randomChoice[0], randomChoice[1] + "()")
    
    else:
        numInputsInnerFunc = randomChoice[2]
       
        stringSoFar = randomChoice[1] + "("
       
        innerFuncSubFuncs = []
       
        for _ in range(numInputsInnerFunc): 
            newRandomExpression = generateRandomExpression(primitives, numInputs, maxDepth-1)
            innerFuncSubFuncs.append(newRandomExpression[0])
            stringSoFar += newRandomExpression[1] + ","
            
        def randomExpression(inputs):
            return randomChoice[0]([f(inputs) for f in innerFuncSubFuncs])
            
        return (randomExpression, stringSoFar + ")")        
        
    
    
def makeNewFunction(primitives, numInputs, maxDepth, name):
    returnString = "def " + name + "(" + \
        stringSum(["inp" + str(i) + "," for i in range(numInputs)]) + "):\n"
    
    compareToZeroExpr = generateRandomExpression(primitives, numInputs, maxDepth)
    returnString += "    if c20(" + compareToZeroExpr[1] + "):\n"
    
    baseCaseExpr = generateRandomExpression(primitives, numInputs, maxDepth)
    returnString += "        return " + baseCaseExpr[1] + "\n"
    
    innerFuncSubFuncs = []
    returnString += "    return " + name + "("
    
    for _ in range(numInputs):
        newRandomExpression = generateRandomExpression(primitives, numInputs, maxDepth)
        innerFuncSubFuncs.append(newRandomExpression[0])
        returnString += newRandomExpression[1] + ","
    
    returnString += ")\n"
    
    def newFunction(inputs):
        if c20(compareToZeroExpr[0](inputs)):
            return baseCaseExpr[0](inputs)
            
        return newFunction([f(inputs) for f in innerFuncSubFuncs])
        
    return (newFunction, returnString)
    
def tryStuffWithoutRemembering(trainingSet, primitives, numInputs, maxDepth, name):
    goodSoFar = False
    
    while not goodSoFar:
        funcToTry = makeNewFunction(primitives, numInputs, maxDepth, name)
        
        goodSoFar = True
        
        for dataPoint in trainingSet:
            try:
                if funcToTry[0](dataPoint[0]) != dataPoint[1]:
                    goodSoFar = False
                    
            except RuntimeError:
                # Endless recursion
                goodSoFar = False
                break
                
    return funcToTry
    
def tryStuffWithRemembering(trainingSet, primitives, numInputs, maxDepth, name):
    goodSoFar = False
    
    listOfFuncs = primitives[:]
    
    attemptCounter = 0
    
    while not goodSoFar:
        funcName = name + "_attempt_" + str(attemptCounter)
                
        funcToTry = makeNewFunction(listOfFuncs, numInputs, maxDepth, funcName)
        
        goodSoFar = True
        noErrorsSoFar = True
        
        for dataPoint in trainingSet:            
            try:
                if funcToTry[0](dataPoint[0]) != dataPoint[1]:
                    goodSoFar = False
                    
            except RuntimeError as e:
                # Endless recursion
                goodSoFar = False
                noErrorsSoFar = False
                
 #               exc_type, exc_value, exc_traceback = sys.exc_info()
 #               traceback.print_tb(exc_traceback, limit=10, file=sys.stdout)
                
                break
                
                
                
        if noErrorsSoFar:
            listOfFuncs.append((funcToTry[0], funcName, numInputs))
                    
        attemptCounter += 1
                                                            
    return funcToTry    
    
trainingSet = [[[1,1], 2], [[1,2], 3]]    
    
f = tryStuffWithRemembering(trainingSet, listOfFuncs, 2, 1, "add")

print f[1]  

print f[0]([3,1])