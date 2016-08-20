import numpy as np

EPSILON = 0.00001
LESS_SERIOUS_EPSILON = 0.1

def orProb(x, y):
    return x + y - x*y

def fMaker(params):
    assert len(params) == 3
    
    def f(inputs):
        assert len(inputs) == 2
        
        for i in params:
            if i > 1.0 or i < -0.0:
                return i/LESS_SERIOUS_EPSILON
        
        a1, a2, a3 = params[0], params[1], params[2]
        
        i1, i2 = inputs[0], inputs[1]
        
        nand = 1-orProb(1-a1, i1)*orProb(1-a2,i2)
        nor = orProb(1-a1, 1-i1)*orProb(1-a2, 1-i2)
        return a3*nand + (1-a3)*nor
        
    return f
    
def errorFunc(params, trainingSet):
    f = fMaker(params)
    
    runningSum = 0
    
    for dataPoint in trainingSet:
        inputs = dataPoint[0]
        output = dataPoint[1][0]
        
        runningSum += (f(inputs)-output)**2
        
    return runningSum        
        
def findGradient(errorFunc, currentParams, trainingSet):
    numParams = len(currentParams)
    global EPSILON
    
    currentError = errorFunc(currentParams, trainingSet)
    
    gradient = np.zeros(numParams)
    
    for i in range(numParams):
        slightlyDifferentParams = currentParams.copy()
        slightlyDifferentParams[i] += EPSILON
        newError = errorFunc(slightlyDifferentParams, trainingSet)
        
        gradient[i] = newError - currentError    
    
    return gradient / EPSILON
        
#f = fMaker([0,0,0])
#print f([0,0])
#print f([0,1])
#print f([1,0])
#print f([1,1])

class GradientDescenter:
    def __init__(self, errorFunc, initParams):
        self.params = initParams
        self.errorFunc = errorFunc
        
    def train(self, trainingSet, scaleByImportance=False,  stepSize=0.01,            
        stopTime=0.005):
        
        numSteps = 0
        
        while True:
            gradient = findGradient(self.errorFunc, self.params, trainingSet)
            
            if scaleByImportance:
                importance = importanceDescent([i[0] for i in trainingSet], self.params)
                thingToAdd = np.multiply(gradient, importance)
                
            else:
                thingToAdd = gradient
            
#            print "grad:"
#            print gradient
#            print "norm imp:"
#            print np.linalg.norm(importance)/importance
#            print "imp:"
#            print importance
            
            self.params -= thingToAdd*stepSize
            
            if np.linalg.norm(gradient) < stopTime:
                break
            
            print self.params    
                
            numSteps += 1    
                
        print numSteps       
            
def importanceDescent(allInputs, currentParams):
    f = fMaker(currentParams)
    numParams = len(currentParams)
    global EPSILON
    
    
    runningSum = np.zeros(numParams)
    
    for inputs in allInputs:
        currentValue = f(inputs)
        
        for i in range(numParams):
            slightlyDifferentParams = currentParams.copy()
            slightlyDifferentParams[i] += EPSILON
            
            slightlyDifferentF = fMaker(slightlyDifferentParams)
            slightlyDifferentValue = slightlyDifferentF(inputs)
            
            dFdAi = (slightlyDifferentValue - currentValue) / EPSILON
        
            runningSum[i] += dFdAi**2
            
    for i, val in enumerate(runningSum):
        runningSum[i] = max(val, LESS_SERIOUS_EPSILON)        

    return (1./runningSum)/np.linalg.norm(1./runningSum)
    

trainingSet = [[[0,0], [1]],
                [[0,1], [1]],
                [[1,0], [1]],
                [[1,1], [0]]]
                
gd = GradientDescenter(errorFunc, np.array([0.5, 0.5, 0.5]))
gd.train(trainingSet, True, 0.01)



#print importanceDescent([i[0] for i in trainingSet], [0.4, 0.4, 0.3])