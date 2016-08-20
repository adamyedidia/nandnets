
import random
import matplotlib.pyplot as p

numBits = 4
pCorrect = 0.6

maxValue = 1000

UNARY_VALS = [maxValue/float(numBits)]*numBits

linearStepSize = (numBits*numBits + numBits)/2.
LINEAR_VALS = [maxValue/linearStepSize*i for i in range(1, numBits+1)]
#QUADRATIC_VALS = [maxValue/]

def findValsForGivenDeviationAmount(x):
    return [i+x*(-i+j) for i,j in zip(UNARY_VALS, LINEAR_VALS)]

def allListsOfSizeX(x):
    if x == 0:
        return [[]]
    
    else: 
        oneLess = allListsOfSizeX(x-1)
        return [[0.] + i for i in oneLess] + \
               [[1.] + i for i in oneLess]

def findBestBitString(vals, targetValue):
    bestDiff = float("Inf")
    bestBitString = None
    
    for l in allListsOfSizeX(numBits):
        ev = 0
        for v,b in zip(vals,l):
            ev += pCorrect*b*v + (1-pCorrect)*(1-b)*v
            
        diff = (ev - targetValue)**2
            
        if ev < bestDiff:
            bestDiff = diff
            bestBitString = l
                
    return bestBitString

def testDeviation(x, numIter=40000):
    print x
    vals = findValsForGivenDeviationAmount(x)
    
    mse = 0
    for _ in range(numIter):
        targetValue = random.randint(0, maxValue-1)
        bestBitString = findBestBitString(vals, targetValue)
        
#        print [(random.random()<pCorrect)*i*j for i,j in zip(bestBitString, 
#            vals)]
        
        noisyValue = sum([(random.random()<pCorrect)*i*j for i,j in zip(bestBitString, 
            vals)])
            
#        print noisyValue, targetValue    
            
        diff = noisyValue - targetValue
        mse += diff*diff
        
#        print mse
        
    mse /= float(numIter)

    return mse**0.5

xAxis = [i/10. for i in range(30)]

p.plot(xAxis, [testDeviation(i) for i in xAxis])
p.savefig("testdev3.png")


#print LINEAR_VALS
#print sum(LINEAR_VALS)



