import random

# Makes an array of None's that is the same dimension as the
# input array
def noneDeepCopy(l):
    returnList = []
    
    for i in l:
        returnList.append([])
        for j in i:
            returnList[-1].append(None)
            
    return returnList

# Transforms a 2D array into its 1D version. 
def flattenWithZeroes(l): 
    flattenedArray = []
    for miniL in l:
        flattenedArray += miniL
    
    return flattenedArray
 
def orProb(x, y):
     return x + y - x*y

class DAG:
    def __init__(self, weightArray):
        self.wArray = weightArray # 2D array 
        self.gArray = noneDeepCopy(weightArray)
        self.condXArray = noneDeepCopy(weightArray)
        
        self.xArray = [None]*len(weightArray)
        self.flawlessXArray = [None]*len(weightArray)
        
    def setInputs(self, inputs):
        for i, inp in enumerate(inputs):
            self.xArray[i] = inp
            
    def extractOutputs(self, numOutputs):
        return self.xArray[-numOutputs:]    
        
    def determineFeedForward(self, inputs):
        dag.setInputs(inputs)
        
        for i in range(len(self.xArray)):
            
            # Later want to change to "if i >= numInputs"    
            if i >= len(inputs): # Don't want to reset the input guys           
            
                noActive0Product = 1.0
            
                for j in range(i):
                    x = self.xArray[j]
                    w = 1.0 * (self.wArray[i][j] > random.random())
                
                    noActive0Product *= orProb(x, 1-w)
                
                self.xArray[i] = 1.0 - noActive0Product
                
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
                
    def feedForward(self, inputs):
        dag.setInputs(inputs)
        
        for i in range(len(self.xArray)):
            
            if i >= len(inputs):
            
                xProduct = 1.0
            
                for j in range(i):
                
                    xProduct *= orProb(self.xArray[j], 1-self.wArray[i][j])
                
                self.xArray[i] = 1.0 - xProduct        
                
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
            
def allSizeXLists(x):
    if x == 0:
        return [[]]
        
    allSizeXMinus1Lists = allSizeXLists(x - 1)
    
    return [littleList + [0.0] for littleList in allSizeXMinus1Lists] + \
        [littleList + [1.0] for littleList in allSizeXMinus1Lists]


        
weightArray = [[], [0.5], [0.5, 0.5], [0.5, 0.5, 0.5]]                
dag = DAG(weightArray)

inp = [0.0]

dag.accurateFeedForward(inp)
#print dag.extractOutputs(1)
print dag.xArray

dag.feedForward(inp)
#print dag.extractOutputs(1)      
print dag.xArray          
                
dag.stochasticFeedForward(inp, 100000)
#print dag.extractOutputs(1)     
print dag.xArray           

dag.flawlessFeedForward(inp)
print dag.xArray                
				
print 45./64                
#            else:
#                self.condXArray[i] = self.xArray[i]