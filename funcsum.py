def argmax(l):
    bestIndex = -1
    bestVal = -float("Inf")
        
    for i, val in enumerate(l):
        if val > bestVal:
            bestIndex = i
            bestVal = val
            
    return bestIndex

# Takes as input a 784-entry array and displays it as a handwritten digit    
def displayDigit(l):
    indexCounter = 0
    for _ in range(28):
        printString = ""
        for _ in range(28):
            printString += str(int(round(l[indexCounter])))
            indexCounter += 1
        print printString


class FuncSum:
    def __init__(self, listOfFuncs):
        self.listOfFuncs = listOfFuncs
        
    def train(self, trainingSet):
        self.weights = [0.] * len(self.listOfFuncs)
        normalizations = [0.] * len(self.listOfFuncs)
        
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1][0]
            
            functionResults = [f(inputs) for f in self.listOfFuncs]
            functionResultsTimesOutputs = [x*output for x in functionResults]
            functionResultsTimesThemselves = [x*x for x in functionResults]
            
            self.weights = [w + i for w, i in \
                zip(self.weights, functionResultsTimesOutputs)]
                
            normalizations = [n + i for n, i in \
                zip(normalizations, functionResultsTimesThemselves)]
                
            # Avoid division by 0; if a func never returns anyting nonzero
            # across the whole training set, its weight is 0    
            normalizations = [n*(n!=0)+1*(n==0) for n in normalizations]    
                                
        self.weights = [w / float(n) for w, n in \
            zip(self.weights, normalizations)]        
        
        print "training complete"
        
    def evaluate(self, inputs):
        functionResults = [f(inputs) for f in self.listOfFuncs]
                
        return sum([w*i for w, i in zip(self.weights, functionResults)])
        
    def test(self, testSet, verbose=False):
        squaredError = 0.0
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            output = dataPoint[1]
            
            result = self.evaluate(inputs)
            if verbose:
                print result, output[0]
            
            squaredError += (result - output[0])**2
            
        mse = float(squaredError) / len(testSet)     
            
        print "MSE:", mse
        
        return mse
        
# For figuring out which of the oneness, twoness, threeness, etc is higher        
class FuncSumClique:        
    def __init__(self, listOfFuncs, numOptions, convertToIndex):    
        self.listOfFuncSums = [FuncSum(listOfFuncs) for i in range(numOptions)]
        self.numOptions = numOptions
        self.convertToIndex = convertToIndex
    
    def getListOfTrainingSets(self, trainingSet):
        listOfTrainingSets = []
        
        for _ in range(self.numOptions):
            listOfTrainingSets.append([])
            
        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            output = dataPoint[1]
            
            for i in range(self.numOptions):
                if i == self.convertToIndex(output):
                    listOfTrainingSets[i].append([inputs, [1.0]])
                else:
                    listOfTrainingSets[i].append([inputs, [0.0]])
        
        return listOfTrainingSets
        
    def train(self, trainingSet):
        listOfTrainingSets = self.getListOfTrainingSets(trainingSet)
        
        [fs.train(ts) for fs, ts in zip(self.listOfFuncSums, listOfTrainingSets)]
        
    def test(self, testSet, verbose=False):
        numCorrect = 0.
        numTotal = 0.
        
        for dataPoint in testSet:
            inputs = dataPoint[0]
            output = dataPoint[1]
            
            results = [fs.evaluate(inputs) for fs in self.listOfFuncSums]
            
            bestResultIndex = argmax(results)
            
            if self.convertToIndex(output) == bestResultIndex:
                numCorrect += 1.
                numTotal += 1.
                
            else:
                numTotal += 1.
                
                if verbose:
                    print "Image"
                    displayDigit(inputs)
                    
                    print "Program's guess:", bestResultIndex
                    
                
        print "Got", numCorrect, "out of", numTotal, "correct."
        
        return numCorrect/numTotal