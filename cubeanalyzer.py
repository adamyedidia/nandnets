def pront(x):
    print x
    
def allTuplesOfSizeX(x):
    if x == 0:
        return [()]
        
    else:
        oneLess = allTuplesOfSizeX(x-1)
        return [i + tuple([0]) for i in oneLess] + [i + tuple([1]) for i in oneLess]   
    
class CubeAnalyzer:
    def __init__(self, f, numParams):
        self.f = f
        self.numParams = numParams
        self.scoreDict = None
        
    def evaluate(self, inputs, params=None):
        if params == None:
            params = self.params
            
        return self.f(inputs, params)        
        
    def testParamAssignmentOnInput(self, inputs, outputs, outputIndex, paramAssignment):
        result = self.evaluate(inputs, paramAssignment)
        
        if outputs[outputIndex] == result[outputIndex]:
            return 1
            
        return 0
        
    def generateDeviationsFromParams(self, paramAssignment):
        listOfDeviateParams = []
        
        for i, val in enumerate(paramAssignment):
            deviateParams = paramAssignment[:]
            deviateParams[i] = 1-val
            
            listOfDeviateParams.append(deviateParams)
            
        return listOfDeviateParams
        
    def scoreParams(self, trainingSet, paramAssignment):
        numOutputs = len(trainingSet[0][1])

        score = 0

        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            outputs = dataPoint[1]
            
            for outputIndex in range(numOutputs):
                score += self.testParamAssignmentOnInput(inputs, outputs, outputIndex, paramAssignment)
                
        return score        
        
    def findScores(self, trainingSet):
        listOfPossibleParams = allTuplesOfSizeX(self.numParams)
        
        self.scoreDict = {}
        bestScore = -1
        
        for paramAssignment in listOfPossibleParams:
            score = self.scoreParams(trainingSet, paramAssignment)
            self.scoreDict[paramAssignment] = score
            
            bestScore = max(bestScore, score)
            
        return bestScore
        
    def isLocalMaximum(self, paramAssignment):
        listOfDeviateParams = self.generateDeviationsFromParams(list(paramAssignment))
        
        localScore = self.scoreDict[paramAssignment]
        
        localMaxness = "STRICT"
        
        for deviateParam in listOfDeviateParams:
            deviateTuple = tuple(deviateParam)
            deviateScore = self.scoreDict[deviateTuple]
            
            if deviateScore > localScore:
                localMaxness = "NOT"
                
            elif deviateScore == localScore:
                if not localMaxness == "NOT":
                    localMaxness = "LOOSE"
                    
        return localMaxness
        
    def countLocalMaxima(self):
        listOfPossibleParams = allTuplesOfSizeX(self.numParams)
        
        numStrictLocalMaxima = 0
        numLooseLocalMaxima = 0
        
        for paramAssignment in listOfPossibleParams:
            localMaxness = self.isLocalMaximum(paramAssignment)
            
            print paramAssignment, localMaxness, self.scoreDict[paramAssignment]
            
            if localMaxness == "STRICT":
                numStrictLocalMaxima += 1
                numLooseLocalMaxima += 1
            
            elif localMaxness == "LOOSE":
                numLooseLocalMaxima += 1
        
        return numStrictLocalMaxima, numLooseLocalMaxima
        
                
            
        