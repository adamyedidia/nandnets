import sys

sys.path.insert(0, "~/neural-networks-and-deep-learning/src")

import mnist_loader

def strSum(l):
    returnString = ""
    for string in l:
        returnString += string
    
    return returnString

class Clause:
    def __init__(self, literalList):
        # a list of tuples (object, boolean)
        self.literalList = literalList
        
    def __str__(self):
        returnString = ""
        
        for literal in self.literalList:
            obj = literal[0]
            boolean = literal[1]
            
            if not boolean:
                returnString += "~"
            
            returnString += str(obj) + " "    
    
        return returnString + "\n"

class Value:
    def __init__(self, exampleID, xIndex):
        self.exampleID = exampleID
        self.xIndex = xIndex
                
    def __str__(self):
        return "x" + str(self.exampleID) + "," + str(self.xIndex) + ",-"
        
class Weight:
    def __init__(self, xIndex, yIndex):
        assert xIndex > yIndex
        
        self.xIndex = xIndex
        self.yIndex = yIndex
        
    def __str__(self):
        return "w-," + str(self.xIndex) + "," + str(self.yIndex)
        
class ActiveZero:
    def __init__(self, exampleID, xIndex, yIndex, valDict, weightDict):
        val = valDict[exampleID][yIndex]
        weight = weightDict[xIndex][yIndex]
        
        self.value = val
        self.weight = weight            
        
        self.exampleID = exampleID
        self.xIndex = xIndex
        self.yIndex = yIndex
        
    def __str__(self):
        return "a" + str(self.exampleID) + "," + str(self.xIndex) + "," + str(self.yIndex)
        
    def getClauses(self):
        # x v ~w v a
        # ~a v ~x
        # ~a v w
        
        return [Clause([(self.value, True), (self.weight, False), (self, True)]),
                Clause([(self, False), (self.value, False)]),
                Clause([(self, False), (self.weight, True)])]
        
class Nand:
    def __init__(self, exampleID, xIndex, valDict, activeDict):
        self.inputList = []
        for yIndex in range(xIndex):
            self.inputList.append(activeDict[exampleID][xIndex][yIndex])
            
        self.output = valDict[exampleID][xIndex]
        
    def getClauses(self):
        # a1 v a2 v ~out
        # ~a1 v out
        # ~a2 v out
        
        listOfClauses = []
        firstClauseLiteralList = [(self.output, False)]
        
        for inp in self.inputList:
            listOfClauses.append(Clause([(inp, False), (self.output, True)]))
            firstClauseLiteralList.append((inp, True))
            
        listOfClauses.append(Clause(firstClauseLiteralList))
        
        return listOfClauses
        
class Input:
    def __init__(self, exampleID, xIndex, boolean, valDict):
        self.value = valDict[exampleID][xIndex]
        self.boolean = boolean
        
    def getClauses(self):
        return [Clause([(self.value, self.boolean)])]    
        
class Output:
    def __init__(self, exampleID, xIndex, boolean, valDict):
        self.value = valDict[exampleID][xIndex]
        self.boolean = boolean
        
    def getClauses(self):
        return [Clause([(self.value, self.boolean)])]        
        
def makeSatInstance(data, numValues):    
    numDataPoints = len(data)
    
    assert numDataPoints > 0
    
    firstDataPoint = data[0]
    
    numInputs = len(data[0][0])
    numOutputs = len(data[0][1])
    
    valDict = {}
    
    # Make values
    for exampleID in range(numDataPoints):
        valDict[exampleID] = {}
        
        for xIndex in range(numValues):
            valDict[exampleID][xIndex] = Value(exampleID, xIndex)
            
    weightDict = {}
    
    # Make weights
    for xIndex in range(numInputs, numValues):
        weightDict[xIndex] = {}
        
        for yIndex in range(xIndex):
            weightDict[xIndex][yIndex] = Weight(xIndex, yIndex)
            
    activeDict = {}
    listOfClauses = []
    
    # Make activeZeroes
    for exampleID in range(numDataPoints):
        activeDict[exampleID] = {}   
        
        for xIndex in range(numInputs, numValues):
            activeDict[exampleID][xIndex] = {}
            
            for yIndex in range(xIndex):
                activeZero = ActiveZero(exampleID, xIndex, yIndex, valDict, weightDict)
                
                activeDict[exampleID][xIndex][yIndex] = activeZero
                listOfClauses += activeZero.getClauses()
                
    # Make NANDs
    for exampleID in range(numDataPoints):
        for xIndex in range(numInputs, numValues):
            listOfClauses += Nand(exampleID, xIndex, valDict, activeDict).getClauses()
            
    # Make inputs and outputs
    for exampleID in range(numDataPoints):
        inputs = data[exampleID][0]
        outputs = data[exampleID][1]
        
        assert len(inputs) == numInputs
        assert len(outputs) == numOutputs
        
        for xIndex in range(numInputs):
            if inputs[xIndex] == 1 or inputs[xIndex] == True:
                boolean = True
            elif inputs[xIndex] == 0 or inputs[xIndex] == False:
                boolean = False
            else:
                raise Exception("Unacceptable (non-boolean) input " + str(inputs[xIndex]))
                
            listOfClauses += Input(exampleID, xIndex, boolean, valDict).getClauses()
            
        for xIndex in range(numValues - numOutputs, numValues):
            outputIndex = xIndex - numValues + numOutputs
            
            if outputs[outputIndex] == 1 or outputs[outputIndex] == True:
                boolean = True
            elif outputs[outputIndex] == 0 or outputs[outputIndex] == False:
                boolean = False
            else:
                raise Exception("Unacceptable (non-boolean) output " + str(outputs[xIndex]))
            
            listOfClauses += Output(exampleID, xIndex, boolean, valDict).getClauses()
            
    return strSum([str(clause) for clause in listOfClauses])
    
output = open("nand.cnf", "w")

#data = [([0,0], [0]), ([0,1], [1]), ([1,0], [1]), ([1,1], [0])]
#output.write(makeSatInstance(data, 6))

def processRound(l):
    for i in range(len(l)):
        l[i] = [[round(x) for x in l[i][0]], l[i][1]]
    
    return l

data = processRound(mnist_loader.load_data_wrapper()[0][:1])
print len(data)
print "data loaded"
output.write(makeSatInstance(data, 840))
#data = [([0,0], [1]), ([0,1], [0]), ([1,0], [0]), ([1,1], [0])]
#output.write(makeSatInstance(data, 3))