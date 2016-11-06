import random

# ^ is the bitwise xor operation in Python
def hammingDistanceInt(int1, int2):
    return bin(int1 ^ int2).count("1")
    
def bitwiseXorInt(int1, int2):
    return int1 ^ int2        

def hammingDistance(list1, list2):
    return sum([i ^ j for i,j in zip(list1, list2)])

def bitwiseXor(list1, list2):
    return [i ^ j for i,j in zip(list1, list2)]

class ParameterMachineInt:
    # P is number of params
    # f is the function's name
    def __init__(self, f, P):
        self.f = f
        self.P = P
        self.listOfLinks = [1 << i for i in range(P)]
        
    def evaluate(self, inputs, params=None):
        if params == None:
            params = self.params

        return self.f(inputs, params)
    
    def rearrange(self, listOfInputs, params=None):
        if params == None:
            params = self.params
            
        intOfOutputsHere = self.generateIntOfOutputs(listOfInputs, params)
        
        listOfLinkedIntsOfOutputs = [self.generateIntOfOutputs(listOfInputs, params ^ link) for link in self.listOfLinks]
        
        listOfJumpsSorted = sorted([intOfOutputsHere^intOfOutputsThere for \
            intOfOutputsThere in listOfLinkedIntsOfOutputs], key=lambda x : bin(x).count("1"))
            
        print listOfJumpsSorted        
    
    def generateIntOfOutputs(self, listOfInputs, params=None):
        intOfOutputs = 0
        for i, inputs in enumerate(listOfInputs):
            intOfOutputs += self.evaluate(inputs, params) << i
            
        return intOfOutputs

class ParameterMachine:
    # P is number of params
    # f is the function's name
    def __init__(self, f, P):
        self.f = f
        self.P = P
        self.listOfLinks = [[1*(i==j) for j in range(P)] for i in range(P)]

    def rearrange(self, params, listOfInputs):
        pass

    def evaluate(self, inputs, params=None):
        if params == None:
            params = self.params

        return self.f(inputs, params)
    
    def generateListOfOutputs(self, listOfInputs, params=None):
        listOfOutputs = []
        for inputs in listOfInputs:
            listOfOutputs.append(self.evaluate(inputs, params))
            
        return listOfOutputs
        
def tinyNandNorInt(inputs, params):
    if params > 3:
        # nand
        return 1*((inputs % 4) != 3)     
    
    else:
        # nor
        return 1*((inputs % 4) != 0)      
        
P = 3        
I = 2
pmi = ParameterMachineInt(tinyNandNorInt, P)
print pmi.listOfLinks

randomParams = random.randint(0,P-1)
fullListOfInputs = range(1<<P)

print randomParams

print pmi.generateIntOfOutputs(fullListOfInputs, randomParams)
#pmi.rearrange(range(1 << P), random.randint(0,P-1))
