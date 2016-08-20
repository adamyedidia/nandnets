import matplotlib.pyplot as p

import random

numRegisters = 15
numValues = 400

UNARY_VALS = [1]*numRegisters
LINEAR_VALS = range(1, numRegisters+1)

QUADRATIC_VALS = [1]
for i in range(numRegisters-1):
    QUADRATIC_VALS += [QUADRATIC_VALS[-1] + LINEAR_VALS[i]]

CUBIC_VALS = [1]
for i in range(numRegisters-1):
    CUBIC_VALS += [CUBIC_VALS[-1] + QUADRATIC_VALS[i]]

BINARY_VALS = [2**x for x in range(numRegisters)]

def dot(x, y):
    return sum([i*j for i, j in zip(x, y)])

def allListsOfSizeX(x):
    if x == 0:
        return [[]]
    
    else: 
        oneLess = allListsOfSizeX(x-1)
        return [[0.] + i for i in oneLess] + \
               [[1.] + i for i in oneLess]

valsToTry = LINEAR_VALS
print valsToTry

valueDict = {}

for i in range(numValues):
    
    valueDict[i] = [{}, 0]
    
    for j in range(numRegisters):
        valueDict[i][0][j] = 0.
    

for number in allListsOfSizeX(numRegisters):
#    print number
    valueOfNumber = dot(number, valsToTry)
    
    if valueOfNumber in valueDict:
        for j, bit in enumerate(number):
            valueDict[valueOfNumber][0][j] += bit
            
        valueDict[valueOfNumber][1] += 1.
           
#    print valueDict            
                
                
for j in range(numRegisters):
#    if j == 4:
    if True:
        p.plot([valueDict[i][0][j]/valueDict[i][1] for i in range(numValues)])
    
p.show()


numDigits = 5 

def computeValue(x):
    if x == []:
        return 0
    return 2*computeValue(x[:-1]) + x[-1]

binaryNumber = [random.randint(0,1) for _ in range(5)]

print "True value:", computeValue(binaryNumber)

errorProb = 0.25

erroneousNumber = "s"

for bit in binaryNumber:
    pass