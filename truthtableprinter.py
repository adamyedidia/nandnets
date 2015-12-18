def allListsOfSizeX(x):
    if x == 0:
        return [[]]
        
    else:
        oneLess = allListsOfSizeX(x-1)
        return [i + [0] for i in oneLess] + [i + [1] for i in oneLess]  

def printTruthTable(func, numInputs, paramSet1, paramSet2=None, printWholeTable=True):
    inputs = allListsOfSizeX(numInputs)
    
    if printWholeTable:
        for inp in inputs:
            if paramSet2 == None:
                print inp, func(inp, paramSet1)
            else:
                print inp, func(inp, paramSet1), func(inp, paramSet2)
                
    else:
        tt = [func(inp, paramSet1) for inp in inputs]
        print tt, paramSet1, sum(i[0] for i in tt)
        pass
        
def printWholeParameterSpace(func, numInputs, numParams):
    paramSpace = allListsOfSizeX(numParams)
    
    for params in paramSpace:
        printTruthTable(func, numInputs, params, None, False)
        
