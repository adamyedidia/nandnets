from funcs import *
from truthtableprinter import *
import matplotlib.pyplot as p

memoryMachineCompression = False
xorCompressionCW = False
xorCompression = True
xorAgreementCW = False
xorAgreement = False
softRandomAgreement = False
randomAgreementTest = False
softSamplingAgreement = False
samplingAgreement = False
agreementSquareTest = False
simpleAgreement = False
hybridRecoveryTestCW = False
hybridRecoveryTestSample = False
randomNandRecoveryTestCW = False        
randomNandRecoveryTestSample = False        
bigNandTest = False
tinyGateTest = False                 
tinyGateTestSoft = False                               
tinyGateTestCW = False
trivialGateTest = False   
trivialGateTestCW = False    
compressionTest = False    
compressionTestCW = False

if memoryMachineCompression:
    
    for patternSize in range(40):
        
        print "patternSize:", patternSize
            
        trainingSetSize = 1000
        testSetSize = 1000
    
        m = MemoryMachine()
    
        compressionData = getCompressionData("../../compression/declarationbits.txt",
            patternSize)
    
        trainingSet = compressionData[:trainingSetSize]
        testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]

    #    trainingSet = generateRandomTruthTable(n, 1)
        
        m.train(trainingSet)
        p.plot(patternSize, m.test(testSet, True, False), "bo")
        
    p.savefig("mmcompression.png")    

if xorCompressionCW:
    patternSize = 15
    numParams = 359
    
    p = 0.5
    
    trainingSetSize = 1000
    testSetSize = 1000
    
    threshold = numParams/2+1*(numParams%2)
    
    randomXORAgreement = randomXORAgreementMaker(patternSize, numParams, threshold)
        
    c = CubeWalker(randomXORAgreement, [1.0*(random.random()<p) for _ in range(numParams)])

    compressionData = getCompressionData("../../compression/declarationbits.txt",
        patternSize)
        
    trainingSet = compressionData[:trainingSetSize]
    testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]

#    trainingSet = generateRandomTruthTable(n, 1)
            
    c.train(trainingSet, False)
    c.test(testSet, True, False)

if xorCompression:
    patternSize = 10
    numParams = 201
    
    p = 0.5
    
    trainingSetSize = 1000
    testSetSize = 1000
    
    threshold = numParams/2+1*(numParams%2)
    
    randomXORAgreement = randomXORAgreementMaker(patternSize, numParams, threshold)
    
    s = SoftSampler(randomXORAgreement, [p]*numParams)
    
    compressionData = getCompressionData("../../compression/declarationbits.txt",
        patternSize)
        
    trainingSet = compressionData[:trainingSetSize]
    testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]
        
    s.train(trainingSet, 100, 100, True)
    s.test(testSet, True, False)
  
if xorAgreementCW:
    n = 5
    p = 0.5
    numParams = 31
    
    threshold = numParams/2+1*(numParams%2)
        
    randomXORAgreement = randomXORAgreementMaker(n, numParams, threshold)
        
    c = CubeWalker(randomXORAgreement, [1.0*(random.random()<p) for _ in range(numParams)])
    
    trueFunc, trueParams = randomInstanceGen(randomXORAgreement, numParams, p)
    trainingSet = generateFullTruthTable(trueFunc, n)

#    trainingSet = generateRandomTruthTable(n, 1)    
    
#    print trueParams
    
    testSet = trainingSet
    
    c.train(trainingSet, True)
    c.test(testSet, False, False)
  
if xorAgreement:
    n = 5
    p = 0.5
    numParams = 31
        
    threshold = numParams/2+1*(numParams%2)
    
    randomXORAgreement = randomXORAgreementMaker(n, numParams, threshold)
    
    s = SoftSampler(randomXORAgreement, [p]*numParams)
    
#    trueFunc, trueParams = randomInstanceGen(randomXORAgreement, numParams, p)
#    trainingSet = generateFullTruthTable(trueFunc, n)
    
    trainingSet = generateRandomTruthTable(n, 1)    
    
    testSet = trainingSet
    
#    print trueParams
    
    s.train(trainingSet, 1000, 100, True)
    s.test(testSet, False, False)
  
if softRandomAgreement:
    n = 7
    p = 0.5
    numParams = 7
    
    threshold = numParams/2+1*(numParams%2)
        
    randomAgreement = randomAgreementMaker(n, numParams, threshold)
        
    s = SoftSampler(randomAgreement, [p]*numParams)    

    trueFunc, trueParams = randomInstanceGen(randomAgreement, numParams, p)
    trainingSet = generateFullTruthTable(trueFunc, n)
    
    testSet = trainingSet
    
    s.train(trainingSet, 1, 20, True)
    s.test(testSet, False, False)        
        
if randomAgreementTest:
    n = 9        
    p = 0.5
    numParams = 29
    
    threshold = numParams/2+1*(numParams%2)
        
    randomAgreement = randomAgreementMaker(n, numParams, threshold)
        
    c = CubeWalker(randomAgreement, [1.0*(random.random()<p) for _ in range(numParams)])
    
    trueFunc, trueParams = randomInstanceGen(randomAgreement, numParams, p)
    trainingSet = generateFullTruthTable(trueFunc, n)

#    trainingSet = generateRandomTruthTable(n, 1)    
    
#    print trueParams
    
    testSet = trainingSet
    
    c.train(trainingSet, True)
    c.test(testSet, False, False)
        
if softSamplingAgreement:
    logN = 3
    n = 2 ** logN
    p = 0.5
        
    orthoSet = makeOrthogonalSet(logN)    
        
    agreementSquare = agreementSquareMaker(n, n/2+1*(n%2), orthoSet)

#    printWholeParameterSpace(agreementSquare, n, 2*n)
    
    agreement = agreementMaker(n, n/2+1*(n%2))
    
#    printWholeParameterSpace(agreement, n, n)
        
    s = Sampler(agreementSquare, [p]*2*n)
    
    trueFunc, trueParams = randomInstanceGen(agreementSquare, 2*n, p)
    trainingSet = generateFullTruthTable(trueFunc, n)
    
    testSet = trainingSet
    
    s.train(trainingSet, 10, 100, False)
    s.test(testSet, False, False)        
        
if samplingAgreement:
    logN = 3
    n = 2 ** logN
    p = 0.5
    
    agreement = agreementMaker(n, n/2+1*(n%2))   
    
    s = Sampler(agreement, [p]*n)     
 
    trueFunc, trueParams = randomInstanceGen(agreement, n, p)
    trainingSet = generateFullTruthTable(trueFunc, n)
    
#    trainingSet = generateRandomTruthTable(n, 1)
    
    testSet = trainingSet
    
    s.train(trainingSet, 5, 2000, True)
    s.test(testSet, False, False)
        
if simpleAgreement:
    n = 13
    
    p = 0.5
    
    agreement = agreementMaker(n, n/2+1*(n%2))
    
    c = CubeWalker(agreement, [1.0*(random.random()<p) for _ in range(n)])

    trueFunc, trueParams = randomInstanceGen(agreement, n, p)
    trainingSet = generateFullTruthTable(trueFunc, n)

#    trainingSet = generateRandomTruthTable(n, 1)
        
    testSet = trainingSet
    
    c.train(trainingSet, False)
    c.test(testSet, False, False)

if agreementSquareTest:
    logN = 2
    n = 2 ** logN
    p = 0.5
    
    agreementSquare = agreementSquareMaker(n, n/2+1*(n%2), makeOrthogonalSet(logN))
    
    printWholeParameterSpace(agreementSquare, n, 2*n)
    
#    agreement = agreementMaker(n, n/2+1*(n%2))
    
    print ""
    
#    printWholeParameterSpace(agreement, n, n)
    
    c1 = CubeWalker(agreementSquare, [1.0*(random.random()<p) for _ in range(n*2)])
    
    trueFunc, trueParams = randomInstanceGen(agreementSquare, 2*n, p)
    trainingSet = generateFullTruthTable(trueFunc, n)

#    trainingSet = generateRandomTruthTable(n, 1)
    
    testSet = trainingSet
    
    c1.train(trainingSet, True, trueParams)
    c1.test(testSet, False, False)
    
#    c2 = CubeWalker(agreement, [1.0*(random.random()<p) for _ in range(n)])
    
#    c2.train(trainingSet, False)
#    c2.test(testSet, False, False)
    
if hybridRecoveryTestCW:
    n = 5
    internals = 3
    outputs = 1
    
    trueP = 0.2
    p = 0.5
    
    hybridFunc = hybridFunctionMaker(n, internals, outputs)
    nandDag = nandDagFunctionMaker(n, internals, outputs)
    numParams = 2*numParamCalculator(n, internals, outputs)
    
    c1 = CubeWalker(hybridFunc, [1.0*(random.random()<p) for _ in range(numParams)])
    
#    trueFunc = randomInstanceGen(hybridFunc, numParams, trueP)
    trueFunc = randomInstanceGen(nandDag, numParams/2, trueP)
    trainingSet = generateFullTruthTable(trueFunc, n)
    
#    print trueFunc([0,1,1,0,1])
    
#    print trainingSet
    
    testSet = trainingSet
    c1.train(trainingSet, True)
    c1.test(testSet, False, False)        
              
    hybridFuncClever = hybridFunctionMakerClever(n, internals, outputs)

    numParams = 2*numParamCalculator(n, internals, outputs)
    
    c2 = CubeWalker(hybridFuncClever, [1.0*(random.random()<p) for _ in range(numParams)])     
    
    c2.train(trainingSet, False)
    c2.test(testSet, False, False)                     
              
        
if hybridRecoveryTestSample:
    n = 5
    internals = 3
    outputs = 1
    
    hybridFunc = hybridFunctionMaker(n, internals, outputs)
    numParams = 2*numParamCalculator(n, internals, outputs)
    
    s = Sampler(hybridFunc, [0.5]*numParams)
    
    trueFunc = randomInstanceGen(hybridFunc, numParams, 0.5)
    trainingSet = generateFullTruthTable(trueFunc, n)
    
    testSet = trainingSet
    s.train(trainingSet, 5, 2000, True)
    s.test(testSet, True, True)        
        
if randomNandRecoveryTestCW:        
    n = 5
    internals = 3
    outputs = 1
    
    nandDag = nandDagFunctionMaker(n, internals, outputs)
    numParams = numParamCalculator(n, internals, outputs)
    
    s = CubeWalker(nandDag, [1.0*(random.random()>0.5) for _ in range(numParams)])
    
    trueFunc = randomInstanceGen(nandDag, numParams)
    trainingSet = generateFullTruthTable(trueFunc, n)
    
    testSet = trainingSet
    
    s.train(trainingSet, True)
    s.test(testSet, True, True)        
                
if randomNandRecoveryTestSample:        
    n = 5
    internals = 3
    outputs = 1
    
    nandDag = nandDagFunctionMaker(n, internals, outputs)
    numParams = numParamCalculator(n, internals, outputs)
    
    s = Sampler(nandDag, [0.5]*numParams)
    
    trueFunc = randomInstanceGen(nandDag, numParams)
    trainingSet = generateFullTruthTable(trueFunc, n)
    
    testSet = trainingSet
    
    s.train(trainingSet, 5, 2000, True)
    s.test(testSet, True, True)
            
if bigNandTest:
    
    n = 2
    internals = 3
    
#    trainingSet = generateRandomTruthTable(n, 1)
    trainingSet = [[[0,0], [0]],            
                    [[0,1], [1]],
                    [[1,0], [1]],
                    [[1,1], [0]]]

    for i in trainingSet:
        print i[0], i[1]
        
    print ""
    
    nandDag = nandDagFunctionMaker(n, internals, 1)
    numParams = numParamCalculator(n, internals, 1)    
    
    c = CubeAnalyzer(nandDag, numParams)
#    c = CubeAnalyzer(bigAnd, n)
    
    c.findScores(trainingSet)
    print c.countLocalMaxima()
                
if tinyGateTest:            
            
    s = Sampler(tinyNandNorImproved, [0.4]*3)

    trainingSet = [[[0,0], [1]],
                    [[0,1], [1]],
                    [[1,0], [1]],
                    [[1,1], [0]]]
                
    random.shuffle(trainingSet)
                
    testSet = trainingSet
    
    s.train(trainingSet, 8, 10000)
    s.test(testSet, True, True)
    
if tinyGateTestSoft:            
            
    s = SoftSampler(tinyNandNor, [0.4]*3)

    trainingSet = [[[0,0], [1]],
                    [[0,1], [1]],
                    [[1,0], [1]],
                    [[1,1], [0]]]
                
    random.shuffle(trainingSet)
                
    testSet = trainingSet
    
    s.train(trainingSet, 100, 100)
    s.test(testSet, True, True)    
    
if tinyGateTestCW:            
            
    c = CubeWalker(tinyNandNorImproved, [random.randint(0, 1) for _ in range(3)])

    trainingSet = [[[0,0], [1]],
                    [[0,1], [1]],
                    [[1,0], [1]],
                    [[1,1], [0]]]
                
    random.shuffle(trainingSet)
                
    testSet = trainingSet
    
    c.train(trainingSet, True)
    c.test(testSet, True, True)
    
if trivialGateTest:
        
    nandDag = nandDagFunctionMaker(2, 0, 1)
    
    s = Sampler(nandDag, [0.2]*4)
        
    trainingSet = [[[0,0], [1]],
                    [[0,1], [1]],
                    [[1,0], [1]],
                    [[1,1], [0]]]
                
    random.shuffle(trainingSet)
                
    testSet = trainingSet
    
    s.train(trainingSet, 10, 10000)
    s.test(testSet, True, True)    
        
if trivialGateTestCW:
    
    nandDag = nandDagFunctionMaker(2, 0, 1)
    
    c = CubeWalker(nandDag, [random.randint(0, 1) for _ in range(2)])
        
    trainingSet = [[[0,0], [1]],
                    [[0,1], [1]],
                    [[1,0], [1]],
                    [[1,1], [0]]]
                
    random.shuffle(trainingSet)
                
    testSet = trainingSet
    
    c.train(trainingSet, True)
    c.test(testSet, True, True)            
        
if compressionTest:
    
    patternSize = 20
    numInternals = 50
    
    trainingSetSize = 1000
    testSetSize = 1000
    
    nandDag = nandDagFunctionMaker(patternSize, numInternals, 1)
    numParams = numParamCalculator(patternSize, numInternals, 1)
    
    s = Sampler(nandDag, [0.1]*numParams)
    
    compressionData = getCompressionData("../../compression/declarationbits.txt",
        patternSize)
        
    trainingSet = compressionData[:trainingSetSize]
    testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]
        
    s.train(trainingSet, 1, 500)
    s.test(testSet, True, True)   
    
if compressionTestCW:
    
    patternSize = 15
    numInternals = 15
    
    trainingSetSize = 1000
    testSetSize = 1000
    
    nandDag = nandDagFunctionMaker(patternSize, numInternals, 1)
    numParams = numParamCalculator(patternSize, numInternals, 1)
        
    c = CubeWalker(nandDag, [1.0 * (random.random() < 0.1) for _ in range(numParams)])
    
    compressionData = getCompressionData("../../compression/declarationbits.txt",
        patternSize)
        
    trainingSet = compressionData[:trainingSetSize]
    testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]
        
    c.train(trainingSet, False)
    c.test(testSet, True, True)  