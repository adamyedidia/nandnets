from funcs import *
from truthtableprinter import *
import matplotlib.pyplot as p
from quicksampler import QuickSampler
import random

gateTest = True
quickSamplerTest = False
pmAssemblerTest = False
mfscTest = False
mfsTest = False
linearFuncSumDigits = False
linearFitDigits = False
funcSumTest = False
majoritarianHardStudy = False
majoritarianSelfLearn = False
majoritarianCompression = False
majoritarianAllFuncs = False
majoritarianAllXOR = False
memoryMachineCompression = False
xorCompressionCW = False
xorCompression = False
majoritarianxorCompression = False
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
randomNandRecoveryTestCWImportance = False
randomNandRecoveryTestCW = False
randomNandRecoveryTestSample = False
bigNandTest = False
tinyGateTest = False
tinyGateTestSoft = False
tinyGateTestCW = False
trivialGateTest = False
trivialGateTestCW = False
compressionTest = False
compressionTestCWImportanceWeighted = False
compressionTestCW = False

if gateTest:
    maj3 = gateFuncMaker(3, [[[1,1,0,0]], [[1,0,0],[1,1,0]], [[0,0],[1,0],[1,1]], [[0],[0],[1],[1]]])

#    print maj3([0,0,0], [1,0,1])

    nand3 = gateFuncMaker(3, [[[0,0,0,0]], [[1,1,1],[0,0,0]], [[1,1],[1,1],[0,0]], [[1],[1],[1],[0]]])

    print nand3([1,1,1], [0,0,0])

if quickSamplerTest:
    lookback = 14

    rangeOfSearch = 5000

    text = open("/Users/adamyedidia/compression/declarationbits.txt").read()

    def getSample():
        start = random.randint(0, rangeOfSearch-1)
        end = start + 300

        return [[convertToOneMinusOne(int(i)) for i in text[start:end]],
                convertToOneMinusOne(text[end])]

    listOfFuncs = []

    for i in range(lookback):
        listOfFuncs.append(justThatInputMaker(i))

    qs = QuickSampler(getSample, getSample, listOfFuncs)

    qs.test(1000)


if pmAssemblerTest:
    n = 10
    trainingSet = generateRandomTruthTable(n, 1)

#    trainingSet = [[[0.,0.], [1.]],
#                   [[0.,1.], [0.]],
#                   [[1.,0.], [0.]],
#                   [[1.,1.], [0.]]]

#    for dataPoint in trainingSet:
#        print dataPoint

    pma = PMAssembler()
    numGates = pma.train(trainingSet, 1)

    testSet = trainingSet
    pma.test(testSet)

    print "Used", numGates, "gates."
    print "Total gate cost:", pma.totalCost


if mfscTest:

    trainingSet, _, testSet = getRawMnistData()

#    trainingSet = (np.reshape(np.arange(9), (3, 3)), np.arange(3))

#    n = 3
#    numDataPoints = 3
#    numOptions = 2

#    trainingSet = (np.reshape(np.array([random.random() for _ \
#        in range(n*numDataPoints)]), (numDataPoints, n)), \
#        np.array([random.randint(0, numOptions-1) for _ in range(numDataPoints)]))

#    testSet = trainingSet

    mfsc = MatFuncSumClique(10, lambda x: x)
    mfsc.train(trainingSet)
    mfsc.test(testSet, True)

if mfsTest:

    n = 4
    numDataPoints = 3

    trainingSet = (np.reshape(np.array([random.random() for _ \
        in range(n*numDataPoints)]), (numDataPoints, n)), \
        np.array([random.random() for _ in range(numDataPoints)]))

    testSet = trainingSet

    mfs = MatFuncSum()
    mfs.train(trainingSet)
    mfs.test(testSet)

if linearFuncSumDigits:
    rawTrainingData, _, rawTestData = getRawMnistData()

#    rawTrainingData = (np.array([[3,4], [1,-4], [2,0]]), "hi")
 #   rawTrainingData = (np.reshape(np.arange(200), (40, 5)), "hi")

#    print len(rawTrainingData[0])

#    trainingSet = trainingData[:5000]
#    testSet = testData[:5000]

#    trainingSet = [[[3,4], [5]],
#                    [[1,-4], [3]],
#                    [[2, 0], [1]]]



    print "Done getting data"

#    prunedTrainingData = removeDudsFromTrainingData(rawTrainingData[0])

#    print "data pruned"

    findBestLinearOrthogFuncsIter(rawTrainingData[0])

#    n = 784

#    f = FuncSumClique(basicFuncFamilyMaker(n), 10, lambda x: x)

#    print "Beginning training"
#    f.train(trainingSet)
#    print "Done training, beginning testing"
#    f.test(testSet)
#    print "Done"


if linearFitDigits:
    trainingData, _, testData = getMnistData()

    trainingSet = trainingData[:50000]
    testSet = testData[:50000]

    print "Done getting data"


    n = 784

    f = FuncSumClique(basicFuncFamilyMaker(n), 10, lambda x: x)

    print "Beginning training"
    f.train(trainingSet)
    print "Done training, beginning testing"
    f.test(testSet)
    print "Done"


if funcSumTest:
    n = 3
    trainingSet = [
        [[-1.,-1.,-1.], [0.32]],
        [[-1.,-1., 1.], [-5.]],
        [[-1., 1.,-1.], [9.2]],
        [[-1., 1., 1.], [1.2]],
        [[ 1.,-1.,-1.], [-3.4]],
        [[ 1.,-1., 1.], [0.21]],
        [[ 1., 1.,-1.], [9.4]],
        [[ 1., 1., 1.], [-2.2]]
    ]

    testSet = trainingSet

    f = FuncSum(allMultSubsetFuncMaker(n, allListsOfSizeX(n)[:]))

    f.train(trainingSet)
    f.test(testSet, True)

if majoritarianHardStudy:
    n = 3

    m = Majoritarian(allXORSubsetFuncMaker(n, allListsOfSizeX(n)))

    trainingSet = generateRandomTruthTable(n, 1)

    print trainingSet

#    for dataPoint in trainingSet:
#        print dataPoint

    testSet = trainingSet

    m.train(trainingSet, False)
    m.test(testSet, False)



#    m = Majoritarian()

if majoritarianSelfLearn:
    n = 8

    m = Majoritarian(allXORSubsetFuncMaker(n, allSubsetsOfSize(n, 2)))
#    m.params = [random.randint(-4, 4) for _ in range(len(m.functionList))]
    m.params = [random.random() for _ in range(len(m.functionList))]
    trainingSet = m.generateTruthTable(n, True)
#    print m.params

#    m = Majoritarian(allXORSubsetFuncMaker(n, allListsOfSizeX(n)))
#    m = Majoritarian(allXORSubsetFuncMaker(n, allSubsetsOfSize(n, 2)))
#    trainingSet = generateRandomTruthTable(n, 1, False, True)

    testSet = trainingSet

    m.train(trainingSet)
    m.test(testSet, True, False, True, True)


if majoritarianCompression:
    patternSize = 10

    trainingSetSize = 1000
    testSetSize = 1000

#    m = Majoritarian(allXORSubsetFuncMaker(patternSize, allSubsetsOfSize(patternSize, 2)))
    m = Majoritarian(allXORSubsetFuncMaker(patternSize, allListsOfSizeX(patternSize)))

    compressionData = getCompressionData("../../compression/declarationbits.txt",
        patternSize)

    trainingSet = compressionData[:trainingSetSize]
    testSet = compressionData[trainingSetSize:testSetSize+trainingSetSize]

#    trainingSet = generateRandomTruthTable(n, 1)

    m.train(trainingSet)
    m.test(testSet, True, False)

if majoritarianAllFuncs:
    n = 4

    m = Majoritarian(allXORSubsetFuncMaker(n, allListsOfSizeX(n))[1:])

    m.enumerateAllPossibleParams(n)

if majoritarianAllXOR:
    n = 4

    m = Majoritarian(allXORSubsetFuncMaker(n, allListsOfSizeX(n))[1:])
#    m = Majoritarian(allXORSubsetFuncMaker(n, allSubsetsOfSize(n, 2)))

#    random16list = [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    random16list = [0.,1.,1.,0.,0.,0.,0.,0.,0.,0.,
        1.,1.,1.,0.,0.,0.]

#    random16list = [1.0*(random.random() > 0.5) for _ in range(16)]

    print random16list

#    trainingSet = makeTruthTable(n, [1.,1.,1.,0.,0.,0.,0.,0.,0.,0.,
#        1.,1.,1.,0.,0.,0.])

    trainingSet = makeTruthTable(n, random16list*1)
#    trainingSet = generateRandomTruthTable(n, 1)
    for dataPoint in trainingSet:
        print dataPoint

    testSet = trainingSet

#    m.train(trainingSet)
#    m.test(testSet, True, False)

#    m.train(trainingSet)
    m.perfectTrain(trainingSet)
    print m.params

    m.test(testSet, True, True)

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

if randomNandRecoveryTestCWImportance:
    n = 10
    internals = 4
    outputs = 1

    nandDag = nandDagFunctionMaker(n, internals, outputs)
    numParams = numParamCalculator(n, internals, outputs)

    s = CubeWalker(nandDag, [1.0*(random.random()>0.5) for _ in range(numParams)])

    trueFunc, trueParams = randomInstanceGen(nandDag, numParams)
    trainingSet = generateFullTruthTable(trueFunc, n)

    testSet = trainingSet

    s.train(trainingSet, True, True)
    s.test(testSet, True, False)

if randomNandRecoveryTestCW:
    n = 10
    internals = 4
    outputs = 1

    nandDag = nandDagFunctionMaker(n, internals, outputs)
    numParams = numParamCalculator(n, internals, outputs)

    s = CubeWalker(nandDag, [1.0*(random.random()>0.5) for _ in range(numParams)])

    trueFunc, trueParams = randomInstanceGen(nandDag, numParams)
    trainingSet = generateFullTruthTable(trueFunc, n)

    testSet = trainingSet

    s.train(trainingSet, False, True)
    s.test(testSet, True, False)

if randomNandRecoveryTestSample:
    n = 8
    internals = 8
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

if compressionTestCWImportanceWeighted:
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

    c.train(trainingSet, True, True)
    c.test(testSet, True, True)

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

    c.train(trainingSet, False, True)
    c.test(testSet, True, True)
