import random
from funcs import tinyNandNorBinOutput, allListsOfSizeX, allTuplesOfSizeX, \
    nandDagOneOutputFunctionMaker, andWithNot, perceptronMaker, \
    smallPerceptronStack

# ^ is the bitwise xor operation in Python
def hammingDistanceInt(int1, int2):
    return bin(int1 ^ int2).count("1")

def bitwiseXorInt(int1, int2):
    return int1 ^ int2

def hammingDistance(list1, list2):
    return sum([int(i) ^ int(j) for i,j in zip(list1, list2)])

def bitwiseXor(list1, list2):
    return tuple([int(i) ^ int(j) for i,j in zip(list1, list2)])

class ParameterMachine:
    # P is number of params
    # f is the function's name
    def __init__(self, f, P, N):
        self.f = f
        self.P = P
        self.N = N
        self.listOfLinks = tuple([[1*(i==j) for j in range(P)] for i in range(P)])

    def rearrangeOnce(self, listOfInputs, params=None):
        if params == None:
            params = self.params

        listOfOutputsHere = self.generateListOfOutputs(listOfInputs, params)


        listOfLinkedListsOfOutputs = [(self.generateListOfOutputs(listOfInputs, bitwiseXor(params, link)), link) \
            for link in self.listOfLinks]

#        print (self.generateListOfOutputs(listOfOutputsHere, bitwiseXor(params, self.listOfLinks[0])), self.listOfLinks[0])

        listOfJumpsSorted = sorted([(bitwiseXor(listOfOutputsHere, listOfOutputsThere[0]), listOfOutputsThere[1]) for \
            listOfOutputsThere in listOfLinkedListsOfOutputs], key=lambda x : sum(x[0]), reverse=True)

        combinedJump = bitwiseXor(listOfJumpsSorted[0][1], listOfJumpsSorted[1][1])

        listOfOutputsAtCombined = self.generateListOfOutputs(listOfInputs, bitwiseXor(combinedJump, params))
        worstJumpSize = sum(listOfJumpsSorted[0][0])
        jumpSizeAtCombined = sum(bitwiseXor(listOfOutputsHere, listOfOutputsAtCombined))

#        print listOfJumpsSorted
#        print combinedJump, jumpSizeAtCombined

        if jumpSizeAtCombined < worstJumpSize:
            self.listOfLinks.remove(listOfJumpsSorted[0][1])
            self.listOfLinks.append(combinedJump)

            print "Removing link", listOfJumpsSorted[0][1]
            print "Adding link", combinedJump

    def evaluate(self, inputs, params=None):
        if params == None:
            params = self.params

#        print inputs, params

        return self.f(inputs, params)

    def generateListOfOutputs(self, listOfInputs, params=None):
        listOfOutputs = []
        for inputs in listOfInputs:
            listOfOutputs.append(self.evaluate(inputs, params))

        return tuple(listOfOutputs)

    def scoreParamAssignment(self, listOfInputs, listOfOutputs, params=None):
        if params == None:
            params = self.params

        return hammingDistance(self.generateListOfOutputs(listOfInputs, params), listOfOutputs)

    def getLinkDifference(self, listOfInputs, link, params):
        return hammingDistance(self.generateListOfOutputs(listOfInputs, params),
                            self.generateListOfOutputs(listOfInputs, bitwiseXor(params, link)))

    def generateRandomParams(self):
        params = []
        for _ in range(self.P):
            params.append(1*random.random()>0.5)

        return tuple(params)

    def generateRandomInput(self):
        inp = []
        for _ in range(self.N):
            inp.append(1*random.random()>0.5)

        return tuple(inp)

    def generateListOfRandomInputs(self, listSize):
        listOfInputs = []
        for _  in range(listSize):
            listOfInputs.append(self.generateRandomInput())

        return tuple(listOfInputs)

    def getLinkDifferenceExact(self, listOfInputs, link):
        totalLinkDifference = 0

        allParams = allTuplesOfSizeX(self.P)

        for params in allParams:
            print params, self.getLinkDifference(listOfInputs, link, params)

            totalLinkDifference += self.getLinkDifference(listOfInputs, link, params)

        return float(totalLinkDifference)/len(allParams)

    def getLinkDifferenceSampledParams(self, listOfInputs, link, numParamSamples=1000):
        totalLinkDifference = 0

        for _ in range(numParamSamples):
            totalLinkDifference += self.getLinkDifference(listOfInputs, link, self.generateRandomParams())

        return float(totalLinkDifference)/numParamSamples

    def getLinkDifferenceSampledParamsSampledInputs(self, listOfInputs, link, numParamSamples=1000, numInputSamples=1000):
        totalLinkDifference = 0

        for _ in range(numParamSamples):
            totalLinkDifference += self.getLinkDifference(self.generateListOfRandomInputs(numInputSamples),
                link, self.generateRandomParams())

        return float(totalLinkDifference)/numParamSamples

    def getAllLinkDifferencesExact(self, listOfInputs):
        listOfAllLinks = allTuplesOfSizeX(self.P)

        listOfLinkValues = []

        for link in listOfAllLinks:
            linkDiff = self.getLinkDifferenceExact(listOfInputs, link)

            print link, linkDiff

            listOfLinkValues.append((link, linkDiff))

        listOfLinkValues.sort(key=lambda x : x[1])

        for linkTup in listOfLinkValues:
            print linkTup

    def lookAtParamLocationLinkDifferences(self, listOfInputs):
        listOfAllParams = allTuplesOfSizeX(self.P)
        listOfAllLinks = allTuplesOfSizeX(self.P)

        for params in listOfAllParams:
            for link in listOfAllLinks:
                linkDiff = self.getLinkDifference(listOfInputs, link, params)

                print params, link, linkDiff

#    def measureAverageDifferences(self, )

    # Returns whether or not we walked a step
    # modifies self.params
    def walkOneStep(self, listOfInputs, listOfOutputs):
        currentPerformance = self.scoreParamAssignment(listOfInputs, listOfOutputs)

        bestLink = None
        bestPerformance = currentPerformance

        for link in self.listOfLinks:
            linkPerformance = self.scoreParamAssignment(listOfInputs, listOfOutputs, bitwiseXor(self.params, link))

            if linkPerformance < bestPerformance:
                bestPerformance = linkPerformance
                bestLink = link

        if bestLink == None:
            return False

        else:
            self.params = bitwiseXor(self.params, bestLink)
            return True

    def cubeWalk(self, listOfInputs, listOfOutputs):
        while self.walkOneStep(listOfInputs, listOfOutputs):
            pass

    def test(self, listOfInputs, listOfOutputs):
        myListOfOutputs = self.generateListOfOutputs(listOfInputs)

        print "Error count:", hammingDistance(myListOfOutputs, listOfOutputs)

P = 7
N = 3

#pm = ParameterMachine(smallPerceptronStack, P, N)

#pm = ParameterMachine(perceptronMaker(N), P, N)
#pm = ParameterMachine(andWithNot, P, N)

#pm = ParameterMachine(tinyNandNorBinOutput, P, N)
pm = ParameterMachine(nandDagOneOutputFunctionMaker(N, 2), P, N)

fullListOfInputs = allTuplesOfSizeX(N)

pm.lookAtParamLocationLinkDifferences(fullListOfInputs)

#pm.params = [0,0,0]
#pm.cubeWalk(fullListOfInputs, [1,1,1,0])
#pm.test(fullListOfInputs, [1,1,1,0])

#for _ in range(30):
#    randomParams = [1*(random.random() < 0.5) for _ in range(P)]
#    pm.rearrangeOnce(fullListOfInputs, randomParams)

#print pm.listOfLinks
#pm.params = [0,0,0]
#pm.cubeWalk(fullListOfInputs, [1,1,1,0])
#pm.test(fullListOfInputs, [1,1,1,0])
