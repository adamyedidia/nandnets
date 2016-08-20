import random
from truthtableprinter import printTruthTable

def pront(x):
    print x

def listEquals(l1, l2):
    assert len(l1) == len(l2)

    for i, val in enumerate(l1):
        if l2[i] != val:
            return False

    return True

class CubeWalker:
    def __init__(self, f, initialParams):
        self.f = f
        self.params = initialParams

    def evaluate(self, inputs, params=None):
        if params == None:
            params = self.params

        return self.f(inputs, params)

    def testParamAssignmentOnInput(self, inputs, outputs, outputIndex, paramAssignment):
        result = self.evaluate(inputs, paramAssignment)

        if outputs[outputIndex] == result[outputIndex]:
            return 1

        return 0

    def generateDeviationsFromParams(self):
        listOfDeviateParams = []

        for i, val in enumerate(self.params):
            deviateParams = self.params[:]
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

    def scoreParamsImportanceWeighted(self, trainingSet, paramAssignment):
        numOutputs = len(trainingSet[0][1])

        totalSwaps = 0.
        goodSwaps = 0.

        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            outputs = dataPoint[1]

            for outputIndex in range(numOutputs):
                # This is inefficient by a factor of 2!
                currentPerformance = self.testParamAssignmentOnInput(inputs, outputs, outputIndex, self.params)
                alternatePerformance = self.testParamAssignmentOnInput(inputs, outputs, outputIndex, paramAssignment)

                if currentPerformance != alternatePerformance:
                    totalSwaps += 1.

                    if alternatePerformance == 1:
                        goodSwaps += 1.

        if totalSwaps == 0.:
            return 0.5

        return goodSwaps / totalSwaps

    def scoreParamsBoth(self, trainingSet, paramAssignment):
        numOutputs = len(trainingSet[0][1])

        totalSwaps = 0.
        goodSwaps = 0.

        score = 0

        for dataPoint in trainingSet:
            inputs = dataPoint[0]
            outputs = dataPoint[1]

            for outputIndex in range(numOutputs):
                # This is inefficient by a factor of 2!
                currentPerformance = self.testParamAssignmentOnInput(inputs, outputs, outputIndex, self.params)
                alternatePerformance = self.testParamAssignmentOnInput(inputs, outputs, outputIndex, paramAssignment)

                score += alternatePerformance

                if currentPerformance != alternatePerformance:
                    totalSwaps += 1.

                    if alternatePerformance == 1:
                        goodSwaps += 1.

        if totalSwaps == 0.:
            return score, 0.5

        return score, (goodSwaps / totalSwaps)

    def walkOneStep(self, trainingSet, importanceWeighted=False):
        listOfDeviateParams = self.generateDeviationsFromParams()

        bestScore = -1
        bestParams = None
        bestSwapFraction = -1
        bestNumCorrect = -1

        for deviateParams in listOfDeviateParams:
            numCorrect, swapFraction = self.scoreParamsBoth(trainingSet, deviateParams)

            if importanceWeighted:
                score = swapFraction
            else:
                score = numCorrect

            if score > bestScore:
                bestScore = score
                bestParams = deviateParams
                bestSwapFraction = swapFraction
                bestNumCorrect = numCorrect

        print bestSwapFraction, bestNumCorrect

        self.params = bestParams
        return bestScore

    def train(self, trainingSet, importanceWeighted=False, verbose=False, solution=None):
        seenParamsBefore = False
        stepCounter = 0

        oldParams = self.params
        oldOldParams = self.params

        score = -1
        oldScore = 0

        if verbose:
            pront("Params: " + str(self.params))

        while not (seenParamsBefore and scoreDiff >= 0):
            score = self.walkOneStep(trainingSet, importanceWeighted)
            stepCounter += 1
            pront("Taken " + str(stepCounter) + " step" + ("s"*(stepCounter != 1)) + ".")
            if verbose:
                pront("Params: " + str(self.params))

#                printTruthTable(self.f, len(trainingSet[0][0]), self.params, solution)
            pront("Score: " + str(score))
            pront("")

            seenParamsBefore = listEquals(self.params, oldOldParams)

            oldOldParams = oldParams
            oldParams = self.params

            scoreDiff = score-oldScore

            oldScore = score

    def test(self, testSet, randomOutcomes=False, verbose=False):
        correctnessCounter = 0.0
        randomCounter = 0.0
        alwaysZeroCounter = 0.0
        overallCounter = 0.0

        for dataPoint in testSet:
            inputs = dataPoint[0]
            correctOutputs = dataPoint[1]

            myOutputs = self.evaluate(inputs)

            if verbose:
                pront("Correct: " + str(correctOutputs))
                pront("Observed: " + str(myOutputs))
                pront("")

            for i in range(len(correctOutputs)):
                if round(myOutputs[i]) == correctOutputs[i]:
                    correctnessCounter += 1.0

                if random.random() < 0.5:
                    randomCounter += 1.0

                if not correctOutputs[i]:
                    alwaysZeroCounter += 1.0

                overallCounter += 1.0

        pront("Got " + str(correctnessCounter) + " out of " + str(overallCounter) + " correct.")
        pront("")

        if randomOutcomes:
            pront("Compare to the random outcome: ")
            pront("Got " + str(randomCounter) + " out of " + str(overallCounter) + " correct.")
            pront("")
            pront("Compare to the outcome you'd have gotten if you always picked zero: ")
            pront("Got " + str(alwaysZeroCounter) + " out of " + str(overallCounter) + " correct.")

        return correctnessCounter / overallCounter
