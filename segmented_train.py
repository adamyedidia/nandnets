import time

def pront(x):
    print x

# Returns whether the toggle was successful (-1 if so); if out-of-bounds error, it returns
# depth is to be changed only in the description of the function
# Also returns what was written
def arrayToggle(array, indexList, depth=0):
    if indexList[0] >= len(array):
        return (depth, None)

    if len(indexList) == 1:
        valueToBeWritten = 1 - array[indexList[0]]

        array[indexList[0]] = valueToBeWritten
        return (-1, valueToBeWritten)

    else:
        return arrayToggle(array[indexList[0]], indexList[1:], depth+1)

# Just returns whether there's an out-of-bounds error; makes no modifications.
def arrayQuery(array, indexList, depth=0):
    if indexList[0] >= len(array):
        return depth

    if len(indexList) == 1:
        return -1

    else:
        return arrayQuery(array[indexList[0]], indexList[1:], depth+1)


def makeIndexListLegal(multiDimensionalArray, indexList):
    indexOfOutOfBounds = arrayQuery(multiDimensionalArray, indexList)

    if indexOfOutOfBounds == -1:
        # no out of bounds, toggle successful
        return True

    else:
        indexList[indexOfOutOfBounds] = 0
        if indexOfOutOfBounds == 0:
            return False

        else:
            indexList[indexOfOutOfBounds-1] += 1
            return makeIndexListLegal(multiDimensionalArray, \
                indexList)

# This one is different from *AndArrayToggle; it will increment the list,
# then check to make sure that the resulting list is valid
def incrementIndexList(multiDimensionalArray, indexList):
    indexList[-1] += 1
    return makeIndexListLegal(multiDimensionalArray, indexList)

# returns (success, value written)
# Warning: modifies multiDimensionalArray AND indexList
def incrementIndexListAndArrayToggle(multiDimensionalArray, indexList):

    indexOfOutOfBounds = arrayToggle(multiDimensionalArray, indexList)
    if indexOfOutOfBounds[0] == -1:
        # no out of bounds, toggle successful
        indexList[-1] += 1
        return (True, indexOfOutOfBounds[1])

    else:
        # out of bounds happened, toggle unsuccessful
        indexList[indexOfOutOfBounds[0]] = 0

        if indexOfOutOfBounds[0] == 0:
            return (False, 0)
            # Failure; this is the loop-breaking case

        else:
            indexList[indexOfOutOfBounds[0]-1] += 1
            return incrementIndexListAndArrayToggle(multiDimensionalArray, \
                indexList)


# returns success
def incrementArray(multiDimensionalArray, dimension):
    indexList = [0] * dimension

    continueLoop = True
    while continueLoop:
        returnValue = incrementIndexListAndArrayToggle(multiDimensionalArray, indexList)
        continueLoop = (returnValue[1] == 0)

    return returnValue[0]

def deepCopy(array, dimension):
    if dimension == 1:
        return array[:]

    else:
        return [deepCopy(i, dimension-1) for i in array]

def exhaustiveSearch(multiDimensionalArray, dimension, funcToTrain, \
    trainingObject, errorFunc, verbose=False):

    indexList = [0] * dimension

    bestError = float("Inf")
    bestArray = None

    t = time.time()

    numSteps = 0

    while True:
        pront("Took " + str(time.time() - t) + " seconds.")
        t = time.time()
        pront("Analyzing array " + str(multiDimensionalArray))


        pront("Current best: " + str(bestError))

        currentFunc = funcToTrain(multiDimensionalArray)
        currentError = errorFunc(currentFunc, trainingObject)

        if currentError < bestError:
            bestError = currentError
            bestArray = deepCopy(multiDimensionalArray, dimension)

        if not incrementArray(multiDimensionalArray, dimension):
            break

        numSteps += 1

        if numSteps == 256:
            # KLUDGE!!
            break

    return bestArray

# modifies multiDimensionalArray
def aggressiveSearch(multiDimensionalArray, dimension, funcToTrain, \
    trainingObject, errorFunc, verbose=False):

    indexList = [0] * dimension

    currentFunc = funcToTrain(multiDimensionalArray)
    currentError = errorFunc(currentFunc, trainingObject)

    t = time.time()

    changeHappened = True
    while changeHappened:
        incrementSuccess = True
        changeHappened = False

        while incrementSuccess:

            arrayToggle(multiDimensionalArray, indexList)

            testFunc = funcToTrain(multiDimensionalArray)
            newError = errorFunc(testFunc, trainingObject)

#            pront("Took " + str(time.time() - t) + " seconds.")
            t = time.time()
#            pront("Analyzing array " + str(multiDimensionalArray))
#            pront("Current best: " + str(currentError))


            if newError < currentError:
                currentError = newError
                changeHappened = True
                pront("Current best array: " + str(multiDimensionalArray) + \
                    " with error " + str(currentError))

            # otherwise we need to toggle back
            else:
                arrayToggle(multiDimensionalArray, indexList)


            incrementSuccess = incrementIndexList(multiDimensionalArray, indexList)

        indexList = [0] * dimension

# WARNING: This function modifies multiDimensionalArray!
# WARNING: This function depends on the mutability of lists!
# If you ever want an example why that feature is desirable, this is one.
def segmentedTrain(multiDimensionalArray, dimension, funcToTrain, \
    trainingObject, errorFunc, verbose=False):

    changeHappened = True
    while changeHappened:
        changeHappened = trainOneStep(multiDimensionalArray, multiDimensionalArray, dimension, \
            funcToTrain, trainingObject, errorFunc, verbose)

        if dimension == 3 and verbose:
            pront("Exploring around gate " + str(multiDimensionalArray))


        if dimension == 2:
#            pront("Trying wiring " + str(multiDimensionalArray))
            pass

def trainOneStep(fullArray, subArray, dimension, funcToTrain, \
    trainingObject, errorFunc, verbose=False):

    if dimension == 1:
        # Base case: find best neighbor and go there

        originalFunc = funcToTrain(fullArray)
        originalError = errorFunc(originalFunc, trainingObject)

        bestError = originalError
        bestIndex = None
        vectorChanged = False

        for i, elt in enumerate(subArray):
            subArray[i] = 1-elt

            # Try the change
            neighborFunc = funcToTrain(fullArray)
            neighborError = errorFunc(neighborFunc, trainingObject)

            if neighborError < bestError:
                bestError = neighborError
                bestIndex = i
                vectorChanged = True

            # change it back
            subArray[i] = elt

        if bestIndex != None:
            subArray[bestIndex] = 1 - subArray[bestIndex]

        if verbose:
            pront("Determined optimal subshift: " + str(fullArray) + " with error " + \
                str(bestError))

        return vectorChanged

    elif dimension > 1:
        # The inductive case

        changeHappened = False

        # We want to trainOneStep on each child
        for i, subSubArray in enumerate(subArray):
            arrayChanged = trainOneStep(fullArray, subArray[i], dimension-1, funcToTrain, trainingObject,
                errorFunc, verbose)

            changeHappened = changeHappened or arrayChanged

        return changeHappened
