import random
import numpy as np
from funcs import allTuplesOfSizeX, hammingDistance

def generateSpace(d, slope):
    allTuples = allTuplesOfSizeX(d)

    space = {}

    optimum = tuple([1*random.random() for _ in range(d)])

    for tup in allTuples:
        tupDistance = hammingDistance(tup, optimum)
        space[tup] = np.random.normal(tupDistance*slope, 1.0)

    return space
