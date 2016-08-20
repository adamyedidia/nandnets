import random

class RandomFullTTExtrapolator:
    def __init__(self, tt):
        self.tt = tt
        self.numInputs = len(tt[0][0])
        
    def evaluateLinear(self, inp):
        randomDataPoint = random.choice(self.tt)
        randomIndex = random.randint(0, self.numInputs-1)
        
        return (randomDataPoint[0][randomIndex] + randomDataPoint[1][0] + \
            inp[randomIndex]) % 2 