import random

class QuickSampler:
    def __init__(self, getTrainSample, getTestSample, listOfFuncs, decreaseProb=0.1):
        self.getTrainSample = getTrainSample
        self.getTestSample = getTestSample
        self.numSamples = 1.
        self.listOfFuncs = listOfFuncs
        self.decreaseProb = decreaseProb
        
    def evaluate(self, inputs):
        sumVotes = 0
        for _ in range(int(self.numSamples)):
            sample = self.getTrainSample()
            
            sampleInputs = sample[0]
            sampleOutput = sample[1]
            
            for func in self.listOfFuncs:
                                
                sumVotes += func(sampleInputs) * sampleOutput * func(inputs)
        
        
#        print "hi", self.numSamples, sumVotes
        
        sumVotes /= self.numSamples
        
#        print sumVotes
        
        if sumVotes < -1:
            self.numSamples *= -sumVotes
            return -1.
            
        elif sumVotes > 1:
            self.numSamples *= sumVotes
            return 1.
            
        else:
            self.decreaseSamples()
            
            if random.random()*2 - 1 < sumVotes:
                return 1
            return -1
            
    def decreaseSamples(self):
        if random.random() < self.decreaseProb:
            self.numSamples -= 1
            
    def test(self, howMany):
        numCorrect = 0.
        numTotal = 0.
        
        for i in range(howMany):
            print i, self.numSamples
            
            testSample = self.getTestSample()
            
            inputs = testSample[0]
            output = testSample[1]
            
            result = self.evaluate(inputs)
            
            if result == output:
                numCorrect += 1.
            numTotal = 1.
            
        print numCorrect / numTotal