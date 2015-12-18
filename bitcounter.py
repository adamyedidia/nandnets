import sys
import string
import random

def allStringsOfLength(x):
	if x == 0:
		return [""]
		
	else:
		oneLess = allStringsOfLength(x-1)
		return [s + "0" for s in oneLess] + [s + "1" for s in oneLess]

def getCompressionData(path, historySize, size=-1):
	
	data = []

	bitStream = string.strip(open(path, "r").read(size))

	scoreTracker = {}
	randomScoreTracker = {}

	history = [0.0] * historySize

	for bit in bitStream:
		data.append([history, [float(bit)]])
			
		history = [float(bit)] + history[:-1]

	return data
	
	