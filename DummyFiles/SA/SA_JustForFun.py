# This file is explanatory. Give it some inputs on the command line, and it will print the algorithm process. 
# IOData is kept two folders up, this can be changed. 
# This file chooses one random problem to tackle and does so, starting from F = 1

import sys

IOFile = "../../IOData/"+sys.argv[1]  # Input/Output data. Contained in IOData folder.
Sweeps = int(sys.argv[2]) # Number of sweeps

import SimulatedAnnealing_DebuggingFile as sa
import json
import numpy as np
import random

io=open(IOFile)
ProblemInstances=json.load(io)
p = random.randint(0,len(ProblemInstances)-1)

problem = ProblemInstances[p]

(solnfound, false_positives, time_to_gs) = sa.SA(problem, 1, Sweeps, [0,3])

if solnfound == 0:
    print("No global solution found.")
else:
    print("Solution found!")

print("Pairs satisfied for each false positive solution: \n%s"%false_positives)

print("Time to solution for each fault: \n %s"%time_to_gs)
