import sys


    # This file takes multiple inputs, explained below, in order to iterate over many problems and several swap values
    # This is for plotting purposes, it's main goal is to find the success probability of the algorithm against Swap steps
    # Among other data collection for unsuccessful attempts


IOFile = "../../IOData/"+sys.argv[1]  # Input/Output data. Contained in IOData folder.
ProblemStart = int(sys.argv[2]) # Problem to start at, should be \in [0,len(IOFile)]
incr = int(sys.argv[3]) # Number of problems to iterate through, starting at ProblemStart
R = int(sys.argv[4]) # Number of replicas to take. Generally, we go up in 2's starting from R=5 for n=3. So R=7 for n=4 etc. 
                     # This is an artifact of the theoretical `ideal` swap probability for parallel tempering.
SwapMin = int(sys.argv[5]) # Lower bound for number of swaps
SwapMax = int(sys.argv[6]) # Upper bound
PointNum = int(sys.argv[7]) # Number of steps between lower and upper to take
name = sys.argv[8] # Output file name

import ParallelTempering as pt
import json
import numpy as np
io=open(IOFile)
ProblemInstances=json.load(io)
faults=len(ProblemInstances[0][0])
F = 1 # Generally we start searching for solutions at F=1

swaplist=np.linspace(SwapMin,SwapMax,PointNum,dtype=int)
for S in swaplist:
    F_c = 0
    P_s = 0
    TimeToGS = 0
    Num_FalsePos = 0
    Num_FalsePosBeforeSol = 0
    Num_PairsSat = 0
    Min_PairsSat = 0
    Max_PairsSat = 0
    for problem in ProblemInstances[ProblemStart:ProblemStart+incr]:
        run = 0
        runmax = 100
        success = 0
        TotalFalsePos = 0
        FalsePosBeforeSol = 0
        PairsSat = 0
        MinPairs = 0
        MaxPairs = 0
        TTGS = 0
        false_counter = 0
        while run < runmax:

            (solnflag, FalsePositives, ttgs)=pt.ParTemp(problem,F,R,S,[0,3])
            
            solns_correct_faults = 0
            temporary_list = []
            for element in FalsePositives:
                if element[0]==faults:
                    solns_correct_faults+=1
                    temporary_list.append(element[1])

            if ttgs[-1][0]==faults:
                final_ttgs = ttgs[-1][1]

            success = success + solnflag
            numnonzero = solns_correct_faults
            
            if solnflag==1:
                FalsePosBeforeSol = FalsePosBeforeSol + numnonzero
            TotalFalsePos = TotalFalsePos+len(FalsePositives)

            if numnonzero==0:
                mean = 0
                MinFalsePositives = 0
                MaxFalsePositives = 0
            else:
                false_counter += 1
                mean = np.mean(temporary_list)
                MinFalsePositives = min(temporary_list)
                MaxFalsePositives = max(temporary_list)
            PairsSat = PairsSat + mean
            MinPairs = MinPairs + MinFalsePositives
            MaxPairs = MaxPairs + MaxFalsePositives
            TTGS = TTGS + final_ttgs
            run+=1
        F_c = F_c + false_counter
        P_s = P_s + success
        Num_FalsePos = Num_FalsePos + TotalFalsePos
        Num_FalsePosBeforeSol = Num_FalsePosBeforeSol + FalsePosBeforeSol
        Num_PairsSat = Num_PairsSat + PairsSat
        TimeToGS = TimeToGS + TTGS
        Min_PairsSat = Min_PairsSat + MinPairs
        Max_PairsSat = Max_PairsSat + MaxPairs
    f=open(FileName,"a+")
    f.write("%s %s %s %s %s %s %s %s %s %s\n"%(S, P_s, Num_FalsePosBeforeSol, Num_FalsePos, Num_PairsSat, Min_PairsSat, Max_PairsSat, TimeToGS, F_c,runmax * incr))
    f.close()
