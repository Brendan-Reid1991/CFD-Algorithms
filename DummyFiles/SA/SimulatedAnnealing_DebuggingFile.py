import PropagateCircuit as prop
import PositionDefinitions as pos 
import numpy as np
import json
from scipy.io import mmread
import random
import os

spinpath=os.path.abspath("spinweights%s.dat"%Dimension)
swfile=open(spinpath)
SpinWeights=json.load(swfile)
jpath=os.path.abspath("JMat%s.mtx"%Dimension)
hpath=os.path.abspath("HMat%s.dat"%Dimension)
cpath=os.path.abspath("const%s.dat"%Dimension)
J=mmread(jpath)
H=np.loadtxt(hpath)
C=np.loadtxt(cpath)


#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# Function below is the simulated algorithm.
# It's arguments are:
#
# data : two dimensional list, the first element should be the correct fault configuration
#        the second element should be another list, an N * 3 matrix with N the number of I/O pairs, 
#            and the columns correspond to input_1, input_2, and output vectors
#
# initial_faults : initial (integer) number of faults to consider
#
# MetSweeps : integer number of metropolis sweeps to consider
#   
#TempRange : inverse temperature range, should be written as [beta_i,beta_f]
#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 


def SA(data, initial_faults, MetSweeps, TempRange):

    # data file decomposed into relevant chunks: true fault config and the input/ouput pairs. 
    # dim(iop) = N * 3
    truefaults = data[0]
    iop = data[1]
    print("True fault configuration: %s"%np.sort(truefaults))

    betavalues = np.linspace(TempRange[0],TempRange[1],MetSweeps)

    #Boltzmann acceptability criteria table defined
    energychange = range(1,101)
    bltz={}
    for delta in energychange:
        for beta in betavalues:
            bltz[delta,beta]=np.exp(- delta * beta)
    
    MaxF=len(truefaults) # Maximum number of faults to consider
    
    L_io=len(iop)

    #Circuit dimension devised. Relevant constants defined.
    # L_vec is the length of the spin vector, including input/output pairs. 
    # Removing these (total length 4n) gives us the max number of spins to perform the sweep over
    n = len(iop[0][0])
    L_vec = 24*n**2-30*n
    FlipMax = L_vec-4*n

    # Fault and (auxiliary) spin positions defined externally for dimension n
    (FaultPos,SpinPos) = pos.Positions(n)
    
    # FlipFaultProb is used to decide whether a fault spin should be flipped, as this requires 
    # a separate process from an auxiliary spin but they should be flipped a proportional
    # amount of time  
    FaultFlipProb = len(FaultPos)/(len(FaultPos)+len(SpinPos))
    
    OutPos = pos.output_pos(n)


    # Spin vector defined, randomly chosen i/o pair to begin
    F = initial_faults
    print("Starting program at %s faults; maxing out at %s"%(F,MaxF))
    
    vector = np.zeros(L_vec) # Spin placement is important which is why we choose an empty vector rather than a list
    
    p = np.random.randint(0,L_io)
    print("Starting with pair %s of %s"%(p,len(iop)))

    #Input vectors
    for x in range(0,n):
        vector[x] = iop[p][0][x]
        vector[x+n] = iop[p][1][x]
    #Output vectors
    it = 0
    for position in OutPos:
        vector[position] = iop[p][2][it]
        it+=1

    #Active faults
    activefaults = random.sample(list(FaultPos),F) # Faults randomly chosen to be active
    inactivefaults = []
    for position in list(FaultPos):
        if position in activefaults:
            vector[position] = 1
        else:
            vector[position] = -1
            inactivefaults.append(position)
    print("\nInitial active faults: %s"%(np.sort(activefaults)))
    # All auxiliary spins randomised
    for position in SpinPos:
        vector[position] = np.random.choice([1,-1])

    #Calculating initial energy:
    energy = vector@J@vector+vector@H+C
    print("\nInitial energy: %s\n"%energy)

    # FalsePos is a data collecting entity. 
    # Records how many pairs are satisfied. 
    # E.g., FalsePos will look like: 
    # FalsePos = [3, 1, 1, 4, 5] so 5 solutions were found before the program quit: 
    # satisfying, respectively, 3 pairs, 1 pair, 1 pair and 4 and 5 pairs. 
    FalsePos = []

    # first_gs will record when the first groundstate was found, per fault. Will return zero if none found.
    first_gs = []

    while F<=MaxF:

        solnflag = 0 # Solution flag set to zero initially
        FailSafe = 0 # Failsafe set to zero. This prevents infinite loops in the algorithm
        if F>initial_faults:
            print("Increasing faults to %s"%F)
            # If faults have increased since the program began, we choose one random fault to turn on;
            # i.e. move it from list (inactivefaults) to (activefaults)
            TurnOn=random.sample(inactivefaults,1)[0]
            
            idx = inactivefaults.index(TurnOn)
            inactivefaults.pop(idx)
            activefaults.append(TurnOn)

            vector[TurnOn] = 1
            energy = vector@J@vector+vector@H+C
            print("New activefaults: %s\n New energy: %s"%(np.sort(activefaults),energy))
        while FailSafe<2*L_io:
            print("\nFailsafe is at %s\n"%FailSafe)
            sweep=0
            sweep_intervals = np.linspace(0,MetSweeps,10,dtype=int)
            while sweep<MetSweeps:
                flip=0
                random.shuffle(list(SpinPos)) # auxiliary spins are chosen sequentially; 
                                              # randomised at the start of each sweep
                auxcounter=0
                while flip<FlipMax:
                    if np.random.uniform(0,1)<FaultFlipProb:

                        # If a fault spin is chosen, we need to turn one bit off and one bit on,
                        # This maintains the fault cardinality of the system

                        # Variables (on,off) are indices rather than actual fault positions,
                        # If the spin flip is accepted it is more efficient to alter list positions rather than elements
                        
                        on = random.sample(activefaults,1)[0]
                        off = random.sample(inactivefaults,1)[0]

                        vector[on] = - vector[on]
                        vector[off] = - vector[off]

                        # These calculate the energy change if we exchange the sign of the two fault spins
                        delta1 = 0
                        for element in SpinWeights[on][1][0:-1]:     
                            delta1 = delta1 + (vector[element[0]-1]*element[1]*vector[on])
                        delta2 = 0
                        for element in SpinWeights[off][1][0:-1]:     
                            delta2 = delta2 + (vector[element[0]-1]*element[1]*vector[off])
                        Delta1 = vector[on]*SpinWeights[on][1][-1] + delta1
                        Delta2 = vector[off]*SpinWeights[off][1][-1] + delta2

                        # If the energy change is negative**, i.e. energy decreases, we accept the flip and exchange list indices
                        #   **or it increases but the Boltzmann acceptability criteria is satisfied

                        # If at any point the cost function reaches zero, we have reached a minimum and it needs to be checked,
                        # At this point the sweep process halts
                        if Delta1+Delta2 <= 0:

                            idx = activefaults.index(on)
                            activefaults.pop(idx)
                            activefaults.append(off)
                            
                            
                            idx = inactivefaults.index(off)
                            inactivefaults.pop(idx)
                            inactivefaults.append(on)
                            
                            
                            energy = energy+Delta1+Delta2
                            if energy == 0:
                                break
                            

                        elif np.random.uniform(0,1)<bltz[int(Delta1+Delta2),betavalues[sweep]]:
                            idx = activefaults.index(on)
                            activefaults.pop(idx)
                            activefaults.append(off)

                            idx = inactivefaults.index(off)
                            inactivefaults.pop(idx)
                            inactivefaults.append(on)
                            
                            energy = energy+Delta1+Delta2
                            if energy == 0:
                                break
                            
                        else:
                            # If the energy increases and the Boltzmann acceptability criteria is NOT satisfied,
                            # The spin signs are flipped back; lists are not amended; cost is not changed
                            vector[on] = - vector[on]
                            vector[off] = - vector[off]
                            
                    else:
                        
                        #Chosen aux spin is done sequentially, spinchoice variable increased by reset at the start of every sweep
                        
                        spin = SpinPos[auxcounter]
                        vector[spin] = - vector[spin]
                        
                        auxcounter = (auxcounter+1)%len(SpinPos)
                        

                        # Spin sign changed and energy change calculated
                        # Same criteria as with fault spin
                        # Except no lists needed amended!

                        delta = 0
                        for element in SpinWeights[spin][1][0:-1]:     
                            delta = delta + (vector[element[0]-1]*element[1]*vector[spin])
                        Delta = vector[spin]*SpinWeights[spin][1][-1]+delta

                        if Delta <= 0:
                            energy = energy + Delta
                            if energy == 0:
                                break
                            
                        elif np.random.uniform()<bltz[int(Delta),betavalues[sweep]]:
                            energy = energy + Delta
                            if energy == 0:
                                break                        
                            
                        else:
                            vector[spin] = - vector[spin]
                            
                    if energy == 0:
                        break
                    flip+=1
                if energy == 0:
                    break
                if sweep in sweep_intervals:
                    print("Sweep %s, energy %s"%(sweep+1,energy))
                sweep+=1
            if sweep==MetSweeps:
                break
            if energy==0:
                
                if FailSafe==0:
                    first_gs.append([F,sweep/MetSweeps])
                active = [x+1 for x in activefaults] # the propagate function takes `real` positions, rather than pythonic
                print('Configuration %s works for pair %s'%(np.sort(active),p))
                for p in range(0,len(iop)):
                    val=prop.propagate(iop[p][0],iop[p][1],iop[p][2],active)[0]
                    if val==0:
                        # if the config worked for that pair, increase solnflag. 
                        # If solnflag==L_io it works for ALL pairs and is a global ground state
                        solnflag+=1
                        if solnflag==L_io:
                            FalsePos.append([F,solnflag])
                            print("\n\nConfiguration worked for all pairs.\n\n")
                            break
                if 0<solnflag<L_io:
                    print("\nConfiguration failed; worked for %s of %s pairs."%(solnflag,L_io))
                    FalsePos.append([F,solnflag]) # solution recorded for faults and how many pairs were satisfied.
                    p=np.random.randint(0,L_io) # new i/o pair chosen
                    print("\nRestarting with pair %s"%p)
                    solnflag = 0 

                    # Similar to the start, new input/output values are inputted.
                    # New faults are chosen as well.

                    #Input vectors
                    for x in range(0,n):
                        vector[x]=iop[p][0][x]
                        vector[x+n]=iop[p][1][x]
                    #Output vectors
                    it=0
                    for position in OutPos:
                        vector[position]=iop[p][2][it]
                        it+=1
                    #Active faults
                    activefaults=random.sample(list(FaultPos),F)
                    print("New active faults: %s"%np.sort(activefaults))
                    inactivefaults=[]
                    for position in list(FaultPos):
                        if position in activefaults:
                            vector[position]=1
                        else:
                            vector[position]=-1
                            inactivefaults.append(position)
                    energy = vector@J@vector+vector@H+C
                    print("New energy: %s"%energy)
                    FailSafe += 1
                if solnflag==L_io:
                    #break out of this "if energy==0:" statement
                    break
            if solnflag==L_io:
                #break out of FailSafe loop
                break
        if solnflag==L_io:
            #break out of Fault loop
            break
        F+=1

    if solnflag==1:
        return(1,FalsePos,first_gs)
    else:
        return(0,FalsePos,first_gs)
