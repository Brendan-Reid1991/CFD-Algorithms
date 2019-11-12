import PropagateCircuit as prop
import PositionDefinitions as pos
import numpy as np
import json
from scipy.io import mmread
import random
import multiprocessing as mp
spinpath = "spinweights%s.dat"%3
swfile = open(spinpath)
SpinWeights = json.load(swfile)
jpath = "JMat%s.mtx"%3
hpath = "HMat%s.dat"%3
cpath = "const%s.dat"%3
J = mmread(jpath)
H = np.loadtxt(hpath)
C = np.loadtxt(cpath)

#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# The Many_Reps function is used in the parallelisation. It is designed to 
#   take a single spin vector and perform a metropolis sweep over its
#   fault and auxiliary spins. As such, for m replicas we can perform
#   m metropolis sweeps in parallel (i.e. many-replicas)
#
# It's arguments are:
# n : the dimension of the multiplier circuit
# vector : the input spin vector
# F : the fault number present
# Temp : the inverse temperature assigned to the replica (decided in the main function) 
#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def Many_Reps(n,vector,F,Temp):
    # To perform the metropolis sweep(s) we need to know which faults are active and which are not, 
    # so we require the positions of each gate fault spin to create the active and inactive lists
    
    # Fault and (auxiliary) spin positions defined externally for dimension n
    (FaultPositions,SpinPositions)  =  pos.Positions(n)

    active = []
    inactive = []
    for value in FaultPositions:
        if vector[value]==1:
            active.append(value)
        else:
            inactive.append(value)
    # These are constants for a multiplier circuit of dimension n. 
    # FlipFaultIf is used to decide whether a fault spin should be flipped, as this requires 
    # a separate process from an auxiliary spin but they should be flipped a proportional
    # amount of times

    FaultLength = 6*n**2-8*n
    FlipMax = 24*n**2-30*n-4*n
    FlipFaultIf = FaultLength/FlipMax
    
    # Number of metropolis sweeps to perform
    Sweeps = 10
    
    # Boltzmann lookup table being generated
    EnergyChange = range(1,101)
    Btz = {}
    for delta in EnergyChange:
        Btz[delta] = np.exp(-delta * Temp)
    
    #Cost value calculated    
    cost = vector@J@vector+vector@H+C
    
    sweep = 0#sweep counter
    
    while sweep<Sweeps:
        # Aux spin positions randomised each time and then iterated through
        # Fault positions will be randomly chosen

        SpinFlips = np.random.permutation(SpinPositions)
        spinchoice = 0
        flip = 0
        while flip<FlipMax: 
#             This if statement determines whether or not a fault spin is chosen 
#            to be flipped or not
            if np.random.uniform()<FlipFaultIf:
                
                # If a fault spin is chosen, we need to turn one bit off and one bit on,
                # This maintains the fault cardinality of the system

                # Variables (on,off) are indices rather than actual fault positions,
                # If the spin flip is accepted it is more efficient to alter list positions rather than elements

                on = random.choice(range(0,F))
                off = random.choice(range(0,FaultLength-F))

                vector[int(active[on])] = - vector[int(active[on])]
                vector[int(inactive[off])] = - vector[int(inactive[off])]
                
                # These calculate the energy change if we exchange the sign of the two fault spins
                Sum1 = 0
                for element in SpinWeights[int(active[on])][1][0:-1]:
                    Sum1 = Sum1 + (vector[element[0]-1]*element[1]*vector[int(active[on])])
                Sum2 = 0
                for element in SpinWeights[int(inactive[off])][1][0:-1]:
                    Sum2 = Sum2+(vector[element[0]-1]*element[1]*vector[int(inactive[off])])
                    
                Delta1 = vector[int(active[on])]*SpinWeights[int(active[on])][1][-1] + Sum1
                Delta2 = vector[int(inactive[off])]*SpinWeights[int(inactive[off])][1][-1] + Sum2
                
                # If the energy change is negative**, i.e. energy decreases, we accept the flip and exchange list indices
                # **or it increases but the Boltzmann acceptability criteria is satisfied

                # If at any point the cost function reaches zero, we have reached a minimum and it needs to be checked,
                # At this point the sweep process halts

                if Delta1+Delta2<=0:
                    temp_var = active[on]
                    active[on] = inactive[off]
                    inactive[off] = temp_var
                    #Active and inactive indices switched, cost altered
                    cost = cost+Delta1+Delta2
                    if cost == 0:
                        break
                elif np.random.uniform(0,1)<Btz[Delta1+Delta2]:
                    temp_var = active[on]
                    active[on] = inactive[off]
                    inactive[off] = temp_var
                    #Active and inactive indices switched, cost altered
                    cost = cost+Delta1+Delta2
                    if cost == 0:
                        break
                else:
                    # If the energy increases and the Boltzmann acceptability criteria is NOT satisfied,
                    # The spin signs are flipped back; lists are not amended; cost is not changed

                    vector[int(active[on])] = -vector[int(active[on])]
                    vector[int(inactive[off])] = -vector[int(inactive[off])]
            else:
                #Chosen aux spin is done sequentially, spinchoice variable increased by reset at the start of every sweep
                spin = SpinFlips[spinchoice]
                spinchoice = (spinchoice+1)%len(SpinPositions)
                vector[spin] = -vector[spin]
                
                # Spin sign changed and energy change calculated
                # Same criteria as with fault spin
                # Except no lists needed amended!

                S = 0
                for element in SpinWeights[spin][1][0:-1]:
                    S = S + (vector[element[0]-1]*element[1]*vector[spin])
                delta = S + vector[spin]*SpinWeights[spin][1][-1]
                
                if delta<= 0:
                    cost = cost + delta
                    if cost == 0:
                        break
                elif np.random.uniform()<Btz[delta]:
                    cost = cost+delta
                    if cost == 0:
                        break
                else:
                    vector[spin] = -vector[spin]
                    
            if cost == 0:
                break
            flip+= 1
        if cost == 0:
            break
        sweep+= 1
    # Function returns the cost, the current active faults, the total spin vector and the sweep.
    # Sweep number lets us know at what stage a minima was reached, if at all. 
    return(cost,active,vector,sweep)





#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 
# Function below is the full parallel tempering algorithm, incorporating the Many_Reps function.
# It's arguments are:
#
# data : two dimensional list, the first element should be the correct fault configuration
#        the second element should be another list, an N * 3 matrix with N the number of I/O pairs, 
#            and the columns correspond to input_1, input_2, and output vectors
#
# initial_faults : initial (integer) number of faults to consider
#
# Replicas : integer number of replicas to consider in the parallel tempering
#
# Swaps : integer number of swaps to consider. More swaps means higher chance of finding minima
#   
#TempRange : inverse temperature range, should be written as [beta_i,beta_f]
#  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def ParTemp(data, initial_faults, Replicas, Swaps, TempRange):

    # Input / Output pairs defined, faults.
    iop = data[1]
    faults = data[0]
    #3 of circuit defined
    n = len(iop[0][0])

    (FaultPositions,SpinPositions) = pos.Positions(n)
    
    RepList = range(0,Replicas)
    
    #Positions of the output spins in the spin vector
    output_pos = pos.output_pos(n)
    
    MaxF = len(faults)
    
    F = initial_faults
    length = 24*n**2-30*n

    #Temperatures evenly spaced 
    Temperatures = np.linspace(TempRange[0],TempRange[1],Replicas)

    #First pair to check randomly chosen.

    p = np.random.randint(0,len(iop))
    remainingpairs = np.setdiff1d(range(0,len(iop)),p)
    

    # The 'vector' variable will become a template. 
    # It is loaded with the inputs and outputs first, ensuring each replica has the same input/output pair
    # Spin placement is important here, which is why we use an empty vector rather than a list
    vector = np.zeros(length)
    for x in range(0,2*n):
        if x<n:
            vector[x] = iop[p][0][x]
        else:
            vector[x] = iop[p][1][x-n]
    it = 0
    for z in output_pos:
        vector[z] = iop[p][2][it]
        it+= 1
    
    #"Active" will be a dim(Replicas*Faults) matrix, where each replica has its own randomised active faults.
    Active = np.zeros((Replicas,F))
    i = 0
    while i<Replicas:
        Active[i] = random.sample(range(2*n,6*n**2-6*n),F)
        i += 1

    #'V' is the matrix of vectors, dim(Replicas*length)
    i = 0
    #Initially the vectors containing only the i/o pair is put into V
    V = np.zeros((Replicas,length))
    while i<Replicas:
        V[i] = vector
        i += 1
    
    # Then the relevant active faults are put in and everything else is randomised for each replica
    i = 0
    while i<Replicas:
        for element in FaultPositions:
            if element in Active[i]:
                V[i][element] = 1
            else:
                V[i][element] = -1
        for element in SpinPositions:
            V[i][element] = np.random.choice([1,-1])
        i+= 1
    
    #Energies calculated
    i = 0
    Energies = np.zeros(Replicas)
    while i<Replicas:
        cost = V[i]@J@V[i]+V[i]@H+C
        Energies[i] = (cost)
        i+= 1
    # The FailSafe variable prevents an infinite loop, and only comes into play if a local minima is found
    FailSafe = 0
    FailSafeMax = 2*len(iop)
    
    FalsePos = []
    first_gs = []

    # FalsePos is a data collecting entity. 
    # Records how many pairs are satisfied. 
    # E.g., FalsePos will look like: 
    # FalsePos = [3, 1, 1, 4, 5] so 5 solutions were found before the program quit: satisfying, respectively, 3 pairs, 1 pair, 1 pair and 4 and 5 pairs. 
    
    # first_gs will record when the first groundstate was found, per fault. Will return zero if none found.
    while F<=MaxF:
        if F>initial_faults:
            Active = np.zeros((Replicas,F))
            i = 0
            while i<Replicas:
                Active[i] = random.sample(range(2*n,6*n**2-6*n),F)
                i += 1
            i = 0
            while i<Replicas:
                for element in FaultPositions:
                    if element in Active[i]:
                        V[i][element] = 1
                    else:
                        V[i][element] = -1
                i+=1
            i = 0
            while i<Replicas:
                cost = V[i]@J@V[i]+V[i]@H+C
                Energies[i] = (cost)
                i+= 1
        while FailSafe<FailSafeMax:
            solnflag = 0
            # Swap counter set to zero. 
            # Falseflag counter set to zero, used to detect local groundstates and do appropriate action
            swapnum = 0
            falseflag = 0
            while swapnum<Swaps:
                #Parallel processing here for the Many_Reps function. 
                pool  =  mp.Pool(mp.cpu_count())
                results = pool.starmap(Many_Reps, [(n,V[M],F,Temperatures[M]) for M in RepList])
                pool.close()
                
                # New energy values, active faults and spin vectors collected. 
                # If no ground state was found sweep value will always be 10, 
                # so this is only useful for telling us WHEN a ground state was found, nothing else.
                Energies = np.asarray([i[0] for i in results])
                Active = np.asarray([i[1] for i in results])
                V = np.asarray([i[2] for i in results])
                Sweeps = [i[3] for i in results]
                #minimum and it's position found. If zero, ground state found on that replica!
                val, idx  =  min((val, idx) for (idx, val) in enumerate(Energies))
                if int(val) == 0:
                    #solnflag indicates how many pairs the fault configuration satisfies; at least 1 in the beginning
                    solnflag = 1
                    M = idx
                    s = Sweeps[M]

                    if FailSafe == 0:
                        # Remember, we only want the first gs.
                        first_gs.append([F,((s+1)*(M+1)*(swapnum+1))/(Replicas*10*Swaps)])
                    #All remainingpairs are checked to see if the fault config satisfies them
                    for p in remainingpairs:
                        enew = prop.propagate(iop[p][0],iop[p][1],iop[p][2],np.asarray(Active[M]+1,dtype = int))[0]
                        if enew == 0:
                            # solnflag increases if the configuration works. 
                            # If it works for all pairs, it breaks out of this loop. 
                            solnflag+= 1
                            if solnflag == len(iop):
                                break
                    if solnflag<len(iop):
                        # If the configuration does *not* work for all pairs, we need to restart.
                        FalsePos.append(solnflag) # Number of pairs satisfied is recorded
                        solnflag = 0 # flag reset
                        falseflag = 1 # falseflag turned on, useful later
                        
                        # New i/o pair chosen. New active faults chosen.
                        # Auxiliary spins in each spin vector are *not* changed
                        M = 0
                        p = np.random.choice(remainingpairs)
                        remainingpairs = np.setdiff1d(range(0,len(iop)),p)
                        Active = np.zeros((Replicas,F))
                        i = 0
                        while i<Replicas:
                            Active[i] = random.sample(range(2*n,6*n**2-6*n),F)
                            i+= 1

                        while M<Replicas:
                            for x in range(0,2*n):
                                if x<n:
                                    V[M][x] = iop[p][0][x]
                                else:
                                    V[M][x] = iop[p][1][x-n]
                            it = 0
                            for z in output_pos:
                                V[M][z] = iop[p][2][it]
                                it+= 1
                            for f in FaultPositions:
                                if f in Active[M]:
                                    V[M][f] = 1
                                else:
                                    V[M][f] = -1
                            M+= 1
                        
                        # New energies calculated
                        i = 0
                        Energies = np.zeros(Replicas)
                        while i<Replicas:
                            cost = V[i]@J@V[i]+V[i]@H+C
                            Energies[i] = (cost)
                            i+= 1
                        # FailSafe variable increased
                        FailSafe+= 1
                        break
                if solnflag == len(iop):
                    # If a solution has been found, this triggers. Solution recorded in FalsePos that all pairs satisfied.
                    FalsePos.append(solnflag)
                    break
                if falseflag>0:
                    # If the falseflag variable is on, this makes sure we don't perform any more swaps. 
                    # We break out of the swap While loop and continue within the FailSafe While loop. 
                    # This ensures the parallel tempering restarts entirely. 
                    break
                # If no solution was found, we perform a swap step.
                # Replicas are swapped, depending on the acceptability criteria, so they continue under a different inverse temperature
                M = 0
                while M<Replicas-1:
                    swapprob = min(1,np.exp((Temperatures[M]-Temperatures[M+1])*(Energies[M]-Energies[M+1])))
                    if np.random.uniform()<swapprob:
                        V[[M,M+1]] = V[[M+1,M]]
                        Energies[[M,M+1]] = Energies[[M+1,M]]
                        Active[[M,M+1]] = Active[[M+1,M]]
                    M += 1
                swapnum+=1
            if swapnum==Swaps:
                break
        if solnflag == len(iop):
            # If solution found; break out of this While
            break
        F+=1
    
    if solnflag == len(iop):
        return(1,FalsePos,first_gs)
    else:
        return(0,FalsePos,first_gs)