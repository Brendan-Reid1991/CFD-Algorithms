import numpy as np

# The network I was using had an older version of numpy installed, so I needed to define the flip function myself. 

def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = np.asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

# Positions of the output vector, Faults and auxiliary spins are defined below. 
# These are constructed by me, specifically for this problem. 
# These positions are essentially arbitrary; as long as you keep track of where they are you can define them however you like.

def output_pos(n):
    final=np.array([n*(9*n-8)-1])
    SStart=(7*n-6)*n
    x=np.arange(SStart+n**2-2*n+2,SStart+n**2-n+1)
    BottomRow=flip(x,0)-1
    LeftHandSide=flip(np.arange(SStart+1,SStart+n**2-2*n+1,n-1),0)-1
    First=np.array([6*n**2-6*n])
    Hpos=np.array(np.concatenate([final,BottomRow,LeftHandSide,First]))
    return(Hpos)
    
def Positions(n):
    L=24*n**2-30*n
    AllPositions=np.arange(0,L)
    InputPos=np.arange(0,2*n)
    OutputPos=output_pos(n)
    FaultPositions=np.arange(2*n,2*n+n**2+2*n+5*n*(n-2))
    FL=6*n**2-8*n
    SpinPositions=np.zeros(L-FL-4*n,dtype=int)
    it=0
    for index in AllPositions:
        if index not in np.concatenate([InputPos, OutputPos, FaultPositions]):
            SpinPositions[it]=index
            it=it+1
    return(FaultPositions,SpinPositions)
            
def Exclusion(IOPairs):
    n=len(IOPairs[0][0])
    FaultPositions=np.arange(2*n,2*n+n**2+2*n+5*n*(n-2))
    BetaPositions=np.arange(10*n**2-8*n,11*n**2-8*n)
    NuPositions=np.arange(11*n**2-6*n,11*n**2-4*n)
    LambdaPositions=np.arange(16*n**2-14*n,21*n**2-24*n)
    #########################################
    outputs=np.transpose(np.asarray(IOPairs))[2]
    OutputsReshaped=np.empty((2*n,len(IOPairs)),dtype=int)
    it1=0
    while it1<2*n:#len(IOPairs):
        it2=0
        for Z in outputs:
            OutputsReshaped[it1][it2]=Z[it1]
            it2+=1
        it1+=1     
    Out1Fault=FaultPositions[0]
    Out2Fault=FaultPositions[n**2+1]
    OutFaultFABlock=np.flip(FaultPositions[n**2+2*n+3:n**2+2*n+5*(n-2)*(n-1):5*(n-1)])
    OutFaultBtmLft=FaultPositions[n**2+2*n-1]
    OutFaultBtmRow=np.flip(FaultPositions[n**2+2*n+5*(n-2)*(n-1)+3:-1:5])
    OutLastFault=FaultPositions[-1]
    OutputFaults=np.concatenate([[OutLastFault],OutFaultBtmRow,[OutFaultBtmLft],OutFaultFABlock,[Out2Fault],[Out1Fault]])
    ExcludeMe=np.zeros(2*n,dtype=int)
    it=0
    for element in OutputsReshaped:
        if -1 in element:
            ExcludeMe[it]=OutputFaults[it]
        it+=1
    ExcludeFaults=np.setdiff1d(ExcludeMe,[0])
    #######################################
    AuxOut1=BetaPositions[0]
    AuxOut2=NuPositions[1]
    AuxOutFABlock=LambdaPositions[3:5*(n-2)*(n-1):5*(n-1)]
    AuxOutBtmLft=NuPositions[-1]
    AuxOutBtmRow=LambdaPositions[5*(n-2)*(n-1)+3:-1:5]
    AuxOutLast=LambdaPositions[-1]
    AuxOutAll=np.concatenate([[AuxOutLast],AuxOutBtmRow,[AuxOutBtmLft],AuxOutFABlock,[AuxOut2,AuxOut1]])
    it=0
    ExcludeMe2=np.zeros(2*n,dtype=int)
    for element in OutputFaults:
        if element in ExcludeFaults:
            ExcludeMe2[it]=AuxOutAll[it]
        it+=1
    ExcludeAux=np.setdiff1d(ExcludeMe2,[0])
    return(ExcludeFaults,ExcludeAux)
            
    
    
    
    
    
    
    
    
    
    
    