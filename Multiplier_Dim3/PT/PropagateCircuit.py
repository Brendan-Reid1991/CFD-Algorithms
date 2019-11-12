import numpy as np
from scipy.io import mmread
jpath="JMat%s.mtx"%3
hpath="HMat%s.dat"%3
cpath="const%s.dat"%3
J = mmread(jpath)
H = np.loadtxt(hpath)
C = np.loadtxt(cpath)

import math as mth

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

# Define logic gate action on spins

def AND(X):
    val=0.5*(X[0]*X[1]+X[0]+X[1]-1)
    return(val)

def OR(X):
    val=0.5*(-X[0]*X[1]+X[0]+X[1]+1)
    return(int(val))
    
def XOR(X):
    val=-(X[0]*X[1])
    return(int(val))

GateList=np.array([AND,XOR,AND,XOR,OR])

def faultpos(dim):
    return(np.arange(2*dim,2*dim+dim**2+2*dim+5*dim*(dim-2)));
    
def output_pos(n):
    final = np.array([n*(9*n-8)-1])
    SStart = (7*n-6)*n
    x = np.arange(SStart+n**2-2*n+2,SStart+n**2-n+1)
    BottomRow=flip(x,0)-1
    LeftHandSide=flip(np.arange(SStart+1,SStart+n**2-2*n+1,n-1),0)-1
    First=np.array([6*n**2-6*n])
    Hpos=np.array(np.concatenate([final,BottomRow,LeftHandSide,First]))
    return(Hpos)

#REMEMBER the "active" list should come from the Mathematica formulation of the code, 
    #where positions are indexed from 1 and not from 0. 
    #This code maps (f1,f2,...) --> (f1-1,f2-1,....)

# This function takes two inputs (x,y) and an output (z) as well as a fault configuration (active)
# It propagates the inputs through the circuit assuming the gates in (active) are faulty and only return (1)
# It measures the cost function at the end, which is only zero if the fault configuration works
# This function is used to quickly check potential fault configurations against all input output pairs for a problem.

def propagate(x,y,z,active):
    n=len(x)
    L=24*n**2-30*n
    FL=6*n**2-8*n
    SpinStart=2*n+FL
    AllPositions=np.arange(0,L)
    InputPos=np.arange(0,2*n)
    OutputPos=output_pos(n)
    FaultPositions=faultpos(n)
    HalfAdderFaultPositions=FaultPositions[n**2:n**2+2*n]
    FullAdderFaultPositions=FaultPositions[n**2+2*n::]
    SpinPositions=np.zeros(L-FL-4*n,dtype=int)
    it=0
    for index in AllPositions:
        if index not in np.concatenate([InputPos, OutputPos, FaultPositions]):
            SpinPositions[it]=index
            it=it+1
    AlphaPositions=np.arange(9*n**2-8*n,10*n**2-8*n)
    BetaPositions=np.arange(10*n**2-8*n,11*n**2-8*n)
    SPositions=np.arange(7*n**2-6*n,8*n**2-7*n)
    STopRow=np.arange(7*n**2-6*n,7*n**2-6*n+n-1)
    CPositions=np.arange(8*n**2-7*n,9*n**2-8*n)
    CTopRow=np.arange(8*n**2-7*n,8*n**2-7*n+n-1)
    MuPositions=np.arange(11*n**2-8*n,11*n**2-6*n)
    NuPositions=np.arange(11*n**2-6*n,11*n**2-4*n)
    ChiPositions=np.arange(11*n**2-4*n,16*n**2-14*n)
    LambdaPositions=np.arange(16*n**2-14*n,21*n**2-24*n)
    VarThetaPositions=np.arange(21*n**2-24*n,21*n**2-24*n+3*n*(n-2))         
    if isinstance(active,int):
        activefaults=np.array([active])-1
    else:
        activefaults=np.array(active)-1
    if len(activefaults)==0:
        FOff=len(FaultPositions)
        inactivefaults=FaultPositions
    else:
        FOn=len(activefaults)
        FOff=len(FaultPositions)-FOn
        inactivefaults=np.zeros(FOff,dtype=int)
        it=0
        for fault in FaultPositions:
            if fault not in activefaults:
                inactivefaults[it]=fault
                it+=1
    sym=np.zeros(L,dtype=int)
    sym[0:n]=x
    sym[n:2*n]=y
    sym[OutputPos]=z
    if len(activefaults)>0:
        sym[activefaults]=[1]*FOn
    sym[inactivefaults]=[-1]*FOff
    AndFaults=np.reshape(sym[2*n:2*n+n**2],(n,n))
    AndInputs=np.zeros((n,n,2),dtype=int)
    AndPositions=np.reshape(AllPositions[2*n+FL:2*n+FL+n**2],(n,n))
    j=0;
    while j<n:
        k=0;
        while k<n:
            AndInputs[j,k]=np.array([sym[n-j-1],sym[2*n-k-1]])
            k=k+1;
        j=j+1
    it=-1;
    for row in AndPositions:
        it=it+1;
        for column in row:
            ar=it;
            ac=(column-SpinStart)%n
            if ar>0 or ac>0:
                sym[column]=AND(AndInputs[ar,ac])*(1-AndFaults[ar,ac])*0.5+(1+AndFaults[ar,ac])*0.5 
    it=0;
    for element in AlphaPositions:
        sym[element]=AND(np.reshape(AndInputs,(n**2,2))[it])
        it+=1
    it=0;
    for element in BetaPositions:
        sym[element]=AND([sym[np.ndarray.flatten(AndPositions)[it]],sym[FaultPositions[it]]])
        it+=1
    HalfAdderInTopRow=np.zeros((n-1,2))
    it=0
    while it<n-1:
        HalfAdderInTopRow[it]=sym[[AndPositions[0,1::][it],(AndPositions.T)[0,1::][it]]]
        it+=1
    it=0
    while it<n-1:
        for element in MuPositions[0:2*n-2]:
            sym[element]=AND(HalfAdderInTopRow[mth.floor(it)])
            it=it+0.5        
    
    it=1
    while it<n-1:
        for element in STopRow[1::]:
            sym[element]=XOR(HalfAdderInTopRow[it])*(1-sym[HalfAdderFaultPositions[2 *it+1]])*0.5 +(1+sym[HalfAdderFaultPositions[2*it+1]])*0.5   
            it+=1
    it=0
    while it<n-1:
        for element in CTopRow:
            sym[element]=AND(HalfAdderInTopRow[it])*(1-sym[HalfAdderFaultPositions[2 *it]])*0.5 +(1+sym[HalfAdderFaultPositions[2*it]])*0.5   
            it+=1
    TopRowOutputs=np.zeros(2*n-2)
    it=0
    while it<2*n-2:
        if it%2==0:
           TopRowOutputs[it]=sym[CTopRow[int(mth.floor(it/2))]]
        else:
            TopRowOutputs[it]=sym[STopRow[int(mth.floor(it/2))]]
        it+=1 
    it=0
    while it<2*n-2:
        for element in NuPositions[0:2*n-2]:
            sym[element]=AND([TopRowOutputs[it],sym[HalfAdderFaultPositions[it]]])
            it+=1
    adder=1
    while adder<(n-1)*(n-2)+1:
        gate=1
        while gate<6:
            ##Generic inputs
            b1=int(max(np.ceil(adder/(n-1))-1,0)+1)+1
            b2=adder-max((b1-2),0)*(n-1)+1
            FirstIn=AndPositions[b1-1,b2-1]
            ThirdIn=CPositions[adder-1]
            if adder%(n-1)==0:
                SecondIn=AndPositions[adder-(n-1)*(adder//n),(adder//n)+1]
            else:
                SecondIn=SPositions[adder]
            if gate==1 or gate==2:
                OutPosition=VarThetaPositions[gate-1 + 3 *(adder-1)]
                sym[OutPosition]=GateList[gate-1]([sym[FirstIn],sym[SecondIn]])*(1-sym[FullAdderFaultPositions[gate-1+5*(adder-1)]])*0.5+(1+sym[FullAdderFaultPositions[gate-1+5*(adder-1)]])*0.5
                sym[ChiPositions[gate-1+5*(adder-1)]]=AND([sym[FirstIn],sym[SecondIn]])
                sym[LambdaPositions[gate-1+5*(adder-1)]]=AND([sym[OutPosition],sym[FullAdderFaultPositions[gate-1+5*(adder-1)]]])
            if gate==3:
                sym[VarThetaPositions[2+3*(adder-1)]]=GateList[gate-1]([sym[ThirdIn],sym[VarThetaPositions[2+3*(adder-1)-1]]])*(1-sym[FullAdderFaultPositions[2+5*(adder-1)]])*0.5+(1+sym[FullAdderFaultPositions[2+5*(adder-1)]])*0.5
                sym[ChiPositions[2+5*(adder-1)]]=AND([sym[ThirdIn],sym[VarThetaPositions[2+3*(adder-1)-1]]])
                sym[LambdaPositions[2+5*(adder-1)]]=AND([sym[VarThetaPositions[2+3*(adder-1)]],sym[FullAdderFaultPositions[2+5*(adder-1)]]])

            if gate==4:
                sym[SPositions[adder+n-2]]=GateList[gate-1]([sym[ThirdIn],sym[VarThetaPositions[3*(adder-1)+1]]])*(1-sym[FullAdderFaultPositions[3+5*(adder-1)]])*0.5+(1+sym[FullAdderFaultPositions[3+5*(adder-1)]])*0.5
                sym[ChiPositions[3+5*(adder-1)]]=AND([sym[ThirdIn],sym[VarThetaPositions[3*(adder-1)+1]]])
                sym[LambdaPositions[3+5*(adder-1)]]=AND([sym[SPositions[adder+n-2]],sym[FullAdderFaultPositions[3+5*(adder-1)]]])
            if gate==4 and adder%(n-1)==1:
                sym[ChiPositions[4+5*(adder-1)]]=AND([sym[ThirdIn],sym[VarThetaPositions[3*(adder-1)+1]]])
                sym[LambdaPositions[4+5*(adder-1)]]=AND([sym[SPositions[adder+n-2]],sym[FullAdderFaultPositions[4+5*(adder-1)]]])
            if gate==5:
                sym[CPositions[adder+n-2]]=GateList[gate-1]([sym[VarThetaPositions[3*(adder-1)]],sym[VarThetaPositions[2+3*(adder-1)]]])*(1-sym[FullAdderFaultPositions[4+5*(adder-1)]])*0.5+(1+sym[FullAdderFaultPositions[4+5*(adder-1)]])*0.5
                sym[ChiPositions[4+5*(adder-1)]]=AND([sym[VarThetaPositions[3*(adder-1)]],sym[VarThetaPositions[2+3*(adder-1)]]])
                sym[LambdaPositions[4+5*(adder-1)]]=AND([sym[CPositions[adder+n-2]],sym[FullAdderFaultPositions[4+5*(adder-1)]]])
            gate+=1
        adder+=1

    sym[CPositions[(n-1)**2]]=AND([sym[CPositions[(n-1)**2-n+1]],sym[SPositions[(n-1)**2-n+2]]])*0.5*(1-sym[HalfAdderFaultPositions[-2]])+0.5*(1+sym[HalfAdderFaultPositions[-2]])
    sym[MuPositions[-2]]=AND([sym[CPositions[(n-1)**2-n+1]],sym[SPositions[(n-1)**2-n+2]]])
    sym[NuPositions[-2]]=AND([sym[CPositions[(n-1)**2]],sym[HalfAdderFaultPositions[-2]]])
    
    sym[MuPositions[-1]]=AND([sym[CPositions[(n-1)**2-n+1]],sym[SPositions[(n-1)**2-n+2]]])
    sym[NuPositions[-1]]=AND([sym[SPositions[(n-1)**2]],sym[HalfAdderFaultPositions[-1]]])
    #Remaining Full Adders
    adder=(n-1)*(n-2)+1
    while adder<n*(n-2):
        gate=1
        while gate<6:
            ##Generic inputs
            FirstIn=CPositions[adder+n-2]
            ThirdIn=CPositions[adder]
            SecondIn=SPositions[adder+1]
            if gate==1 or gate==2:
                OutPosition=VarThetaPositions[3*(adder-1)+gate-1]
                sym[OutPosition]=GateList[gate-1]([sym[FirstIn],sym[SecondIn]])*0.5*(1-sym[FullAdderFaultPositions[5*(adder-1)+gate-1]])+(1+sym[FullAdderFaultPositions[5*(adder-1)+gate-1]])*0.5
                sym[ChiPositions[5*(adder-1)+gate-1]]=AND([sym[FirstIn],sym[SecondIn]])
                sym[LambdaPositions[5*(adder-1)+gate-1]]=AND([sym[OutPosition],sym[FullAdderFaultPositions[5*(adder-1)+gate-1]]])
            if gate==3:
                sym[VarThetaPositions[2+3*(adder-1)]]=GateList[gate-1]([sym[ThirdIn],sym[VarThetaPositions[2+3*(adder-1)-1]]])*0.5*(1-sym[FullAdderFaultPositions[5*(adder-1)+gate-1]])+(1+sym[FullAdderFaultPositions[5*(adder-1)+gate-1]])*0.5
                sym[ChiPositions[5*(adder-1)+2]]=AND([sym[ThirdIn],sym[VarThetaPositions[2+3*(adder-1)-1]]])
                sym[LambdaPositions[5*(adder-1)+2]]=AND([sym[VarThetaPositions[2+3*(adder-1)]],sym[FullAdderFaultPositions[5*(adder-1)+2]]])
            if gate==4:
                
                sym[ChiPositions[5*(adder-1)+3]]=AND([sym[VarThetaPositions[3*(adder-1)+1]],sym[ThirdIn]])
                sym[LambdaPositions[5*(adder-1)+3]]=AND([sym[SPositions[adder+n-1]],sym[FullAdderFaultPositions[5*(adder-1)+3]]])
            if gate==5:
                sym[CPositions[adder+n-1]]=OR([sym[VarThetaPositions[3*(adder-1)]],sym[VarThetaPositions[3*(adder-1)+2]]])*(1-sym[FullAdderFaultPositions[5*(adder-1)+4]])*0.5+(1+sym[FullAdderFaultPositions[5*(adder-1)+4]])*0.5
                sym[ChiPositions[5*(adder-1)+4]]=AND([sym[VarThetaPositions[3*(adder-1)]],sym[VarThetaPositions[3*(adder-1)+2]]])
                sym[LambdaPositions[5*(adder-1)+4]]=AND([sym[CPositions[adder+n-1]],sym[FullAdderFaultPositions[5*(adder-1)+4]]])
            gate+=1
        adder+=1
    #Final Adder
    FirstIn=CPositions[-2]
    SecondIn=AndPositions[-1,-1]
    ThirdIn=CPositions[-n]
    
    sym[VarThetaPositions[-3]]=AND([sym[FirstIn],sym[SecondIn]])*(1-sym[FullAdderFaultPositions[-5]])*0.5+(1+sym[FullAdderFaultPositions[-5]])*0.5
    sym[ChiPositions[-5]]=AND([sym[FirstIn],sym[SecondIn]])
    sym[LambdaPositions[-5]]=AND([sym[VarThetaPositions[-3]],sym[FullAdderFaultPositions[-5]]])
    
    sym[VarThetaPositions[-2]]=XOR([sym[FirstIn],sym[SecondIn]])*(1-sym[FullAdderFaultPositions[-4]])*0.5+(1+sym[FullAdderFaultPositions[-4]])*0.5 
    sym[ChiPositions[-4]]=AND([sym[FirstIn],sym[SecondIn]])
    sym[LambdaPositions[-4]]=AND([sym[VarThetaPositions[-2]],sym[FullAdderFaultPositions[-4]]])
      
    sym[VarThetaPositions[-1]]=AND([sym[ThirdIn],sym[VarThetaPositions[-2]]])*(1-sym[FullAdderFaultPositions[-3]])*0.5+(1+sym[FullAdderFaultPositions[-3]])*0.5 
    sym[ChiPositions[-3]]=AND([sym[ThirdIn],sym[VarThetaPositions[-2]]])
    sym[LambdaPositions[-3]]=AND([sym[VarThetaPositions[-1]],sym[FullAdderFaultPositions[-3]]])
    
    sym[ChiPositions[-2]]=AND([sym[ThirdIn],sym[VarThetaPositions[-2]]])
    sym[LambdaPositions[-2]]=AND([sym[SPositions[-1]],sym[FullAdderFaultPositions[-2]]])
    
    sym[ChiPositions[-1]]=AND([sym[VarThetaPositions[-3]],sym[VarThetaPositions[-1]]])
    sym[LambdaPositions[-1]]=AND([sym[CPositions[-1]],sym[FullAdderFaultPositions[-1]]])
    return(sym@J@sym+sym@H+C,sym)


