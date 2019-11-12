# CFD-Algorithms
Algorithms for solving circuit-fault-diagnosis problems

These files aim to solve a circuit fault diagnosis (CFD) problem via annealing techniques. 

CFD problems are conceptually simple: given a circuit C, and some inputs X,Y and an output Z, if Z != C(X, Y)
then a gate within the circuit must be faulty. The question is: how do we determine which gate is faulty? 

Naturally for small circuits it's fairly simple, however in general this problem is NP-Hard.

Using the example of binary multiplier circuits of dimension n (that is, circuits that multiply two binary strings
each of length n) we have created a quadratic unconstrained binary optimisation (QUBO) problem that can be tackled by our algorithm. 
The goal is to find a circuit-gate 'fault configuration' that explains all of the relevant input /  output data. 


A binary multiplier of dimension n contains 6n^2 - 8n gates, and for a QUBO problem this requires 24n^2 - 30n individual spins. 4n of these are inputs and outputs. The algorithms contained within search this solution space for a `fault configuration' that explains all of the input output data.


This data is generated in the Mathematica file: choosing circuit dimension, number of faults, and number of input/output 
pairs and the number of unique problems you wish to tackle creates a .dat file containing this information in the folder "IOData". File name formats are "Dx_Fy_Nz_Pt.dat" where x is the dimension of the circuit, y is the number of faults per problem, z is the number of unique problems, and t is the number of input/output pairs per problem. 

The algorithm we have developed aims to minimise the number of faults whilst maximising the fidelity to the 'true' fault configuration, stored in the .dat file. 

These types of optimisation problems generally consist of minimising a cost function, i.e. min_{s} G

                    where G = \sum_{i,j} s_i J_{i,j} s_j + \sum_k H_k s_k + C

(see, e.g., https://arxiv.org/pdf/1705.09844.pdf)

The matrices J, H and constant C are also created in the Mathematica file. These are unique to each dimension, so they only 
need generated once.

The python file CreateFiles.py takes a single integer argument, the dimension of the circuit you want to investigate. 
Both this and the Mathematica file will create a folder "Multiplier_DimX" where X is the integer. Which you run first, will create the folder. However both are required to tackle the problem. 

CreateFiles.py dips into DummyFiles to get templates for the simulated annealing (SA) and parallel tempering (PT) algorithms, and copies them into the relevant folder, inserting the integer dimension where necessary. 

in short, multiple dimensions with accompanying problem data and matrices can be generated in minutes. 

I have inserted an example, Multipler_Dim3, the smallest multiplier this problem tackles. Within here there are two solders, SA and PT, and within those are the relevant program files to run the algorithm. Each of these are explained within. 

PropagateCircuit and PositionDefinitions are modules required to run the algorithm, with the latter not reliant on the J,H, and C files such that it can be copied and pasted between multiplier dimensions without needing to be changed. 

The others are:

SimulatedAnnealing / ParallelTempering --- bare bones algorithms

'' _DebuggingFile --- as above but with printed messages

SA/PT_DataColleciton --- programs that iterate over multiple problems and multiple variables for data collection

SA/PT_JustForFun --- an example file that takes varying arguments and prints out what the algorithm is doing, for explanatory 
purposes.
