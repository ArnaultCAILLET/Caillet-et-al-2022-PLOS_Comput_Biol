""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------
"""
import numpy as np
import random
from scipy import signal

def subset_coher_func(nb_tests, time, Nb_MN, Binary_matrix, fs=2048 ):
    '''
This function derives the average coherence in [1; 10Hz] between the CSTs 
aross a user-defined number of random subsets of the identified MNs. 
This can predict, using additional results from (Negro, 2016), the amount of 
coherence between the CST obtained with the identified MNs and the one that 
would be obtained if the data of all the MNs of the MN pool were available. 

Parameters
----------
$ nb_tests : number of tests, float
    Defines the number of randomly formed subsets of the same number of but
    different MNs between which the coherence is computed
    Example: in a set of 4 MNs, if nb_tests=2, the MN pairs {(1;2);(3;4)} and
    {(1;3);(2;4)} can be obtained randomly
$ time : the time-duration of the simulation, array
    time typically covers the entirety of the duration of the experimental 
    data, is in [s] with a 1/fs time step
$ Nb_MN : the number of discharging MNs under study, integer
$ Binary_matrix : matrix of the time-location of the MN discharges, matrix
   In each cell (i,t), if the ith MN discharges at time t (in samples), a 
   binary value of 1 is stored, otherwise 0. Example: if the 1st MN 
   discharges at the 4th time sample (i.e. at 4/fs [s]), 
   Binary_matrix = [[0, 0, 0, 1, ...], ...]
$ fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures. 
    
Returns
-------
$ avg_all_coher : average coherence between CSTs across MN subsets in 
[1; 10Hz], array
    '''   
    
    MN_list=np.arange(0,Nb_MN,1)
    avg_coher=np.ones(nb_tests) #to store the average coherence calculated for each test and averaged over [1; 10Hz]

    for k in (np.arange(0,nb_tests,1)):
        CST1=np.zeros(len(time)) #CST of the 1st subset
        CST2=np.zeros(len(time))
        set_1=np.array([random.sample(range(Nb_MN), Nb_MN//2)])[0] #1st random subset
        set_2=np.setdiff1d(MN_list, set_1) #2nd random subset
        #appropriately redistributing the Binary matrix among the two subsets
        for j in range(len(set_1)):
            CST1=CST1+Binary_matrix[set_1[j]]
        for j in range(len(set_2)):
            CST2=CST2+Binary_matrix[set_2[j]]
        f, Cxy = signal.coherence(CST1, CST2, fs, nperseg=fs*2, noverlap=fs) #1s non overlapping window as in (Negro 2016)
        avg_coher[k]=np.mean(Cxy[2:21]) # averaging over [1; 10Hz]

    avg_all_coher=np.average(avg_coher)  # averaging results across the number of random tests
    return avg_all_coher