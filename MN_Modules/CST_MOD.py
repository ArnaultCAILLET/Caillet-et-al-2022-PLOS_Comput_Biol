""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

This code computes the cumulative spike train from spike trains
"""

import numpy as np

def CST_func(Nb_MN, time, disch_times,unit='sample', fs=2048):
    '''
This function computes the cumulative spike train (CST) from provided 
lists of MN discharge times.

Parameters
----------
$ Nb_MN : the number of discharging MNs involved in the CST, integer
$ time : the time-duration of the CST, array
    time typically covers the entirety of the duration of the experimental 
    data, is in [s] with a 1/fs time step
$ disch_times : the lists of the MN discharge times, matrix
    The discharge times are typically given in samples (fs=2048 Hz in all 
    datasets). 
$ unit : unit of disch_times, string
    As the CST is built in integer samples following fs and not in 
    seconds, it must be specified if disch_times is provided in samples or
    seconds, in which case it is preliminary trasnformed into samples
$ fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures. 

Returns
-------
$ Binary_matrix : matrix of the time-location of the MN discharges, matrix
    In each cell (i,t), if the ith MN discharges at time t (in samples), a 
    binary value of 1 is stored, otherwise 0. Example: if the 1st MN 
    discharges at the 4th time sample (i.e. at 4/fs [s]), 
    Binary_matrix = [[0, 0, 0, 1, ...], ...]
$ CST : Cumulative spike train, array
    Array returning at each time instant (in samples) the nb of MNs firing. 
    Example: if the 5 MNs siùultaneously discharge for the first time at 
    the 4th time sample (i.e. at 4/fs [s]), CST = [0, 0, 0, 5, ...]
    '''
    
    Binary_matrix=np.empty((Nb_MN,), dtype=object)
    for i in range (Nb_MN): 
        disch_time_full_array=np.zeros(len(time)) #initalizing the binary array for MN i with zeros everywhere
        dt_arr=disch_times[i] #dt_arr is the list of fs-time samples at which MN i discharges
        for j in range(len(dt_arr)): #looping for all the discharge time samples of MN i
            if unit=='sample':
                dt=int(dt_arr[j]) 
            elif unit=='sec':
                dt=int(dt_arr[j]*fs) #scaling seconds into samples
            disch_time_full_array[dt]=1 #populating the Binary matrix with 1s at discharge times
        Binary_matrix[i]=disch_time_full_array
    CST=Binary_matrix.sum(axis=0) #Summing the MN binary lists sample by sample. Builds the CST
    return Binary_matrix, CST