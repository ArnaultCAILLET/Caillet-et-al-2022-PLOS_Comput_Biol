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
import pandas as pd
import scipy.io

def EXP_DATA_PROCESSING_func(author, test):
    '''
This function
    1. loads the .mat file that stores the experimental data (MN discharge times and transducer force time histories)
    obtained from the decomposition of HDEMG signals. 
    2. orders the identified MNs in the order of incresaing force recruitment thresholds

Parameters
----------
author : name of the first author of the paper that provides the experimental data, string
test : name of the set of experimental data under study, string


Returns
-------
Nb_MN : the number of identified MNs in the experimental dataset, integer
Force : the time-history of transducer Force amplitude, array, arbitrary units
disch_times : the lists of the MN discharge times, matrix
    The discharge times are returned in samples (fs=2048 Hz in all datasets). disch_times is not a rectangle matrix.
    '''

# loading data
    path_to_data = './Input_Exp_Data/' 
    mat = scipy.io.loadmat(path_to_data + test+'.mat') 

#Extracting relevant data    
    for key, value in mat.items(): 
        if key=='MUPulses':
            disch_times_raw=np.array((value))[0]
        if key=='ref_signal':
            Force=np.array((value))[0]        
    # print(min(Force))
    Nb_MN=len(disch_times_raw) #Number of recorded MNs

# Ordering the spike trains from earliest to latest first discharge time
    disch_times_disorganised=np.empty((Nb_MN,), dtype=object) 
    first_disch=np.ones(Nb_MN) #storing the first discharge times, helping in raking the data

# first, reshaping the spike train data, and storing the first discharge times of each spike train    
    for i in range (Nb_MN): 
        disch_times_disorganised[i]=disch_times_raw[i][0].astype(object)
        first_disch[i]=disch_times_disorganised[i][0] #adding the recruitment time
    
# then, going through the array of recruitment times, and ranking each index of first_disch to sort the data
    order = first_disch.argsort()
    ranks = order.argsort()        
    disch_times=np.empty((Nb_MN,), dtype=object) 
    for i in range (Nb_MN): 
        j=np.argwhere(ranks==i)[0][0]
        disch_times[i]=disch_times_disorganised[j]        
                          
    return  Nb_MN, Force, disch_times
