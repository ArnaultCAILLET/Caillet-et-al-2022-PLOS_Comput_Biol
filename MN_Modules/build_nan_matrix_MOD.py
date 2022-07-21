""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------
"""

import numpy as np

def build_nan_matrix_func(Firing_times_sim, author, trial='exp'):
    #Building the matrix with nans
    max_length=0
    for i in range(len(Firing_times_sim)): 
        max_length = max(max_length, len(Firing_times_sim[i]))
    Firing_times_sim_nan=np.empty((len(Firing_times_sim),max_length), dtype=object) #empty list of 32 cells
    Firing_times_sim_nan[:]=np.nan
    for i in range (len(Firing_times_sim)): #looping through the MN data
        Firing_times_sim_nan[i][0:len(Firing_times_sim[i])]=Firing_times_sim[i]
    return Firing_times_sim_nan