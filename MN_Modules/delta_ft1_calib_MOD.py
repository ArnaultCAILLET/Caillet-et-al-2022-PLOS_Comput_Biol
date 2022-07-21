""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------

This code computes the error in predicting the first firing time
"""
import numpy as np

def delta_ft1_calib_func(Nb_MN, Real_MN_pop, Firing_times_sim, THRESHOLDS): 
    delta_ft1_end= np.empty((Nb_MN,), dtype=object)
    for i in range (len(delta_ft1_end)):
        if len(Firing_times_sim[Real_MN_pop[i]])>2:
            delta_ft1_end[i]=Firing_times_sim[Real_MN_pop[i]][0]
        else:
            delta_ft1_end[i]=0
    delta_ft1_end=delta_ft1_end-THRESHOLDS[:,0]
    return delta_ft1_end