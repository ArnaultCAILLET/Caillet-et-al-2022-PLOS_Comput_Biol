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
from RC_LIF_MOD import RC_solve_func

def delta_ft1_noncalib_func(Nb_MN, I, t_start, t_stop, t_plateau_end, Calib_sizes, Cm_rec, Cm_derec,  step_size, ARP_table, THRESHOLDS):
    sim_first_disch_times= np.empty((Nb_MN,), dtype=object)
    for i in range (len(sim_first_disch_times)):
        sim_first_disch_times[i]=RC_solve_func(I, t_start, t_stop, t_plateau_end, Calib_sizes[i], Cm_rec, Cm_derec,  step_size, ARP_table[i])[2][0]
    delta_tf1=sim_first_disch_times-THRESHOLDS[:,0]    
    return delta_tf1
