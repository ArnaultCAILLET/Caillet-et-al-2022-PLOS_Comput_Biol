""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------

Simulating the firing activity of the complete MN pool by iteratively calling the LIF model adequately scaled for each MN
"""

import numpy as np
from RC_LIF_MOD import RC_solve_func
from FF_filt_func import FF_filt_func
from PLOTS import plot_pred_exp_FIDF_func

def run_LIF_simulation_func(Nb_MU, MN_pop, Real_MU_pop, time, t_start, t_stop, plateau_time2, FF_FULL, FF_filt_arr, Virtual_size_arr, Virtual_ARP_arr,  I, Cm_rec, Cm_derec, step_size, kR, plot, adapt_kR='n', kR_derec=1.7*10**-10, fs=2048):
    
    Firing_times_sim=np.empty((MN_pop,), dtype=object)
    RMS_table_sim=np.empty((Nb_MU,), dtype=object) #stores the RMS errors between experimental filtered FF and virtual simulated FF for the same MN
    nME_table_sim=np.empty((Nb_MU,), dtype=object) #nME errors 
    Corrcoef_table_sim=np.empty((Nb_MU,), dtype=object) #r2
    MN_list=np.arange(1,MN_pop+1,1)
    range_start=int(t_start*fs)
    range_stop=int(t_stop*fs)


    for MN in MN_list:
        if MN%10==0: print('Simulating MN n°', MN)
        
        tim_list, V, firing_times_arr, parameters=RC_solve_func(I, t_start, t_stop,plateau_time2, Virtual_size_arr[MN-1], Cm_rec, Cm_derec,  step_size, Virtual_ARP_arr[MN-1] , kR, adapt_kR, kR_derec)
        Firing_times_sim[MN-1]=firing_times_arr

    
    return nME_table_sim, RMS_table_sim, Corrcoef_table_sim, Firing_times_sim
