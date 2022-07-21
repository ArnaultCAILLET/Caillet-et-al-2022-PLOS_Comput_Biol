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
from RC_LIF_MOD import RC_solve_func
from Hanning_filter_MOD import Hanning_filter_func
from IDF_MOD import IDF_func


def FF_filt_func(I,  t_start, t_stop, t_plateau_end, Size, Cm_rec, Cm_derec, step_size, ARP,  kR, adapt_kR='n', kR_derec=1.7*10**-10, fs=2048):
    '''
This function runs the LIF model, outputs the simulated discharge times [s], 
and computes from them the filtered instantaneous discharge frequencies 
(FIDFs) of the MN under study subjected to the current input I. The simulated
FIDFs are used in the cost function to minimize for the calibration of the 
MN sizes. 

Parameters
----------
$ I : current input [A], function
$ t_start : starting time [s] for the calibration step, float
$ t_stop : ending time [s] for the calibration step, float
$ t_plateau_end : ending time [s] of the exp plateau of force, array
$ Size : Input MN size [m2], float
    Parameter to be then calibrated to minimize the error function
$ Cm_rec : Specific capacitance during recruitment phase, float
$ Cm_derec : Specific capacitance during derecruitment phase, float
$ step_size : step size of the LIF solver, float
    Should be lower or equal to 10**-4
$ ARP : MN-specific ARP value [s], float
$ kR : Ith-R gain, float
    Typical gain Ith=kR*R in the literature (Caillet, 2021)
$ fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures. 

Returns
-------
$ FIDF_sim : Simulated FIDFs, array
$ FF_FULL : lists of the time-location (1/fs step) of the MN IDFs, list
    Example: if the 1st MN discharges for the first time at time t=3/fs [s] at 
    IDF=12 Hz, FF_FULL = [[0, 0, 12, ...], ...]
$ disch_times : the lists of the MN discharge times [s], matrix
    '''            
    
    #RUNNING THE LIF MODEL
    t,v,disch_times, p=RC_solve_func(I, t_start, t_stop, t_plateau_end, Size, Cm_rec, Cm_derec, step_size, ARP, kR, adapt_kR, kR_derec) #f returns the discharge times in seconds
     
    time=np.arange(t_start, t_stop, fs) 

    if len(disch_times)<2: #if a MN size is input that does not return a steady firing,
        FIDF_sim=np.ones(len(time))*1 #return an aberrant solution that cannot be mistaken for a potential solution
        FF_FULL=FIDF_sim
    else:
        IDF_dt, FF_FULL=IDF_func(1, time, disch_times,t_start, t_stop, fs=2048)
        FIDF_sim=Hanning_filter_func(1, FF_FULL) #FIDFs obtained from a moving average filter of Hanning window 400 ms (DeLuca 1987)        
    
    return FIDF_sim,  FF_FULL, disch_times