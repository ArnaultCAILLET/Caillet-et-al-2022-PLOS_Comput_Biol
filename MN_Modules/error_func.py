""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------
"""
from RMS_func import RMS_func
from FF_filt_func import FF_filt_func


def error_func(Size, I, t_start, t_stop, t_plateau_end, Cm_rec, Cm_derec, step_size, ARP,  kR, FIDF_exp):
    '''
This function computes the RMS difference (error) between the experimental and
simulated time histories of filtered instantaneous discharge frequencies 
(FIDFs). The simulated FIDFs are obtained with the LIF model. The error is to
be minimized for MN size calibration. 

Parameters
----------
$ Size : MN size [m2], float
    Parameter to be calibrated to minimize the error function
$ I : current input [A], function
$ t_start : starting time [s] for the calibration step, float
$ t_stop : ending time [s] for the calibration step, float
$ t_plateau_end : ending time [s] of the exp plateau of force, array
$ Cm_rec : Specific capacitance during recruitment phase, float
$ Cm_derec : Specific capacitance during derecruitment phase, float
$ step_size : step size of the LIF solver, float
    Should be lower or equal to 10**-4
$ ARP : MN-specific ARP value [s], float
$ kR : Ith-R gain, float
    Typical gain Ith=kR*R in the literature (Caillet, 2021)
$ FIDF_exp : Time-histories of experimental FIDFs, matrix


Returns
-------
$ error : RMS difference between exp and sim FIDFs to minimize
    '''            
    FIDF_sim,  NA, NB =FF_filt_func(I, t_start, t_stop, t_plateau_end,Size, Cm_rec, Cm_derec, step_size, ARP, kR)
    error=RMS_func(FIDF_exp, FIDF_sim)
    return error