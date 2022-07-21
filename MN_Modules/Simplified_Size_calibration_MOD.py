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
from error_func import error_func
from scipy.optimize import minimize_scalar
from FF_filt_func import FF_filt_func
from MN_properties_relationships_MOD import R_S_func
from PLOTS import plot_exp_pred_FIDFs_func

def Size_calibration_function(Cm_rec, time, t_start,  t_stop,  plateau_time2, end_force, Size_min, Size_max, step_size, kR,  Nb_MN, FIDF_exp, ARP_table, I, plot='y', fs=2048 ):
    '''
This function batch calibrates the size of the MNs under study to minimize the RMS
error between experimental and simulated filtereds instantaneous discharge 
frequencies (FIDFs). 

Parameters
----------
$ Cm_rec : values of Cm, float
$ time : the time-duration of the whole simulation, array
    time typically covers the entirety of the duration of the experimental 
    data, is in [s] with a 1/fs time step
$ t_start : starting time [s] for the (Cm_rec & Size) calibration step, float
$ t_stop : ending time [s] for the (Cm_rec & Size) calibration step, float
USUALLY TAKEN AS MIDDLE OF PLATEAU TO IGNORE DERECRUITMENT PHASE
$ plateau_time2 : ending time [s] of the exp plateau of force, float
$ end_force: ending time [s] of the simulation, float
$ Size_min : minimum physiological MN size, float
    Constrains the calibration. MN size = MN head surface area [m2]
$ Size_max : maximum physiological MN size, float
    Constrains the calibration. MN size = MN head surface area [m2]    
$ step_size : step size of the LIF solver, float
    Should be lower or equal to 10**-4
$ kR : Ith-R gain, float
    Typical gain Ith=kR*R in the literature (Caillet, 2021)
$ Nb_MN : the number of discharging MNs under study, integer
$ FIDF_exp : Time-histories of experimental FIDFs, matrix
$ ARP_table : absolute refractory periods [s] of identified MNs, list
$ I : current input [A], function
$ plot : plotting results 'y' / 'n', string
$ fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures. 

Returns
-------
$ Calib_sizes_final : Calibrated MN sizes [m2], list
    List of the calibrated MN sizes of the identified MNs that provides the 
    minimum RMS error between experimental and LIF-simulated FIDFs between
    t_start and t_stop. Obtained with a Cm from sensitivity analysis or with
    Cm=2.0*10**-2 F/m2
$ R_table : Corresponding MN resistance values, list
$ RMS_table : RMS error between exp vs sim FIDFs per MN, list
$ Corrcoef_table : Correlation coeff between exp vs sim FIDFs per MN, list
If Sensitivity analysis upon Cm:
$ Size_table : Calibrated sizes, matrix
    Lists of Calibrated MN sizes for each tested Cm value
$ RMS_avg_table : Average RMS errors across MNs per tested Cm value, list
$ Corcoeff_avg_table : Average corr coef across MNs per tested Cm value, list
$ Cm_rec : Specific capacitance value returning lowest RMS error and corr coef,
            float
    '''        
    
    #INITIALIZATION
    range_start=int(t_start*fs) #restricting the time range to a specific user-chosen time range for calibration
    range_stop=int(t_stop*fs)
    RMS_table=np.empty((Nb_MN,), dtype=object)
    Corrcoef_table=np.empty((Nb_MN,), dtype=object)
    Calib_sizes_final=np.empty((Nb_MN,), dtype=object)


#------------------------------------------------------------------------------
       
    for i in range (Nb_MN):
        print('Calibrating size of MN n°',i)
        # EXPERIMENTAL
        FF_filt_exp=FIDF_exp[i][range_start:range_stop]   # experimental FIDF time history for that MN under study  
        ARP=ARP_table[i] #experimental ARP value [s] for that MN under study
        #CALIBRATING the MN size to minimize RMS(FIDF_exp-FIDF_sim), tolerance 10**-9
        Size_calib=minimize_scalar(error_func, bracket=(Size_min, Size_max), bounds=(Size_min, Size_max), args=(I, t_start, t_stop, plateau_time2, Cm_rec, Cm_rec, step_size, ARP, kR, FF_filt_exp), method='bounded', options={'xatol':1e-9})            
        # PROCESSING
        FIDF_sim, NB, NC= FF_filt_func(I,  t_start, t_stop, plateau_time2, Size_calib.x, Cm_rec, Cm_rec, step_size, ARP, kR) # simulated FIDF time history for the calibrated MN size just obtained

        if len(FIDF_sim)>1:
            RMS_table[i]=Size_calib.fun/max(FF_filt_exp)*100 # RMS error returned for best calibration
            Corrcoef_table[i]=np.corrcoef(FIDF_sim, FF_filt_exp)[0][1]**2 # Correlation of determination returned for best calibration
            Calib_sizes_final[i]=Size_calib.x # Adding the newly calibrated MN size to the list of calibrated MN sizes
        else:
            RMS_table[i]=RMS_table[i-1] # RMS error returned for best calibration
            Corrcoef_table[i]=Corrcoef_table[i-1] # Correlation of determination returned for best calibration
            Calib_sizes_final[i]=Calib_sizes_final[i-1] # Adding the newly calibrated MN size to the list of calibrated MN sizes  
 
        if plot=='y': plot_exp_pred_FIDFs_func(time,range_start, range_stop,  FF_filt_exp, I,  t_start, t_stop, plateau_time2, Size_calib.x, Cm_rec, Cm_rec, step_size, ARP, i, kR)

   
# RETURNING OUTPUTS 

    R_table=R_S_func(Calib_sizes_final, kR) #Typical Ith-R relationship (Caillet, 2021)
    return Calib_sizes_final, R_table, RMS_table, Corrcoef_table