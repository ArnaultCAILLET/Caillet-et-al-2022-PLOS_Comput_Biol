""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code produces the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

This code plots and returns the simulation results obtained from the 1_MAIN_MN_model.py
and 2_MN_Model_Validation python codes. 
"""

import sys
sys.path.insert(0,'MN_Modules')
import numpy as np
from PLOTS import *
from CST_MOD import CST_func
from But_filter_MOD import But_filter_func
from IDF_MOD import IDF_func
from Hanning_filter_MOD import Hanning_filter_func
from pathlib import Path
root = Path(".")
path_to_data = root / "Results" 
    
#------------------------------------------------------------------------------
#Choose Trial the results of which you wish to plot
test = 'TA_35_D'
# test = 'TA_35_H'
# test = 'TA_50'
# test='GM_30'
path_to_data = path_to_data / test 
plot_FIDFs='no'
#------------------------------------------------------------------------------

# #############################################################################
# # 0. # LOADING SIMULATION RESULTS + SOME POST-PROCESSING
# #############################################################################
from load_MN_model_results_MOD import load_MN_model_results
time, muscle, MVC, Force, Nb_MN, MN_pop, Real_MN_pop, exp_disch_times, Firing_times_sim, range_start,range_stop, t_start, plateau_time1, plateau_time2, end_force, fs, Calib_sizes, [author, Cm_calib, Cm_derec_calib, adapt_kR, Cm_rec, Cm_derec, a_size, c_size, r2_size_calib, kR_derec, r2_exp, nRMSE_exp, r2_sim, nRMSE_sim], [deltaf1_table_sim, RMS_table_sim, Corrcoef_table_sim, Size_distrib_matrix, FIDF_sim_test], [calibration_onset_error,calibration_nRMSE,calibration_r2, calibration_FIDF] = load_MN_model_results(test, path_to_data)

#Computing exp and sim CSTs and CIs
Binary_matrix_exp, CST_exp=CST_func(Nb_MN, time, exp_disch_times)
common_control_exp=But_filter_func(4, CST_exp)
common_input_exp=But_filter_func(10, CST_exp)
Binary_matrix_sim, CST_sim = CST_func(MN_pop, time[range_start:range_stop], Firing_times_sim, 'sec')
common_control_sim=But_filter_func(4, CST_sim)
common_input_sim=But_filter_func(10, CST_sim)

#Displaying the experimental data (spike trains and recorded transducer force)
plot_spike_trains_force_CI_func(Nb_MN, Force, time, common_input_exp, common_control_exp,  t_start, end_force, fs, exp_disch_times, MVC)   

print('The processed dataset is ', test, '. There are '+str(Nb_MN)+' recorded MNs in this dataset.')
print('Was Cm_derec calibrated? ', Cm_derec_calib)
print('Was kR adapted for derecruitment? ', adapt_kR)
print('#-----------------------------------------------------------------------')



# #############################################################################
# # 1. # DISPLAYING OUTPUTS FROM 1_MAIN_MN_model.py
# #############################################################################
print('OUTPUTS FROM 1_MAIN_MN_model.py')
print('The sensitivity analysis (if performed) returned Cm_derec = ', Cm_derec)
print('The trendline fitted to the calibrated sizes returned a=', a_size, 'and c = ', c_size)

# Performance of the MN Sizecalibration step (Plots Calibrated sizes, onset errors, nRMSE and r2 between exp and sim FIDFs)
plot_achieved_error_S_calibration(Nb_MN,Calib_sizes, calibration_onset_error,calibration_nRMSE,calibration_r2)

# Displaying the experimental vs simulated FIDFs after Size calibration
if plot_FIDFs == 'yes':
    IDF_dt, FF_FULL = IDF_func(Nb_MN, time, exp_disch_times, t_start, end_force)
    FIDF_exp = Hanning_filter_func(Nb_MN, FF_FULL)
    try:
        for i in range(Nb_MN):
            plot_exp_vs_sim_FIDF_func(time, range_start,FIDF_exp[i],  calibration_FIDF[i], i, 'calibration')
    except: pass

#Plotting distribution of MN sizes across MN population
def size_power(x,a,c): return a*2.4**(((x+1)/MN_pop)**c)
plot_size_distribution_func(Real_MN_pop, size_power, np.array([a_size, c_size]), a_size, c_size, r2_size_calib, Calib_sizes, MN_pop)

#Plotting spike trains for completely reconstructed MN population, and FIDFs representation
plot_spike_trains_force_CI_func(MN_pop, Force, time, common_input_sim, common_control_sim,  t_start, end_force, fs, Firing_times_sim, MVC, 'sim') 
plot_onionskin_representation_func(time, Firing_times_sim, t_start, end_force, MN_pop)

print('Global validation between normalized Force trace and neural drive -->')
print('Coef of determination between EXP Force trace and Common control = '+ str(r2_exp))
print('nRMSE EXP= ' + str(nRMSE_exp))
print('Coef of determination between SIM Force trace and Common control = '+ str(r2_sim))
print('nRMSE SIM = ' + str(nRMSE_sim))
print('#-----------------------------------------------------------------------')

# Displaying eND_N and eND_Nr
plot_final_results_func(Nb_MN, time, t_start, range_start, end_force, range_stop, common_control_exp, common_control_sim, common_input_exp, common_input_sim, Force)



# #############################################################################
# # 2. # DISPLAYING OUTPUTS FROM 2_MN_Model_Validation.py
# #############################################################################
print('OUTPUTS FROM 2_MN_Model_Validation.py')

# Accuracy of the blind spike train predictions (Plots onset errors, nRMSE and r2 between exp and sim FIDFs)
plot_final_results_validation_func(Nb_MN-1, deltaf1_table_sim+0.08, 'n', RMS_table_sim, Corrcoef_table_sim, Size_distrib_matrix)

# Displaying the experimental vs blindly simulated FIDFs 
if plot_FIDFs == 'yes':
    try:
        for i in range(Nb_MN):
            plot_exp_vs_sim_FIDF_func(time, range_start,FIDF_exp[i],  FIDF_sim_test[i], i, 'validation')
    except: pass

print('Local validation between exp and sim FIDFs')
print('Average delta_ft1= '+ str(np.mean(deltaf1_table_sim[deltaf1_table_sim>-1])))
print('Average nRMSE = ' + str(np.mean(RMS_table_sim)))
print('Average r2 = '+ str(np.mean(Corrcoef_table_sim[Corrcoef_table_sim>0.4])))
print('Average power = ' + str(np.mean(Size_distrib_matrix[:,1])))
print('Average min Size = ' + str(np.mean(Size_distrib_matrix[:,0])))
print('#-----------------------------------------------------------------------')


