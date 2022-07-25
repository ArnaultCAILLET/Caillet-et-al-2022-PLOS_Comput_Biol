""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code produces the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

In brief, the code follows this chronological workflow:
Experimentalata --> MN location + common input + IP identification --> S calibration (+ Cm sensitivity) --> MN pool activity

In detail, this code:
    1. Processes the available experimental data of decomposed HDEMG signals
        1.1. Loads and processes experimental data of discharge times decomposed from HDEMG signals 
        1.2. Computes the cumulative spike train ~> common input/control to the MN pool
        1.3. Relocates the identified MNs in a real MN population (using experimental & external muscle-specific literature data on force recruitment thresholds)
        1.4. Obtains the time-histories of the experimental (filtered) discharge frequencies of the identified MNs
        1.5. Obtains the distirbution of inert periods among the experimental MN pool
    2. Derives the current input to the model (using external muscle-specific literature data)
    3. Calibrates the size of the identified MNs to minimize the error in simulated vs experimental filtered discharge frequencies
    3a. The MN capacitance during derecruitment Cm_derec can be calibrated at this step
    4. Derives the distribution of MN size in the MN pool out of the calibrated sizes (fitted trendline of adequation mathematical expression)
    5. Computes and predicts the discharge behaviour of a virtual complete MN pool of the whole muscle
    6. Compares the simulated common input from the virtual MN pool to the experimental common input and to the experimental force trace. 
    7. Saves the outputs
    
Parameters
----------
test : name of the experimental data set, string
    Name of the experimental dataset to be used for the 7-steps workflow presented above   
Cm_derec_calib : Specific capacitance calibration, string
    Defines whether the specific capacitance at derecruitment should be calibrated or used as a set value at step 5. Takes the values 'yes' or 'no'
adapt_kR : R tuning at derecruitment, string
    Defines whether the value of R should be tuned at derecruitment. Takes the values 'yes' or 'no'
plot : string
    Defines whether the relevant variables derived by the workflow above should be automatically plotted. Takes the values 'y' or 'n'


Returns
-------
CST_exp : Experimental cumulative spike train, array
    Array returning at each time instant (in samples given by fs) the number of identified MNs firing
common_input_exp : Experimental common input, array
    Array returning the Butterworth [0; 10]Hz-filtered CST_exp. Yields the common input to the MN pool (Farina et al., 2015)
common_control_exp : Experimental common control, array
    Array returning the Butterworth [0; 4]Hz-filtered CST_exp. Yields the common control to the MN pool (Farina et al., 2015)
Real_MN_pop : Real MN population, array
    Array retruning the location of the identified MNs in a real MN population specific to the muscle under study. Obtained from 
    typical distributions of MU force recruitment thresholds for the muscle under study in the literature.
FIDF_exp : Experimental filtered discharge frequencies, matrix
    Matrix returning for each identified MN the filtered time-history using a moving-average Hanning window of the instantaneous discharge frequencies
ARP_table : Experimental inert periods, array
    Array returning for each identified MN the minimum time period during which no AP can be elicited. Such value is purely phenomenological, relies
    on several physiological mechanisms, some of which are not yet identified or understood, and is a pure model parameter artificially constraining the
    maximum firing frequency of the MNs. 
I_array : Current input, array
    Array returning at each time instant the value of the current input to the MN pool. The current input is a discontinuous linear scaling/ transformation 
    of the common input. The gain is obtaine from typical distributions of MN rheobase in cats and rats limb muscles in the literature.
Cm_rec : Specific capacitance, float
    If required, calibrated constant value of the MN membrane specific capacitance returning minimal RMS error and highest correlation coefficients between
    simulated (using the LIF model) and experimental FIDF for the identified MNs. If not calibrated, set as Cm_rec=1.8*10**-2 F/m2 (see Caillet et al., 2021)
Calib_sizes: Calibrated MN sizes, array
    Array returning the MN sizes of the identified MNs after a calibration step aiming to minimize the RMS error between experimental and simulated 
    (using the LIF model) FIDFs
Virtual_size_arr : MN sizes of the virtual MN population, array
    Array returning the sizes of the complete virtual MN population, using the equation of a trendline distirbution fitted to Calib_sizes
Virtual_ARP_arr : ARP values of the virtual MN population, array
    Array returning the ARPs of the complete virtual MN population, using the equation of a trendline distirbution fitted to ARP_table
Firing_times_sim : simulated firing times, matrix
    Matrix returning for each virtual MN of size in Virtual_size_arr the firing times [s] predicted by the LIF model
CST_sim : simulated Cumulative spike train
    Array returning at each time instant (in samples given by fs) the number of virtual MNs firing
common_input_sim : simulated 
common_control_sim : simulated common input, array
    Array returning the Butterworth [0; 10]Hz-filtered CST_sim. Yields the common input to the MN pool (Farina et al., 2015)
common_control_sim : Simulated common control, array
    Array returning the Butterworth [0; 4]Hz-filtered CST_sim. Yields the common control to the MN pool (Farina et al., 2015)

Note: the inert period IP is sometimes named absolute refractory period ARP in this code
"""

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0,'MN_Modules')
import numpy as np
import time as TIME
from PLOTS import *
start_code_time = TIME.time()

# ###############################################################################
# # 0 # TO BE CHOSEN BY THE USER by uncommenting
# ##############################################################################

test='TA_35_D'
# test='TA_35_H'
# test='TA_50'
# test='GM_30'

Cm_derec_calib='yes'
# Cm_derec_calib='no'

if Cm_derec_calib=='yes':
    Cm_derec_array=np.arange(1.6,2.4,0.2)*10**-2
else:
    Cm_derec=2.0*10**-2#np.mean(Cm_rec_array)*2 

adapt_kR = 'y' #y/n adapt MN resistance value during derecruitment phase
# adapt_kR = 'n'

plot='y'
# plot='n'


save='y'
# save='n'





# ###############################################################################
# # 1.1. # DATA STORING + REARRANGING FROM LOW TO HIGH THRESHOLD
# ##############################################################################
test_cases= np.array([['DelVecchio', 'TA_35_D', 'TA', 10.3,   20.5, 30,   0.35, 400, 2048],
                      ['Hug',       'TA_35_H', 'TA', 10.5,   20.5, 30,   0.35, 400, 2048],
                      ['Hug',       'TA_50',   'TA', 12,     21.8, 34.5, 0.5,  400, 2048],
                      ['Hug',       'GM_30',   'GM', 9.1,    19.1, 33.5, 0.3,  400, 2048],
                      ], dtype=object)

[author, test, muscle, plateau_time1, plateau_time2, end_force, MVC, MN_pop, fs]= test_cases[np.argwhere(test_cases==test)[0][0]]

from EXP_DATA_PROCESSING_MOD import EXP_DATA_PROCESSING_func
Nb_MN, Force, disch_times = EXP_DATA_PROCESSING_func(author, test)

from Reshaping_MOD import preprocessing_func
Force, time, MN_list, t_start, t_stop, t_stop_calib, kR, Cm_rec, step_size = preprocessing_func(author, Force, end_force, fs, Nb_MN, plateau_time1, plateau_time2)

true_MN_pop = MN_pop
MN_pool_list=np.arange(0,true_MN_pop,1)

print('The processed dataset is ', test, '. There are '+str(Nb_MN)+' recorded MNs in this dataset.')
print('Is Cm_derec calibrated? ', Cm_derec_calib)
print('Is kR adapted for derecruitment? ', adapt_kR)

print('#--------------------------------------------------------------------')



###############################################################################
# 1.2.a. #  CUMULATIVE SPIKE TRAIN CST
###############################################################################
print('Computing the Experimental CST')
from CST_MOD import CST_func
Binary_matrix_exp, CST_exp=CST_func(Nb_MN, time, disch_times)
if plot=='y': plot_CST_func(time, CST_exp, end_force)

###############################################################################
# 1.2.b. #  COMMON INPUT, CONTROL, NOISE
###############################################################################
print('Computing the Experimental Common Input')
from But_filter_MOD import But_filter_func
common_input_exp= But_filter_func(10, CST_exp)
common_control_exp= But_filter_func(4, CST_exp)
common_noise_exp=common_input_exp-common_control_exp
if plot=='y': plot_common_inputs_func(time, common_input_exp, common_control_exp, common_noise_exp, end_force)

# ##############################################################################
# 1.2.c. #  COHERENCE BETWEEN CSTs
# ##############################################################################
print('Assessing coherence between experimental CSTs')
from subset_coher_MOD import subset_coher_func
nb_tests=20
avg_all_coher=subset_coher_func(nb_tests, time, Nb_MN, Binary_matrix_exp)
print('The avg coher between 2 subsets of MNs after ',nb_tests,' trials is: ', avg_all_coher)

###############################################################################
# 1.3.a. #  FIRST & LAST DISCHARGE TIMES / CURRENT - FORCE RECRUITMENT THRESHOLDS
###############################################################################
print('Extracting experimental threshold information')
from EXP_THRESHOLDS_MOD import exp_thresholds_func
THRESHOLDS = exp_thresholds_func(Nb_MN, MVC, disch_times, common_input_exp, Force)
popt_th = plot_force_thresholds_func(THRESHOLDS[:,2],THRESHOLDS[:,5], muscle, MVC, plot, MN_pop)
kR_derec = kR/popt_th[0]

###############################################################################
# 1.3.b. #  REAL MN POPULATION
###############################################################################
print('Deriving experimental MN locations into the true MN pool')
from MN_distirbution_MOD import MN_distirbution_func
Real_MN_pop=MN_distirbution_func(Nb_MN, true_MN_pop, MN_pool_list, muscle, THRESHOLDS)
if plot=='y': plot_loc_in_real_MN_pop(MN_list, Real_MN_pop, true_MN_pop)
  
  
###############################################################################
# 1.4 #  FIDFs
###############################################################################
print('Computing experimental (filtered) instantaneous discharges frequencies')
from IDF_MOD import IDF_func
IDF_dt, FF_FULL = IDF_func(Nb_MN, time, disch_times, t_start, t_stop)
#Filtered experimental IDF
from Hanning_filter_MOD import Hanning_filter_func
FIDF_exp = Hanning_filter_func(Nb_MN, FF_FULL)
k=0 #what MN to display
if plot=='y': plot_IDF_FIDF_func(time, disch_times[k], IDF_dt[k], FF_FULL[k], FIDF_exp[k], end_force)


###############################################################################
# 1.5. #  DERIVING PARAMETER IP
###############################################################################
print('Deriving experimental distributions of inert periods')
from exp_ARP_MOD import exp_ARP_func
saturating_MN, exp_ARP=exp_ARP_func(Nb_MN, plateau_time1, plateau_time2, end_force, disch_times, IDF_dt)
from ARP_distrib_MOD import ARP_distrib_func
a_arp,b_arp,ARP_table, saturating_MN, Non_saturating_MN=ARP_distrib_func(test, Nb_MN, true_MN_pop, Real_MN_pop, saturating_MN, exp_ARP, muscle, plot, MN_pop)
if plot=='y': plot_MN_saturation_func(saturating_MN, Non_saturating_MN, Real_MN_pop, ARP_table)  


###############################################################################
# 1 - end #  PLOTIING EXPERIMENTAL DISCHARGE EVENTS, COMMON INPUT, FORCE TRACE
###############################################################################
try:
    if plot=='y':plot_spike_trains_force_CI_func(Nb_MN, Force, time, common_input_exp, common_control_exp,  t_start, end_force, fs, disch_times, MVC)   
except: pass
#------------------------------------------------------------------------------    


###############################################################################
# 2. # CURRENT INPUT I(t) 
###############################################################################
print('Computing the Experimental current input I(t)')
from Curr_input_MOD import Common_to_current_input_func#, I
G, I1=Common_to_current_input_func(THRESHOLDS, Real_MN_pop, true_MN_pop, common_input_exp, fs=2048)
def I(t): #[s] 
    if (t<THRESHOLDS[0][0]) or (t>max(THRESHOLDS[:,3])): 
        return 0
    else: 
        return I1+G*common_input_exp[int(t*fs)]

def I_smooth(t): #[s] 
    if (t<THRESHOLDS[0][0]) or (t>max(THRESHOLDS[:,3])): 
        return 0
    else: 
        return I1+G*common_control_exp[int(t*fs)]

if plot=='y': I_list =plot_current_input(time, I, end_force)  
if plot=='y': I_smooth_list = plot_current_input(time, I_smooth, end_force)      
#------------------------------------------------------------------------------  
    
print('#--------------------------------------------------------------------')




###############################################################################
# 3. # SIZE (Cm) CALIBRATION
###############################################################################
# Constraining the calibration to physiological values of MN size
from MN_properties_relationships_MOD import *
I_array= [I(time[i]) for i in range (len(time))] 
Size_min=0.08*10**-6 #lowest value of the MN surface area found in the literature
Size_max= S_Ith(min(I_smooth_list[int(plateau_time1*fs):int(plateau_time2*fs)])*10**-9) #Defining boundaries for solver
Size_start = S_Ith(10*10**-9)

print('Performing MN Size calibration')
from Simplified_Size_calibration_MOD import Size_calibration_function 
sol=Size_calibration_function(Cm_rec, time, t_start,  t_stop_calib,  plateau_time2, end_force, Size_min, Size_max, step_size, kR,  Nb_MN, FIDF_exp, ARP_table, I)
print('MN sizes are calibrated !')

Calib_sizes=sol[0] #Array of calibrated MN sizes
Calib_RMS_table = sol[2] #Array of achieved nRMSE errors
Calib_r2_table = sol[3] #Array of achieved r2
#Obtaining the error in first discharge times between exp and sim
from delta_ft1_noncalib_MOD import delta_ft1_noncalib_func
Calib_delta_tf1=delta_ft1_noncalib_func(Nb_MN, I, t_start, t_stop, plateau_time2, Calib_sizes, Cm_rec, Cm_rec,  step_size, ARP_table, THRESHOLDS)  
if plot=='y': plot_achieved_error_S_calibration(Nb_MN,Calib_sizes, Calib_delta_tf1, Calib_RMS_table, Calib_r2_table)

print('#--------------------------------------------------------------------')



###############################################################################
# 3. # (optional) Cm_derec sensitivity analysis 
###############################################################################
# AT THIS STAGE, Cm_rec is given, Cm_derec=Cm_rec. We can perform a sensitivity analysis upon Cm_derec, which is just running the LIF model multiple times 
# for varying values of Cm_derec. Size_calib and Cm_rec are unchanged. Only the value of Cm_derec is updated
if Cm_derec_calib=='yes':
    print('Sensitivity analysis on the value of Cm_derec')
    from Cm_derec_sensitivity_MOD import Cm_derec_sensitivity_func
    Cm_derec_array, RMS_normalized_table_derec, r2_mean_table_derec, Cm_derec = Cm_derec_sensitivity_func (Nb_MN, time, I, Cm_rec, Cm_derec_array, Calib_sizes, ARP_table, FIDF_exp, plateau_time1 + (plateau_time2-plateau_time1)/2, end_force, step_size, kR, adapt_kR, kR_derec, fs=2048)
    print('The Cm_derec value providing best results is Cm_derec = ', Cm_derec*100)
    if plot=='y': plot_Cm_derec_sensitivity_analysis_func(Cm_derec_array, r2_mean_table_derec, RMS_normalized_table_derec)  
else:
    print('We arbitrary set Cm_derec = ', Cm_derec*100)

#------------------------------------------------------------------------------ 
# Plotting exp vs simulated FIDFs after MN size calibration and potential sensitivity analysis upon Cm_derec
Calib_FIDF_list = np.empty((Nb_MN,), dtype=object)
if plot=='y': 
    for i in range (Nb_MN):
        Calib_FIDF_list[i] = plot_calibrated_FIDF(i, time, FIDF_exp,  I,  t_start, t_stop, plateau_time2, Calib_sizes, Cm_rec, Cm_derec, step_size, ARP_table,  kR, adapt_kR, kR_derec, fs=2048)

print('#--------------------------------------------------------------------')



###############################################################################
# 4. # SIZE BACK TO REAL POPULATION
###############################################################################
print('Reconstructing complete MN size population for the ', MN_pop, ' MNs')
from calibrated_sizes_MOD import calibrated_sizes_func
Real_MN_pop, Calib_sizes, popt, r2_size_calib= calibrated_sizes_func(Real_MN_pop,Calib_sizes.astype(float), muscle, MN_pop) 
a_size,c_size =round(popt[0],11), round(popt[1],4)

def size_power(x,a,c): return a*2.4**(((x+1)/true_MN_pop)**c)
if plot=='y': plot_size_distribution_func(Real_MN_pop, size_power, popt, a_size, c_size, r2_size_calib, Calib_sizes.astype(float), MN_pop)

print('#--------------------------------------------------------------------')



###############################################################################
# 5. # SIMULATING ACTIVITY OF COMPLETE MN POPULATION
###############################################################################
print('Final simulation of the firing activity of the ', MN_pop, ' MNs.')
range_start=int(t_start*fs)
range_stop=int(t_stop*fs)

from Virtual_pop_size_arp_MOD import Virtual_size_arp_pop
Virtual_size_arr, Virtual_ARP_arr = Virtual_size_arp_pop(MN_pop, a_size, c_size, exp_ARP, a_arp, b_arp)

from run_LIF_simulation_MOD import run_LIF_simulation_func
Sol_simulation=run_LIF_simulation_func(Nb_MN, MN_pop, Real_MN_pop, time, t_start, t_stop, plateau_time2, FF_FULL, FIDF_exp, Virtual_size_arr, Virtual_ARP_arr,  I, Cm_rec, Cm_derec, step_size, kR, plot, adapt_kR, kR_derec)
nME_sim, RMS_table_sim, Corrcoef_table_sim, Firing_times_sim = Sol_simulation[0], Sol_simulation[1], Sol_simulation[2], Sol_simulation[3]

from delta_ft1_calib_MOD import delta_ft1_calib_func
delta_tf1_end = delta_ft1_calib_func(Nb_MN, Real_MN_pop, Firing_times_sim, THRESHOLDS)

print('#--------------------------------------------------------------------')



###############################################################################
# 6. # GLOBAL VALIDATION (neural drive vs force)
###############################################################################    
Binary_matrix_sim, CST_sim = CST_func(MN_pop, time[range_start:range_stop], Firing_times_sim, 'sec')
common_control_sim=But_filter_func(4, CST_sim)
common_input_sim=But_filter_func(10, CST_sim)

if plot=='y': plot_spike_trains_force_CI_func(MN_pop, Force, time, common_input_sim, common_control_sim,  t_start, end_force, fs, Firing_times_sim,  MVC, 'sim') 
if plot=='y': plot_final_results_func(Nb_MN, time, t_start, range_start, end_force, range_stop, common_control_exp, common_control_sim, common_input_exp, common_input_sim, Force)

from RMS_func import RMS_func
r2_exp = round(np.corrcoef(common_control_exp[range_start:range_stop], Force[range_start:range_stop])[0][1]**2,2)
nRMSE_exp = RMS_func(common_control_exp[range_start:range_stop]/max(common_control_exp[range_start:range_stop]), Force[range_start:range_stop]/max(Force[range_start:range_stop]))*100
r2_sim = round(np.corrcoef(common_control_sim[range_start:range_stop], Force[range_start:range_stop])[0][1]**2,2)
nRMSE_sim = RMS_func(common_control_sim[range_start:range_stop]/max(common_control_sim[range_start:range_stop]), Force[range_start:range_stop]/max(Force[range_start:range_stop]))*100


print('Coef of determination between EXP Force trace and Common control = '+ str(r2_exp))
print('nRMSE EXP= ' + str(nRMSE_exp))
print('Coef of determination between SIM Force trace and Common control = '+ str(r2_sim))
print('nRMSE SIM = ' + str(nRMSE_sim))


print('#--------------------------------------------------------------------')




###############################################################################
# 7. # SAVING OUTPUTS
############################################################################### 
if save == 'y': 
    prefix = author+'_'+test+'_'
    np.save(prefix+'time_array', time[range_start:range_stop], allow_pickle=True) # TIME ARRAY
    np.save(prefix+'exp_force', Force[range_start:range_stop], allow_pickle=True) #EXP FORCE
    np.save(prefix+'exp_discharge_times', disch_times, allow_pickle=True) #EXP SPIKE TRAINS
    np.save(prefix+'PRED_discharge_times', Firing_times_sim, allow_pickle=True) #PREDICTED DISCH TIMES
    np.save(prefix+'parameters', [range_start,range_stop, t_start, end_force, Nb_MN, Cm_derec_calib, adapt_kR], allow_pickle=True) # PARAMETERS TAKEN FOR SIMULATION
    np.save(prefix+'MAIN_results', [Real_MN_pop, Calib_sizes, Cm_rec, Cm_derec, a_size, c_size, r2_size_calib,  kR_derec, r2_exp, nRMSE_exp, r2_sim, nRMSE_sim], allow_pickle=True) # MAIN RESULTS
    # cALIBRATION PERFORMANCE
    np.save(prefix+'calib_onset_error', Calib_delta_tf1, allow_pickle=True) 
    np.save(prefix+'calib_nRMSE', Calib_RMS_table, allow_pickle=True) 
    np.save(prefix+'calib_r2', Calib_r2_table, allow_pickle=True) 
    np.save(prefix+'calib_FIDF', Calib_FIDF_list, allow_pickle=True) 



stop_code_time=TIME.time()
print ("My program took", (stop_code_time - start_code_time)/60, "minutes to run")



