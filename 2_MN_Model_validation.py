""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code produces the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

This code loops through the Nr experimental spike trains. At each iteration, one spike train 
is removed (test set). The Nr-1 spike trains (training set) are used to
- define the current input
- calibrate the Nr MN sizes
- reconstruct the distribution of MN sizes in the MN pool
- Predict the MN size of the test MN
- predict the psike train of the test MN
- validate pred vs exp FIDFs of the test MN with delta_ft1, nRMSE, r2

The MAIN_MN_model.py code should be run first 
- to perform the sensitivity analysis on Cm_derec
- to have the validation of the predicted neural drive
"""

import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0,'Modules')
import numpy as np
import time as TIME
from PLOTS import *
from RC_LIF_MOD import RC_solve_func
from RMS_func import RMS_func
start_code_time = TIME.time()

# ###############################################################################
# # 0 # TO BE CHOSEN BY THE USER by uncommenting
# ##############################################################################
test='TA_35_D'
# test='TA_35_H'
# test='TA_50'
# test='GM_30'

Cm_derec=2.0*10**-2 #obtained from sensitivity analysis performed by 1_MAIN_MN_model.py

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
                      ['Hug',       'GM_30',   'GM', 9.1,    19.1, 33.5, 0.3,  550, 2048],
                      ['Hug',       'TA_50',   'TA', 12,     21.8, 34.5, 0.5,  400, 2048],
                      ], dtype=object)
[author, test, muscle, plateau_time1, plateau_time2, end_force, MVC, MN_pop, fs]= test_cases[np.argwhere(test_cases==test)[0][0]]

from EXP_DATA_PROCESSING_MOD import EXP_DATA_PROCESSING_func
Nb_MN, Force, disch_times_complete = EXP_DATA_PROCESSING_func(author, test)

print('The processed dataset is ', test, '. There are '+str(Nb_MN)+' recorded MNs in this dataset.')

deltaf1_table_sim=np.empty((Nb_MN,), dtype=object) #stores the onset errors between experimental filtered FF and virtual simulated FF for the same MN
RMS_table_sim=np.empty((Nb_MN,), dtype=object) #stores the RMS errors 
nME_table_sim=np.empty((Nb_MN,), dtype=object) #nME errors 
Corrcoef_table_sim=np.empty((Nb_MN,), dtype=object) #r2
Size_distrib_matrix=np.zeros((Nb_MN,2)) #Calibrated sizes
FIDF_sim_test_array=np.empty((Nb_MN,), dtype=object) #LIF_-simulated FIDFs

print('#--------------------------------------------------------------------')

for i in range (Nb_MN):
    print('Let us validate the spike train of the identified MN n°', str(i+1))
    print('Preprocessing experimental training data.....')

    ###############################################################################
    # 1.2.0. #  CREATING TRAINING AND TEST SETS
    ###############################################################################    
    if i==0: Nb_MN=Nb_MN-1 #each training set has Nr-1 spike trains
    disch_times_test=disch_times_complete[i]
    disch_times=np.delete(disch_times_complete, i, 0)

    from Reshaping_MOD import preprocessing_func
    Force, time, MN_list, t_start, t_stop, t_stop_calib, kR, Cm_rec, step_size = preprocessing_func(author, Force, end_force, fs, Nb_MN, plateau_time1, plateau_time2)
    MN_pool_list=np.arange(0,MN_pop,1) 
    t_stop_calib = (plateau_time1+plateau_time2)/2
    ###############################################################################
    # 1.2.a. #  CUMULATIVE SPIKE TRAIN CST
    ###############################################################################
    from CST_MOD import CST_func
    Binary_matrix_exp, CST_exp=CST_func(Nb_MN, time, disch_times)
    
    ###############################################################################
    # 1.2.b. #  COMMON INPUT, CONTROL, NOISE
    ###############################################################################
    from But_filter_MOD import But_filter_func
    common_input_exp= But_filter_func(10, CST_exp)
    common_control_exp= But_filter_func(4, CST_exp)
    common_noise_exp=common_input_exp-common_control_exp
    
    ###############################################################################
    # 1.3.a. #  FIRST & LAST DISCHARGE TIMES / CURRENT - FORCE RECRUITMENT THRESHOLDS
    ###############################################################################
    from EXP_THRESHOLDS_MOD import exp_thresholds_func
    THRESHOLDS = exp_thresholds_func(Nb_MN, MVC, disch_times, common_input_exp, Force)
    THRESHOLDS_test = exp_thresholds_func(1, MVC, disch_times_test, common_input_exp, Force)
    popt_th = plot_force_thresholds_func(THRESHOLDS[:,2],THRESHOLDS[:,5], muscle,MVC, 'n', MN_pop)
    kR_derec = kR/popt_th[0]
    
    ###############################################################################
    # 1.3.b. #  REAL MN POPULATION
    ###############################################################################
    from MN_distirbution_MOD import MN_distirbution_func
    Real_MN_pop=MN_distirbution_func(Nb_MN, MN_pop, MN_pool_list, muscle, THRESHOLDS)
    Real_MN_pop_test=MN_distirbution_func(1, MN_pop,MN_pool_list, muscle, THRESHOLDS_test)
      
    ###############################################################################
    # 1.4 #  FIDFs
    ###############################################################################
    from IDF_MOD import IDF_func
    IDF_dt, FF_FULL = IDF_func(Nb_MN, time, disch_times, t_start, t_stop)
    #Filtered experimental IDF
    from Hanning_filter_MOD import Hanning_filter_func
    FIDF_exp = Hanning_filter_func(Nb_MN, FF_FULL)

    k=disch_times_test    
    IDF_dt_exp_test, FF_FULL_exp_test = IDF_func(1, time, k, t_start, t_stop)
    FIDF_exp_test = Hanning_filter_func(1, FF_FULL_exp_test)    
    
    ###############################################################################
    # 1.5. #  DERIVING ARP
    ###############################################################################
    from exp_ARP_MOD import exp_ARP_func
    saturating_MN, exp_ARP=exp_ARP_func(Nb_MN, plateau_time1, plateau_time2, end_force, disch_times, IDF_dt)
    from ARP_distrib_MOD import ARP_distrib_func
    a_arp,b_arp,ARP_table, saturating_MN, Non_saturating_MN=ARP_distrib_func(test, Nb_MN, MN_pop, Real_MN_pop, saturating_MN, exp_ARP, muscle, 'n', MN_pop)
    if plot=='y': plot_MN_saturation_func(saturating_MN, Non_saturating_MN, Real_MN_pop, ARP_table)  
    
    ###############################################################################
    # 2. # CURRENT INPUT I(t) 
    ###############################################################################
    from Curr_input_MOD import Common_to_current_input_func#, I
    G, I1=Common_to_current_input_func(THRESHOLDS, Real_MN_pop, MN_pop, common_input_exp, fs=2048)
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

    I_smooth_list = plot_current_input(time, I_smooth, end_force) 
    #------------------------------------------------------------------------------  

    ###############################################################################
    # 3. # SIZE (Cm) CALIBRATION
    ###############################################################################
    print('Performing Size calibration....')
    # Constraining the calibration to physiological values of MN size
    from MN_properties_relationships_MOD import *
    I_array= [I(time[i]) for i in range (len(time))] 
    Size_min=0.08*10**-6 #lowest value of the MN surface area found in the literature
    Size_max= S_Ith(min(I_smooth_list[int(plateau_time1*fs):int(plateau_time2*fs)])*10**-9)
    Size_start = S_Ith(10*10**-9)

    from Simplified_Size_calibration_MOD import Size_calibration_function #the calibration runs from t_start to t_stop_calib (end plateau)
    sol=Size_calibration_function(Cm_rec, time, t_start,  t_stop_calib,  plateau_time2, end_force, Size_min, Size_max, step_size, kR,  Nb_MN, FIDF_exp, ARP_table, I, 'n')
    Calib_sizes=sol[0]
    print('MN sizes are calibrated !')
    #------------------------------------------------------------------------------  

    ###############################################################################
    # 4. # SIZE BACK TO REAL POPULATION
    ###############################################################################
    print('Reconstructing complete MN size population...')
    from calibrated_sizes_MOD import calibrated_sizes_func
    Real_MN_pop, Calib_sizes, popt, r2= calibrated_sizes_func(Real_MN_pop,Calib_sizes, muscle, MN_pop) 
    a_size,c_size =round(popt[0],11), round(popt[1],4)
    
    def size_power(x,a,c): return a*2.4**(((x+1)/MN_pop)**c)
    if plot=='y': plot_size_distribution_func(Real_MN_pop, size_power, popt, a_size, c_size, r2, Calib_sizes, MN_pop)
    #------------------------------------------------------------------------------ 
    
    ###############################################################################
    # 5. # SIMULATING TEST MN
    ###############################################################################
    print('Simulating the firing activity of the identified MN n°', str(i+1),'...')
    range_start=int(t_start*fs)
    range_stop=int(t_stop*fs)
    
    #Obtaining ARP and Size of the test MN from the distirbutions obtained with training set
    from Virtual_pop_size_arp_MOD import Virtual_size_arp_pop
    Virtual_size_arr, Virtual_ARP_arr = Virtual_size_arp_pop(MN_pop, a_size, c_size, exp_ARP, a_arp, b_arp)
    Virtual_size_arr_test, Virtual_ARP_arr_test=Virtual_size_arr[Real_MN_pop_test], Virtual_ARP_arr[Real_MN_pop_test]
    
    #Simulating the firing activity of the test MN + FIDFs
    tim_list_test, V_test, firing_times_sim_test, parameters_test=RC_solve_func(I, t_start, t_stop,plateau_time2, Virtual_size_arr_test[0], Cm_rec, Cm_derec,  step_size, Virtual_ARP_arr_test[0] , kR, adapt_kR, kR_derec, )
    IDF_dt_sim_test, FF_FULL_sim_test=IDF_func(1, time, firing_times_sim_test,t_start, t_stop, fs=2048)
    FIDF_sim_test=Hanning_filter_func(1, FF_FULL_sim_test) #FIDFs obtained from a moving average filter of Hanning window 400 ms (DeLyca 1987)            
    FIDF_sim_test_array[i] =FIDF_sim_test

    ###############################################################################
    # 6. # Computing and storing the validation metrics (exp vs pred FIDFs)
    ###############################################################################    
    delta_tf1_sim_test=firing_times_sim_test[0]-THRESHOLDS_test[0][0]
    deltaf1_table_sim[i]=delta_tf1_sim_test #stores the RMS errors between experimental filtered FF and virtual simulated FF for the same MN

    non_zero=np.argwhere(FIDF_exp_test>10**-4)[:,0]
    nE= max(abs((FIDF_sim_test[non_zero]- FIDF_exp_test[non_zero])/FIDF_exp_test[non_zero]*100))  
    nME_table_sim[i]=nE #stores the RMS errors between experimental filtered FF and virtual simulated FF for the same MN
    
    RMS_normalized=RMS_func(FIDF_exp_test,FIDF_sim_test)/max(FIDF_exp_test)*100
    RMS_table_sim[i]=RMS_normalized #stores the RMS errors between experimental filtered FF and virtual simulated FF for the same MN

    Corrcoef_table_sim[i]=np.corrcoef(FIDF_sim_test, FIDF_exp_test)[0][1]**2 #COEFFICIENT OF DETERMINATION

    Size_distrib_matrix[i] = np.array([a_size,c_size])

    if plot=='y': plot_exp_vs_sim_FIDF_func(time, range_start,FIDF_exp_test,  FIDF_sim_test, i, 'validation')
    
    print('MN n°'+str(Real_MN_pop_test[0])+' --> Deltaf1, nRMSE, r2 = '+str(round(delta_tf1_sim_test,2)) + ' s, '+ str(round(RMS_normalized,1))+'%, '+str(round(r2,2)))      
    print('#--------------------------------------------------------------------')
    
###############################################################################
# 7. # PLOT FINAL RESULTS
############################################################################### 
if plot=='y': plot_final_results_validation_func(Nb_MN, deltaf1_table_sim, nME_table_sim, RMS_table_sim, Corrcoef_table_sim, Size_distrib_matrix)

###############################################################################
# 8. # SAVE OUTPUTS
############################################################################### 
if save == 'y': 
    prefix = author+'_'+test+'_validation_'
    np.save(prefix+'onset_error', deltaf1_table_sim, allow_pickle=True) # errors in predicting first discharge time
    np.save(prefix+'nRMSE', RMS_table_sim, allow_pickle=True) # nRMSE
    np.save(prefix+'r2', Corrcoef_table_sim, allow_pickle=True) # r2
    np.save(prefix+'Size_distributions', Size_distrib_matrix, allow_pickle=True) # Calibrated sizes 
    np.save(prefix+'FIDF_sim_test', FIDF_sim_test_array, allow_pickle=True) # FIDFs









