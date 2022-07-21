""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

This code loads the results obtained from the 1_MAIN_MN_model.py
and 2_MN_Model_Validation python codes. 
"""

from EXP_DATA_PROCESSING_MOD import EXP_DATA_PROCESSING_func

def load_MN_model_results(test, path_to_data, author='Hugh'):
    import numpy as np   
    test_cases= np.array([['DelVecchio', 'TA_35_D', 'TA', 10.3,   20.5, 30,   0.35, 400, 2048],
                          ['Hugh',       'test_dataset', 'TA', 10.5,   20.5, 30,   0.35, 400, 2048],
                          ['Hugh',       'TA_35_H', 'TA', 10.5,   20.5, 30,   0.35, 400, 2048],
                          ['Hugh',       'GM_30',   'GM', 9.1,    19.1, 33.5, 0.3,  400, 2048],
                          ['Hugh',       'TA_50',   'TA', 12,     21.8, 34.5, 0.5,  400, 2048],
                          ['Caillet',    'CA_TA_30','TA', 12.5,  32.5, 41, 0.3,  400, 2048],
                          ['Caillet',    'CaAr30_2','TA', 9,  29, 37, 0.3,  400, 2048],
                          ['Caillet',    'CaAr30_3','TA', 9,  29, 37, 0.3,  400, 2048],
                          ['Caillet',    'CaAr35_2','TA', 10,  30, 38, 0.35,  400, 2048],
                          ], dtype=object)
    [author, test, muscle, plateau_time1, plateau_time2, end_force, MVC, MN_pop, fs]= test_cases[np.argwhere(test_cases==test)[0][0]]
    
    #Loading the results obtained from the 1_MAIN_MN_model code
    prefix = author+'_'+test+'_'
    Data = np.array(['time_array', 'exp_force', 'exp_discharge_times', 'PRED_discharge_times', 'parameters', 'MAIN_results', 'calib_onset_error', 'calib_nRMSE', 'calib_r2', 'Calib_FIDF' ])
    Data = np.core.defchararray.add(prefix, Data)
    Data = np.core.defchararray.add(Data, '.npy')
    
    time=np.load( path_to_data / Data[0], allow_pickle=True) # TIME ARRAY
    Force=np.load(path_to_data / Data[1], allow_pickle=True) #EXP FORCE
    exp_disch_times=np.load(path_to_data / Data[2], allow_pickle=True) #EXP SPIKE TRAINS
    Firing_times_sim=np.load(path_to_data / Data[3], allow_pickle=True) #PREDICTED SPIKE TRAINS
    [range_start,range_stop, t_start, end_force, Nb_MN, Cm_calib, Cm_derec_calib, adapt_kR] = np.load(path_to_data / Data[4], allow_pickle=True) # time boundaries, conditions of simulation
    range_start,range_stop, t_start, end_force, Nb_MN = int(range_start),int(range_stop), int(t_start), int(float(end_force)), int(Nb_MN)
    [Real_MN_pop,Calib_sizes, Cm_rec, Cm_derec, a_size, c_size, r2_size_calib, kR_derec, r2_exp, nRMSE_exp, r2_sim, nRMSE_sim] = np.load(path_to_data / Data[5], allow_pickle=True)
    Cm_rec, Cm_derec, kR_derec, r2_exp, nRMSE_exp, r2_sim, nRMSE_sim = round(Cm_rec,3), round(Cm_derec,3), round(kR_derec,3), round(r2_exp,2), round(nRMSE_exp,1), round(r2_sim,2), round(nRMSE_sim,1)

    try:
        calibration_onset_error = np.load(path_to_data / Data[6], allow_pickle=True) 
        calibration_nRMSE = np.load(path_to_data / Data[7], allow_pickle=True) # FDIFs
        calibration_r2 = np.load(path_to_data / Data[8], allow_pickle=True) # FDIFs
        calibration_FIDF = np.load(path_to_data / Data[9], allow_pickle=True) # FDIFs

    except: 
        calibration_onset_error = 0
        calibration_nRMSE = 0
        calibration_r2 = 0
        calibration_FIDF =0
    #Loading the results obtained from the 2_MN_model_validation code
    prefix2 = author+'_'+test+'_validation_'
    Data2 = np.array(['onset_error', 'nRMSE', 'r2', 'Size_distributions', 'FIDF_sim_test' ])
    Data2 = np.core.defchararray.add(prefix2, Data2)
    Data2 = np.core.defchararray.add(Data2, '.npy')
    deltaf1_table_sim= np.load(path_to_data / Data2[0],  allow_pickle=True) # Onset error
    RMS_table_sim= np.load(path_to_data / Data2[1],  allow_pickle=True) # nRMSE
    Corrcoef_table_sim= np.load(path_to_data / Data2[2],  allow_pickle=True) # r2
    Size_distrib_matrix= np.load(path_to_data / Data2[3],  allow_pickle=True) 
    try:
        FIDF_sim_test = np.load(path_to_data / Data2[4], allow_pickle=True) # FDIFs
    except: FIDF_sim_test = 0


        
    return time, muscle, MVC, Force, Nb_MN, MN_pop, Real_MN_pop, exp_disch_times, Firing_times_sim, range_start,range_stop, t_start, plateau_time1, plateau_time2, end_force, fs, Calib_sizes, [author, Cm_calib, Cm_derec_calib, adapt_kR, Cm_rec, Cm_derec, a_size, c_size, r2_size_calib, kR_derec, r2_exp, nRMSE_exp, r2_sim, nRMSE_sim], [deltaf1_table_sim, RMS_table_sim, Corrcoef_table_sim, Size_distrib_matrix, FIDF_sim_test], [calibration_onset_error,calibration_nRMSE,calibration_r2, calibration_FIDF]