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

def exp_thresholds_func(Nb_MN, MVC, disch_times, common_input, Force, fs=2048):
    '''
This function computes the MN CI and force (de)recruitment thresholds 
from the lists of MN discharge times obtained from decomposed HDEMG signals 

Parameters
----------
Nb_MN : the number of identified discharging MNs, integer
MVC : target % max voluntary contraction during experiments, integer
disch_times : the lists of the MN discharge times, matrix
    The discharge times are typically given in samples (fs=2048 Hz in all 
    datasets). 
common_input : Common Input, array
    Computed as the filtered CST in [0; 10Hz]
Force : the time-history of exp transducer Force amplitude, array 
fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures.    

    
Returns
-------
THRESHOLDS : matrix of MN thresholds, matrix
    THRESHOLDS[0]: list of MN first discharge times [s], precision = ms
    THRESHOLDS[1]: list of recruitment thresholds (% CI)
    THRESHOLDS[2]: list of recruitment thresholds (% MVC)
    THRESHOLDS[3]: list of MN last discharge times [s], precision = ms
    THRESHOLDS[4]: list of derecruitment thresholds (% CI)
    THRESHOLDS[5]: list of derecruitment thresholds (% MVC)
    '''   
    
    THRESHOLDS=np.zeros((Nb_MN,6))
    max_CI=max(common_input) 
    max_trans_force=max(Force)
    
    for i in range (Nb_MN):
        if Nb_MN==1:
            first_firing=disch_times[0]
            last_firing=disch_times[-1]  
        else:
            first_firing=disch_times[i][0]
            last_firing = disch_times[i][-1]                
        first_disch_time= round(first_firing/fs,3) #seconds
        CI_th=common_input[int(first_firing)]/max_CI*100
        F_th=Force[int(first_firing)]/max_trans_force*MVC*100
        last_disch_time=round(last_firing/fs,3)
        CI_dth=int(common_input[int(last_firing)]/max_CI*100)
        F_dth=int(Force[int(last_firing)]/max_trans_force*MVC*100)
            
        THRESHOLDS[i][0]=first_disch_time #Time of first discharge (s)
        THRESHOLDS[i][1]=CI_th # % of common input when the MN discharges for the first time
        THRESHOLDS[i][2]=F_th #Force recruitment threhsold %MVC
        THRESHOLDS[i][3]= last_disch_time#time of last discharge
        THRESHOLDS[i][4]=CI_dth # % of common input when the MN discharges for the last time
        THRESHOLDS[i][5]=F_dth
        
    return THRESHOLDS