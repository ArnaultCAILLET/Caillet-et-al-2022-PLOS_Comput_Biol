""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

This code performs the sensitivity analysis upon the values of Cm_derec to best fit the MN FIDFs during derecruitment
"""

import numpy as np
from FF_filt_func import FF_filt_func
import matplotlib.pyplot as plt
from RMS_func import RMS_func

def Cm_derec_sensitivity_func (Nb_MN, time, I, Cm_rec, Cm_derec_array, Calib_sizes_final, ARP_table, FIDF_exp, plateau_time1, end_force, step_size, kR, adapt_kR='n', kR_derec=1.7*10**-10, fs=2048):
    t1=plateau_time1
    t2=end_force
    
    range_t1=int(t1*fs)
    range_t2=int(t2*fs)
    RMS_table_derec=np.empty((Nb_MN,), dtype=object)
    r2_table_derec=np.empty((Nb_MN,), dtype=object)
    RMS_cumulated_table_derec=np.empty((len(Cm_derec_array),), dtype=object)
    r2_mean_table_derec=np.empty((len(Cm_derec_array),), dtype=object)
    
    for j in range (len(Cm_derec_array)):
        Cm_derec=Cm_derec_array[j]
        
        for i in range (Nb_MN):
            if i%5==0 : print('Firing MN n°', i, ' with Cm_derec = ', np.round(Cm_derec, 3))
            MNSize=Calib_sizes_final[i]
            ARP=ARP_table[i]
            filt_FF, NA,  NB= FF_filt_func(I,  t1, t2, t1, MNSize, Cm_rec, Cm_derec, step_size, ARP,  kR, adapt_kR, kR_derec)
            #Making sure all arrays have the same length
            k1=len(filt_FF)
            k2=len(FIDF_exp[i][range_t1:range_t2])
            k3=k2-k1
            FF_filt_exp=FIDF_exp[i][range_t1:range_t2-k3] 
            
            rms_absolute=RMS_func(FF_filt_exp, filt_FF)
            rms_relative_max=rms_absolute/max(FF_filt_exp)*100
            corcoeff=np.corrcoef(filt_FF, FF_filt_exp)[0][1]**2
            RMS_table_derec[i]=rms_relative_max
            r2_table_derec[i]=corcoeff**2 #COEFFICIENT OF DETERMINATION
            
            if i%4==0: #Cm_derec in np.array([0.01, 0.02, 0.03, 0.04, 0.05]) :
                try: 
                    plt.plot (time[range_t1:range_t2], FF_filt_exp, 'k', label='Experimental')
                    # plt.xlim(10,11)
                    plt.plot (time[range_t1:range_t2], filt_FF, label='Simulated')   
                except:
                    plt.plot (time[range_t1:range_t2-1], FF_filt_exp, 'k', label='Experimental')
                    # plt.xlim(10,11)
                    plt.plot (time[range_t1:range_t2-1], filt_FF, label='Simulated')                  
                plt.xlabel('Time(s)')
                plt.ylabel('Filtered discharge intervals')
                plt.title('MN '+str(i))
                # plt.ylim(0,0.0009)
                # plt.xlim(2,4)
                plt.grid()
                plt.legend()
                plt.show()            
            
        RMS_cumulated_table_derec[j]=np.average(RMS_table_derec) #average nRMSE across the Nb_MN calibrations
        r2_mean_table_derec[j]=np.average(r2_table_derec)
    RMS_normalized_table_derec=RMS_cumulated_table_derec/max(RMS_cumulated_table_derec)   #Normalizing the nRMSE sensitivity table to pick the best Cm_derec
    
    Cm_derec=Cm_derec_array[np.argmin(RMS_normalized_table_derec-r2_mean_table_derec)] 
    return Cm_derec_array, RMS_cumulated_table_derec, r2_mean_table_derec, Cm_derec #Coefficient of determination