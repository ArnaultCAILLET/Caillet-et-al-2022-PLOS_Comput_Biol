""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------

Defining some initial parameters and reshaping some input arrays
"""
import numpy as np 

def preprocessing_func(author, Force, end_force, fs, Nb_MN, plateau_time1, plateau_time2):
    Force=Force[0:int(end_force*fs)]
    Force = Force-np.min(Force[:6000]) #removing possible offset
    
    #time list of step time 1/fs
    time=np.linspace(0,end_force,int(end_force*fs)) 
    #Numbering of the indentified MNs
    MN_list=np.arange(0,Nb_MN,1)
    #Giving flexibility on starting and finishing calibrations / plots at different times
    t_start=0
    t_stop=end_force
    t_stop_calib=(plateau_time2+plateau_time1)/2
    kR=1.68*10**-10
    Cm_rec=1.3*10**-2
    step_size=10**-4 #time step s, given in ms; for accurate solutions, the time step needs to be low enough. For high frequencies, prefer 10**-5. 
    
    return Force, time, MN_list, t_start, t_stop, t_stop_calib, kR, Cm_rec, step_size


