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

def IDF_func(Nb_MN, time,   disch_times, t_start, t_stop, fs=2048):
    '''
This function computes from known discharge times the time-history of the 
instantaneous discharge frequencies (IDFs) of the MNs under study

Parameters
----------
$ Nb_MN : the number of discharging MNs involved in the CST, integer
$ time : the time-duration of the simulation, array
    time typically covers the entirety of the duration of the experimental 
    data, is in [s] with a 1/fs time step
$ disch_times : the lists of the MN discharge times, matrix
    if NB_MN==1: disch_times is typically simulated from the LIF model which 
    returns results in sec
    elif Nb_MN>1: disch_times is typically experimental, in which case the 
    values are given in fs-samples
$ t_start : starting time of the simulation, string
    Useful if the FIDFs are only useful over a time window narrower than time
$ t_stop : stopping time of the simulation, string
    Useful if the FIDFs are only useful over a time window narrower than time
$ fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures. 

Returns
-------
$ IDF_dt : lists of MN instantaneous discharge frequencies [Hz], matrix
$ FF_FULL : lists of discharge times, matrix
    Example: if the 1st MN discharges for the first time at time t=3/fs [s] at 
    IDF=12 Hz, FF_FULL = [[0, 0, 1, ...], ...]
    '''    
    
    
    if Nb_MN==1: #Results in [s] obtained from the LIF model
        IDF_dt=np.empty((len(disch_times)-1))
        for j in range(len(IDF_dt)):
            if disch_times[0]>t_stop*2: #
                ISI_dt=(disch_times[j+1]-disch_times[j])/fs #[s] interspike interval
                IDF_dt[j]=1/ISI_dt #[Hz] Corresponding IDF                
            else: #discharge times given in seconds
                ISI_dt=(disch_times[j+1]-disch_times[j]) #[s] interspike interval
                IDF_dt[j]=1/ISI_dt #[Hz] Corresponding IDF
        
        FF_FULL=np.zeros(int(fs*(t_stop-t_start))) #size 61440
        if disch_times[0]<t_stop*2:    #if the discharge times were given in seconds
            dt_arr=(disch_times*fs).astype(int)-int(t_start*fs) #dt_arr contains the time samples at which the MN discharges
        else:
            dt_arr=(disch_times).astype(int)-int(t_start*fs)
        for j in range(len(dt_arr)-1): #looping for all the MN discharge times
            dt=int(dt_arr[j]) 
            FF_FULL[dt]= 1
        
    
    
    else: #Results in [fs samples] obtained from experiments for all MNs
        IDF_dt=np.empty((Nb_MN,), dtype=object) 
        for i in range (Nb_MN):
            IDF_dt[i]=np.empty(len(disch_times[i])-1)
            for j in range (len(disch_times[i])-1):
                ISI_dt=(disch_times[i][j+1]-disch_times[i][j])/fs #[s] interspike interval
                IDF_dt[i][j]=1/ISI_dt #[Hz] Corresponding IDF
        
        FF_FULL=np.empty((Nb_MN,), dtype=object) 
        for i in range (Nb_MN): 
            FF_full_MNi=np.zeros(len(time)) #initalizing the binary array for MN i with zeros everywhere
            dt_arr=disch_times[i] #dt_arr contains the numerotation of the time samples at which MN i discharges
            for j in range(len(dt_arr)-1): #looping for all the discharge time sample of MN i
                dt=int(dt_arr[j]) #the discharge time sample is an integer value that will give the index of the time array at which a discharge occurs
                FF_full_MNi[dt]= 1 
            FF_FULL[i]= FF_full_MNi
            
    return IDF_dt, FF_FULL