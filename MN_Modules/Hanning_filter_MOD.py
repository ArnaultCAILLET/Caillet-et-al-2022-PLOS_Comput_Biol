""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------

This code smoothes arrays of discrete instantaneous discharge frequencies by convolution with a 400ms Hanning window
"""
import numpy as np
from scipy import signal
def Hanning_filter_func(Nb_MU, FF_FULL, fs=2048):
    window_length=0.4 #s, according to DeLuca works
    L=int(window_length*fs) #number of corresponding samples
    Hanning_window=signal.windows.hann(L)#np.hanning(L)
    sum_Han=sum(Hanning_window)    
    
    if Nb_MU>1:
        FF_filt_arr=np.empty((Nb_MU,), dtype=object)
        for i in range (Nb_MU):
            MN_FF_exp=FF_FULL[i]#[range_start:range_stop]
            FF_filt_exp=signal.convolve(MN_FF_exp,Hanning_window, mode='same')/sum_Han
            FF_filt_arr[i]=FF_filt_exp
    
    else:
        FF_filt_arr= signal.convolve(FF_FULL,Hanning_window, mode='same')/sum_Han
         
    return FF_filt_arr*fs