""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------
"""

from scipy import signal

def But_filter_func(fc, raw_signal, fs=2048):
    '''
This function applies a 4th-order Butterworth filter to data sampled at fs.

Parameters
----------
$ fc : cut-off frequency of the filter, float
    When filtering a CST, the common input and control are respectively 
    obtained for fc=10Hz and fc=4Hz (Farina et al., 2015)
$ raw signal : signal to be filtered, array
    typcally a CST in this workflow
 

Returns
-------
$ filtered_signal : filtered signal, array
    Typically return a CI or a CC in this worflow
    '''   
    
    w = fc / (fs / 2) # Normalize the frequency by the Nyquist frequency
    o, p = signal.butter(4, w, 'low', analog=False)
    filtered_signal = signal.filtfilt(o, p, raw_signal) 
    return filtered_signal