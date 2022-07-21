""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------

Computes the root-mean-square value between two arrays of same size
"""

import numpy as np

def RMS_func(arr_exp, arr_simul):
    diff= arr_exp-arr_simul
    return np.sqrt(np.mean(diff**2))