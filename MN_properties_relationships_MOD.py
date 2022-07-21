""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

Relationships for the LIF model between MN properties. Obtained from Caillet et al. 2022, eLife 'Mathematical Relationships between MN properties'
"""

import numpy as np
import matplotlib.pyplot as plt

def Ith_S_func(S):
    return 3.82*10**8*S**2.52

def S_Ith(Ith):
    return 3.96*10**-4*Ith**0.396

def R_S_func(S, kR=1.68*10**-10):
    return kR/S**2.43

def C_S_func(S, Cm_rec): 
    return Cm_rec*S #[F]

def tau_R_C_func(R,C):
    return R*C #[s]




def Fth_distrib_func(MN, muscle, MN_pop, k=1): #Obtained from processed literature data, typical muscle-specific distributions of MU force recruitment thresholds (%MVC)
    if muscle=='TA':
        # if k==1 : return 0.93*(30.6*MN/MN_pop+66.2**((MN/MN_pop)**2.09))
        return 0.5052*(58.1*MN/MN_pop+120**((MN/MN_pop)**1.83)) #0.6503*(47.6*MN/MN_pop+90**((MN/MN_pop)**1.92)) Should be changed to accomodate a 90x factor!!
        if k==2 : return 1.11*np.exp(0.045*MN/4) #0.6503*(47.6*MN/MN_pop+90**((MN/MN_pop)**1.92))

    elif muscle=='GM':
        return 0.6562*(46.7*MN/MN_pop+90**((MN/MN_pop)**1.79))

def Ftet_norm_distrib_func(MN, muscle, MN_pop, k=1): 
#FOR EXPLANATIONS AND DETAILS? SEE Fitting_Fth_experimental_for_Ta_model.py
# Ratio of extreme MU maximum forces = 11
    return 8.9324*(3.0*MN/MN_pop+8.20**((MN/MN_pop)**5.29))    
    
def Ith_distrib_func(MN, true_MN_pop): #Obtained from literature data
    '''
Function returning the typical rheobase distribution across a MN pool, 
obtained from rats and cats experimental data
Parameters
----------
$ MN : location of the MN in the real MN population, integer

Returns
-------
$ Ith : rheobase [A] of the MN, float    
    '''
    Ith=3.85*10**-9*9.1**((MN/true_MN_pop)**1.1831)#2.09*10**-9*16**((MN/400)**1.5725)
    return Ith

