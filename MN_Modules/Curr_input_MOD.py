""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
"""

from MN_properties_relationships_MOD import Ith_distrib_func

def Common_to_current_input_func(THRESHOLDS, Real_MN_pop, true_MN_pop, common_input, fs=2048):
    '''
This function computes an affine transformation to the common input CI(t) to 
derive the time-history of the current input I(t). If CI(t)=0, I(t)=0.
At the first time instant at which CI>0 (i.e. when the first identified MN 
fires), I(t) takes the value of the Ith of this first identified MN (Ith value)
obtained from typical literature distirbution of Ith in a MN poom + from the 
previsouly derived location of that MN in the real MN pool. Same done for the
last firing identified MN. 

I1=I(t1)=Ith(MN1)
In=I(tn)=Ith(MNn)
G = (Ith(MNn)-Ith(MN1))/(CIn - CI1)

I(t)=0 for CI(t)=0
I(t) = I1 + G*CI(t) for CI(t)>0

Parameters
----------
$ THRESHOLDS : matrix of recruitment thresholds, string
    THRESHOLDS[:,1] contains the list of CI recruitment thresholds (%MVC)
    of the MNs under study
$ Real_MN_pop : Real MN locations in real MN pool, list
$ common_input : Common Input, list
    Common input (CI) to the MN pool derived as the filtered cumulative spike 
    train 
$ fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures. 

Returns
-------
$ G, I1: current input coefficients, floats
    Intercept and gain of the CI(t) -> I(t) transformation; I(t) in [A]
    '''    

    max_CI=max(common_input)
    I1= Ith_distrib_func(Real_MN_pop[0], true_MN_pop)   #Expected Ith of the first recruited and identified MN
    CI1=THRESHOLDS[0][1]/100*max_CI #Absolute value of the common input when the first identified MN is recruited
    Ilast=Ith_distrib_func(Real_MN_pop[-1], true_MN_pop)  #Expected Ith of the last recruited and identified MN
    CIlast=THRESHOLDS[-1][1]/100*max_CI #Absolute value of the common input when the last identified MN is recruited
    G= (Ilast-I1)/(CIlast-CI1)
    return G, I1

