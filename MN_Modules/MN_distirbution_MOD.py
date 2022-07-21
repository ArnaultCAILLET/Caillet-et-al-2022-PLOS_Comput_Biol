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
from MN_properties_relationships_MOD import Fth_distrib_func

def MN_distirbution_func(Nb_MN, true_MN_pop, MN_pool_list, muscle, THRESHOLDS):
    '''
This function computes from the experimental force thresholds of the MNs under
study their location in the real MN population of the muscle, in the order of
increasing force thresholds. 

Parameters
----------
$ Nb_MN : the number of discharging MNs to relocate, integer
$ true_MN_pop : Number of MNs in the muscle under study, integer
    This value is either guessed either comes from external exp findings
$ MN_pool_list : list of MNs in complete MN pool, array
$ muscle : muscle under study, string
$ THRESHOLDS : matrix of recruitment thresholds, string
    THRESHOLDS[:,2] contains the list of force recruitment thresholds (%MVC)
    of the MNs under study

Returns
-------
$ Real_MN_pop : Real MN locations in real MN pool, list
    '''    
    
    Real_MN_pop=np.zeros(Nb_MN)
    for i in range (Nb_MN):
        Real_MN_pop[i]=(  np.abs( Fth_distrib_func(MN_pool_list, muscle, true_MN_pop)-THRESHOLDS[i][2] )  ).argmin() #looking for the closest match between typical and experimental %MVC recruitment threshold
    Real_MN_pop=Real_MN_pop.astype(int)
    return Real_MN_pop