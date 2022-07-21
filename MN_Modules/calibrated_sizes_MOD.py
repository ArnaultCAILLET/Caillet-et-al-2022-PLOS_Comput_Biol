""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------

Finding a power trendline fitting the calibrated MN sizes
"""
from Regression import regression
# import numpy as np


def calibrated_sizes_func(Real_MU_pop,Calib_sizes_final, muscle, MN_pop):
    popt, r2=regression(Real_MU_pop,Calib_sizes_final, 'size', muscle, MN_pop)
    return Real_MU_pop, Calib_sizes_final, popt, r2