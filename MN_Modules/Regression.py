""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------

Convenient code to perform the few regressions necessary in the main codes
"""

import numpy as np
from scipy.optimize import curve_fit

#Tool to obtain regressions and the r2
def func_Lin(x,a): return a*x
def func_aff(x,a,b): return a*x+b
def func_quadra(x,a,b,c):return a*x**2+b*x+c
def func_cubic(x,a,b,c,d):return a*x**3+b*x**2+c*x+d
def func_power(x,a,b): return a*x**b    




def regression(X,Y, fun, muscle, MN_pop, sigma=None):

    if muscle =='TA' or muscle =='GM':
        def size_power(x,a,c): return a*2.4**(((x+1)/MN_pop)**c)# a*4**(((x+1)/400)**c)
        def threshold_power(x,a,b,k): return k*(a*(x+1)/MN_pop+90**(((x+1)/MN_pop)**b))
    else:
        def size_power(x,a,c): return a*2.4**(((x+1)/550)**c)# a*4**(((x+1)/400)**c)
        def threshold_power(x,a,b,k): return k*(a*(x+1)/550+90**(((x+1)/550)**b))   

    if fun=='lin': func= func_Lin
    elif fun=='quadra': func= func_quadra
    elif fun=='aff': func=func_aff
    elif fun=='power': func= func_power
    elif fun=='size': func=size_power
    elif fun=='threshold': func=threshold_power

    popt, pcov = curve_fit(func, X,Y, sigma=sigma)
    residuals = Y- func(X, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y-np.mean(Y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    return popt, r_squared