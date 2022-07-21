""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
---------
"""

import numpy as np
from Regression import regression
import matplotlib.pyplot as plt
from PLOTS import plot_ARP_trendline_func
def ARP_distrib_func(test, Nb_MN, true_MN_pop, Real_MN_pop, saturating_MN, exp_ARP, muscle, plot, MN_pop):
    '''
This function computes a trendline ARP(MN)=a*MN**b fitting the (MN, ARP) pairs
for the saturating MNs previously identified. The MNs are located in 
the real MN pool. The ARP of the identified but non saturating MNs is predicted
from this trendline. 

Parameters
----------
$ test : name of the set of experimental data under study, string
$ Nb_MN : the number of discharging MNs under study, integer
$ MN_pop : muscle-specific number of MNs in the MN pool, integer
$ Real_MN_pop : Real MN locations in real MN pool, list
$ saturating_MN : list of the MNs showing saturation in IDFs, array
$ exp_ARP : ARPs [s] of the saturating MNs, array
$ plot : whether the most relevant findings are plotted, string
Returns
-------
$ a_arp, b_arp : (MN, ARP) trendline coefficients, floats 
$ ARP_table : list of ARPs of the (non-)saturating MNs under study, array
    ''' 
    
    if len(saturating_MN)>4: #if enough data is available to derive a power law of ARP values
        def func_reg(x,a,b): return a*x**b
        X=Real_MN_pop[saturating_MN][:,0].astype(float) #Saturating MNs in the real population
        Y=exp_ARP[:,0].astype(float)
        popt, r2=regression(X,Y, 'power',muscle, MN_pop)
        a_arp,b_arp =popt[0], popt[1]
        
        if plot=='y': plot_ARP_trendline_func(X, Y, true_MN_pop, func_reg, popt, a_arp, b_arp, r2)
    
        #Building the final ARP table
        ARP_table=np.empty((Nb_MN,), dtype=object)
        ARP_table[saturating_MN]=exp_ARP #first filling the table with the data of the saturating MNs
        Non_saturating_MN= np.argwhere(ARP_table==None)[:,0] #Indexes of non-saturating MNs
        ARP_table[Non_saturating_MN]=func_reg(Real_MN_pop[Non_saturating_MN], *popt)#Filling the rest of the table with the trendline
    
    else: #Defining a custom ARP law max 30Hz for all MNs (for now)
        a_arp=0
        b_arp=0
        set_ARP=1/30
        ARP_table=np.ones(Nb_MN)*set_ARP
        saturating_MN, Non_saturating_MN=0,0
        if test=='GM_10':
            set_ARP=1/20
            ARP_table=np.ones(Nb_MN)*set_ARP    
    
    return a_arp,b_arp,ARP_table, saturating_MN, Non_saturating_MN
            
            
            
            
            
            
            
            