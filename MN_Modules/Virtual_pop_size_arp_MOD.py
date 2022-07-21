""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)
----------
Reconstructing with the Size and ARP trendlines the MN properties of the complete MN pool
"""
import numpy as np

def Virtual_size_arp_pop(MN_pop, a_size, c_size, exp_ARP, a_arp, b_arp):
    
    Virtual_size_arr=np.empty((MN_pop,), dtype=object)
    Virtual_ARP_arr=np.empty((MN_pop,), dtype=object)

    MN_list=np.arange(1,MN_pop+1,1)
    for MN in MN_list:
        Virtual_size_arr[MN-1]= a_size*2.4**((MN/MN_pop)**c_size) #Find the relationship from MN population vs size. 
        if len(exp_ARP)>4:
            Virtual_ARP_arr[MN-1] = a_arp*MN**b_arp
        else:
            Virtual_ARP_arr[MN-1]=0.3
    
    return Virtual_size_arr, Virtual_ARP_arr
    