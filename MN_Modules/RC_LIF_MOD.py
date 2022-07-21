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
from MN_properties_relationships_MOD import R_S_func, C_S_func, tau_R_C_func

def RC_solve_func(I, t_start, t_stop, t_plateau_end, Size, Cm_rec, Cm_derec,  step_size, ARP, kR=1.68*10**-10, adapt_kR='n', kR_derec=1.7*10**-10,  ARP_rand=10): 
    '''
This function solves the ODE of integrator RC circuit using a hybrid analytical 
numerical method of Fourier transforms and convolution. 
This method is computationally cheaper than a numerical Runge-Kutta method
For simplicity without loss of generality, V varies in [0,27mV] rather than
in [-85;-58mV]. The precision is given by step_size. Inputs and outputs are in
[s]. Two different values of specific capacitance are given for recruitment and
derecruitment phases. 
MN Size defines all other relevant MN parameters R, C and tau

    Parameters
    ----------
    I : function of time, [A]
        Time-course of the current input to the system
    t_start : float, [s]
        Start time of the simulation
    t_stop : float, [s]
        End time of the simulation.
    Size : float, [m2]
        The size of the MN. This parameter controls the whole model and the 
        value of the other parameters
    Cm_rec : float, [F/m2]
        Specific capacitance of the RC circuit for the recruitment phase:
        t<t_plateau_end
    Cm_derec : float, [F/m2]
        Specific capacitance of the RC circuit for the derecruitment phase:
        t>t_plateau_end
    step_size : float, [s]
        Time step. For accuracte predictions, use a time step of 10**-4 or less
    ARP : MN-specific ARP value [s], float
    kR : Ith-R gain, float
        Typical gain Ith=kR*R in the literature (Caillet, 2021)

    Returns
    -------
    tim_list : numpy array, [s]
        List of time instants of time step 'step_size' at which the values of 
        the membrane potential V are calculated
    V : numpy array, [V]
        List of calculated membrane potentials at each time instants V(tim_list)
    firing_times_arr : numpy array, [s]
        List of the time instants at which an action potential is fired
    parameters: numpy array
        List of important parameters related to the size of the MN input to 
        the func

    '''
    # MN PARAMETERS
    Vth = 27*10**-3 #[V] 
    ARP_ini=ARP #[s] 
    R = R_S_func(Size, kR)# [Ohm] 
    C = C_S_func(Size, Cm_rec) #[F]
    tau = tau_R_C_func(R,C) #[s]
    parameters=[Vth, ARP_ini, R, C, tau]
    
    #TIME LIST WITH THE DEFINED STEP SIZE
    tim_list=np.arange(t_start,t_stop,step_size) #generating the time list
    
    #INITIALIZATION
    V=np.zeros(len(tim_list))
    firing_times_arr=np.array([])
    t_fire=-7*tau #•initializing

    #SOLVING
    for i in range(len(tim_list)): #solving at each time instant
        # INITIAL TIME
        if i==0: Vnt=R*step_size/tau*I(t_start) # Initial condition on V
        
        else:
            #DERECRUITMENT PHASE
            if tim_list[i]>t_plateau_end: #switch Cm_rec to Cm_derec if in derecruitment phase
                if adapt_kR == 'y':
                    R = R_S_func(Size, kR_derec)
                C = C_S_func(Size, Cm_derec) #[F]
                tau = tau_R_C_func(R,C) #[s]
            
            nt=tim_list[i] #time instant
            Vnt=np.exp(-step_size/tau)*Vnt+R*step_size/tau*I(nt) #LIF equation of MN membrane potential
            
            
            #FIRING TIME
            if Vnt>Vth: #If the threshold is reached at time nt
                Vnt=0 #The potential is reset to rest state
                V[i]=0 #V(nt)=0
                firing_times_arr=np.append(firing_times_arr, nt) #Storing the firing time nt
                t_fire=nt #Reset tfire with the latest found value nt
                ARP=np.random.normal(ARP_ini, ARP_ini/ARP_rand) #Randomly vary the ARP value following a Gaussian probabilistic curve of sigma=ARP/10, to model the fulctuations in firing rate saturation
                
                
            #REFRACTORY PERIOD    
            elif nt>t_fire and nt<t_fire+ARP: #If during the refractory period
                V[i]=0 #The potential remains at resting state
                Vnt=0
            
            # MEMBRANE CHARGING    
            else: V[i]=Vnt #In any other cases, calculate V(nt) and store
    return tim_list, V, firing_times_arr, parameters
        
    


