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
import matplotlib.pyplot as plt

def exp_ARP_func(Nb_MN, plateau_time1, plateau_time2, end_force, disch_times, IDF_dt, fs=2048):
    '''
This function identifies the MNs that show a saturation in IDFs, i.e. the IDFs
stop increasing despite increasing common input at the same time instant. For
each saturating MN, an absolute refractory period is derived as 1/ average
saturating IDF.

Parameters
----------
$ Nb_MN : the number of discharging MNs under study, integer
$ plateau_time1 : starting time [s] of the exp plateau of force, array
$ plateau_time2 : ending time [s] of the exp plateau of force, array
$ disch_times : the lists of the MN discharge times, matrix
    typically given in sample times
$ IDF_dt : lists of MN instantaneous discharge frequencies [Hz], matrix
$ fs : sampling frequency, float
    Typically 2048Hz, this value is consistent with the exp measures. 

Returns
-------
$ saturating_MN : list of the MNs showing saturation in IDFs, array
    The saturation in IDFs is validated if IDF(t) reaches the average value 
    IDF over the plateau of force at least 1 sec before plateau_time1. 
$ exp_ARP : ARPs [s] of the saturating MNs, array
    '''    
    
    #DERIVING THE TRENDLINE
    FF_trend=np.empty((Nb_MN,), dtype=object) #will include the trendline of FF for the whole simulation
    FF_trend_reduced=np.empty((Nb_MN,), dtype=object) #will include the trendline of FF for the plateau only
    FF_trend_max=np.empty((Nb_MN,), dtype=object) #Table storing the maximum value return by the FF trendline for each MN
    FF_trend_reduced_mean=np.empty((Nb_MN,), dtype=object) #will return the mean value of the trendline during the force plateau
    
    for MN in range (Nb_MN):
        #Creating a trendline for each MN for the time course of their instandaneous frequencies
        X=disch_times[MN][0:(len(disch_times[MN])-1)]/fs
        Y=IDF_dt[MN]
        FF_trend[MN]=np.poly1d(np.polyfit(X.astype(float),Y.astype(float),12))(X.astype(float)) #Deriving the trendline
        FF_trend_reduced[MN]=FF_trend[MN][np.argwhere((X > plateau_time1) & (X < plateau_time2))[:,0]] #only interested in what happens during the plateau (firing times)
        FF_trend_max[MN]= max(FF_trend[MN])
        FF_trend_reduced_mean[MN]=np.average(FF_trend_reduced[MN].astype(float))
        
        # plt.plot(X,Y, 'r', label='Instantaneous frequencies of MN'+str(MN))    
        # plt.plot(X,FF_trend[MN],'k', linewidth=4, label='trendline of MN'+str(MN))
        # plt.plot(np.linspace(0,end_force,1000), np.ones(1000)*FF_trend_reduced_mean[MN],linewidth=2, label='Mean of trendline during force plateau')
        # plt.plot([plateau_time1-1.0,plateau_time1-1.0], [0, 30], linewidth=2, label='Force plateau start')
        # plt.xlabel('Time [s]', color='k',fontsize=15, fontweight='bold')
        # plt.ylabel('MN'+str(MN)+' instantaneous frequency', color='r',fontsize=15, fontweight='bold')
        # plt.ylim(0,30)
        # plt.xlim(0,end_force)
        # plt.legend(loc=2)
        # plt.grid()
        # plt.show()
    
    
    #IDENTIFYING THE SATURATING MNs and their saturation time
    is_saturating_table=np.empty((Nb_MN,), dtype=object) #this table contains for each MN either '0' if the MN is not saturating, either the time instant at which it is
    for MN in range (Nb_MN):
        DT=disch_times[MN][disch_times[MN]<(plateau_time1-1.0)*fs]/fs #MN discharge times in sec occuring before plateautime1-1.0sec
        if DT.size==0: # if the MN is recruited during the plateau of force
            is_saturating_table[MN]=0 #this MN is not saturating
        else:
            trend_FF=FF_trend[MN][0:len(DT)] #FF trendline of the MN for the whole simulation, it is of the same length as the list of discharge times
            mean_trend=FF_trend_reduced_mean[MN]*0.95 #mean FF during the plateau
            K=np.argwhere(trend_FF>mean_trend) #Looking for all the indexes of firing times for which the FF trendline is above its mean value calculated during the force plateau
            if K.size==0: #if the FF trendline is always below its plateau mean or if the first occurence of trendline > mean occurs during the plateau, and if the initial FF is above the mean of the trendline, we cannot conclude anything
                is_saturating_table[MN]=0 #the MN is not saturating
            elif trend_FF[0]>mean_trend:
                is_saturating_table[MN]=0 #the MN is not saturating
            else: #in all other cases the MN is saturating: FF trendline >= mean FF trendline plateau at time t < start plateau
                idx = K[0][0]-1 # np.argwhere(   np.argwhere(trend_FF<mean_trend*0.9)[:,0] <  np.argwhere(DT<plateau_time1-1.0)[:,0][-1]  )  [-1][0]#index of value in trend_FF when the trendline reaches for the first time 90% of the mean value plateau
            #for idx0 (example for MN0), it is searched the indexes of trend_FF for which the value are below mean trend, then constrained to the indexes of time values occuring previous to time_plateau_1, and then the last index of the remaining list is recovered: this is the last firing time before the FF trendline reaches  the mean plateau value
                is_saturating_table[MN]=disch_times[MN][idx]/fs#corresponding time of saturation
    saturating_MN=np.argwhere(is_saturating_table>0)# & (is_saturating_table<plateau_time1-1.0)) #list of the MNs known to be saturating 
    
    #DERIVING THE ARP TRENDLINE
    exp_ARP=1/FF_trend_max[saturating_MN] #Experimental ARPs for the saturating MNs
    return saturating_MN, exp_ARP