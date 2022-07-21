""" 
Author: Arnault CAILLET
arnault.caillet17@imperial.ac.uk
July 2022
Imperial College London
Department of Civil Engineering
This code contributes to producing the results presented in the manuscript Caillet et al. 'Estimation of the firing behaviour of a complete motoneuron pool by combining electromyography signal decomposition and realistic motoneuron modelling' (2022)

---------
This code produce the plots for the three main python codes
"""

import matplotlib.pyplot as plt
from Regression import regression
import numpy as np
import matplotlib
from FF_filt_func import FF_filt_func

def plot_CST_func(time, CST_exp, end_force):
    plt.plot(time,CST_exp)
    plt.xlabel('Time [s]', color='k',fontsize=12)
    plt.xlim(0, end_force)
    plt.ylabel('CST', color='k',fontsize=12)
    plt.show()
    
    
def plot_common_inputs_func(time, common_input_exp, common_control_exp, common_noise_exp, end_force):
    plt.plot(time, common_input_exp, 'r', label ='common input  [0-10Hz]')
    plt.plot(time, common_control_exp, 'k', label ='common control  [0-4Hz]')
    plt.plot(time, common_noise_exp, label='common noise [0-10Hz]')
    plt.xlim(0, end_force)
    plt.xlabel('Time [s]', color='k',fontsize=12)
    plt.legend()
    plt.show()
    
def plot_force_thresholds_func(rec_th, derec_th, muscle, MVC, plot, MN_pop):     
    def func_lin(x,a): return a*x
    popt, r_squared=regression(rec_th, derec_th, 'lin', muscle, MN_pop)
    if plot=='y': 
        plt.scatter(rec_th, derec_th, marker='x', color='tab:blue')
        plt.plot(rec_th, func_lin(rec_th, *popt), 'tab:blue', linewidth=3, label=str(round(popt[0],2))+'x'+' , '+r'$r^2=$'+str(round(r_squared,2)))
        plt.plot(np.arange(35), np.arange(35), linestyle='dotted', color='k')
        plt.xlabel('Force recruitment threshold (%MVC)', color='k',fontsize=12)
        plt.ylabel('Force derecruitment threshold (%MVC)', color='k',fontsize=12)
        plt.xlim(0,MVC*100)
        plt.ylim(0,MVC*100)
        plt.show()
    return popt
    
def plot_loc_in_real_MN_pop(MN_list, Real_MN_pop, MN_pop):
    plt.scatter(MN_list, Real_MN_pop)
    plt.ylabel('n° in real MU pool')
    plt.xlabel('Identified MUs in order of increasing Fth')
    plt.ylim(0,MN_pop)
    plt.grid()
    plt.title('Location of identified MNs into real MN pool')
    plt.show()

def plot_IDF_FIDF_func(time, disch_times,IDF, FF_FULL, FIDF_exp, end_force, fs=2048):
    # plt.plot(time,IDF)
    plt.scatter(disch_times[0:len(IDF)]/fs, IDF, s=5, label='IDF')
    smoothed_IDF = np.poly1d(np.polyfit(time[FF_FULL>0].astype(float),IDF.astype(float),6))(time[FF_FULL>0].astype(float))
    plt.plot(time[FF_FULL>0], smoothed_IDF, color='g', linewidth = 3, label='FIDF (6th order polynomial)')
    # plt.scatter(time[IDF>0],IDF[IDF>0], color='tab:blue', marker='.')
    plt.title('Firing frequency time-history')
    plt.xlim(0, end_force)
    # plt.ylim(0, max(IDF)*1.05)
    plt.plot(time,FIDF_exp, 'r', label = 'FIDF (Hanning window)')
    # plt.scatter(time,FIDF_exp, 'r')
    plt.xlabel('Time [s]', color='k',fontsize=12)
    plt.ylabel(r'$(F)IDF$', color='k',fontsize=12)
    plt.legend()
    plt.show()

def plot_ARP_trendline_func(X, Y, MN_pop, func_reg, popt, a_arp, b_arp, r2):
    plt.plot(np.arange(min(X), max(X)+1,1), func_reg(np.arange(min(X), max(X)+1,1), *popt), 'k', label=str(round(a_arp, 3))+'*x^'+str(round(b_arp, 3))+' , '+r'$r^2 = $'+str(round(r2,2)))
    plt.scatter(X, Y)
    plt.xlim(0,MN_pop)
    plt.ylim(0,max(Y)*1.1)
    plt.xlabel('Location in real MN population')
    plt.ylabel('IP duration (s)')
    plt.title('Inert Period (IP) trendline')
    plt.show()

def plot_MN_saturation_func(saturating_MN, Non_saturating_MN, Real_MN_pop, ARP_table):    
    plt.scatter(Real_MN_pop[saturating_MN], ARP_table[saturating_MN]*1000, marker='x', c='k', label='obtained')
    plt.scatter(Real_MN_pop[Non_saturating_MN], ARP_table[Non_saturating_MN]*1000, c='r', label = 'predicted')
    plt.xlabel('MN population')
    plt.ylabel('Inert period IP [ms]')
    plt.ylim(0,1.1*max(ARP_table)*1000)
    # plt.legend()
    # plt.grid()
    plt.title('Obtained and predicted MN IP')
    plt.show()    

def plot_spike_trains_force_CI_func(Nb_MN, Force, time, common_input_exp, common_control_exp,  t_start, end_force, fs, disch_times, MVC, stage='exp' ):    
    if stage == 'exp': width = 0.7
    else: width, fs = 0.99, 1
    colors1, lineoffsets1, linelengths1 = ['C{}'.format(i) for i in range(Nb_MN)], np.arange(1,Nb_MN+1,1), np.ones((1,Nb_MN))[0]*width
    fig, axs = plt.subplots(nrows=1, ncols=1,  figsize=(9, 6))
    axs.set_xlabel('Time [s]', color='k',fontsize=20)
    if stage == 'exp':
        ylabel='Motoneurons ' + r'$(N_{r})$'
    else: 
        ylabel='Motoneurons ' + r'$(N)$'
    axs.set_ylabel(ylabel, color='k',fontsize=20)
    axs.set_ylim(0,  Nb_MN+5)
    # axs.set_xlim(11,14)

    axs.ax2= axs.twinx()
    f = lambda x,pos: str(x).rstrip('0').rstrip('.')
    axs.ax2.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    axs.ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(f))
    axs.ax2.set_ylabel('Transducer force (%MVC)',fontsize=15)
    # axs.ax2.set_ylabel('Transducer force (%MVC) - Common inputs (AU)',fontsize=15)
    axs.ax2.set_xlim(t_start, end_force)
    axs.ax2.set_ylim(0, MVC*140)#max(common_input_exp)*0.78)
    
    # PLOTTING THE DISCHARGE EVENT PLOT 
    axs.eventplot(disch_times/fs, colors=colors1, lineoffsets=lineoffsets1,linelengths=linelengths1)
    # axs.plot(time, common_input_exp/0.32, 'r', label ='common input  [0-10Hz]')
    # axs.plot(time, common_control_exp/0.32, 'k', label ='common control  [0-4Hz]')
    # axs.plot(time, common_noise_exp/0.32, label='common noise [0-10Hz]')
    

    #PLOTTING THE COMMON INPUT & CONTROL & FORCE TRACES
    # norm_CI = common_input_exp/max(common_input_exp)*MVC*140
    # norm_CC = common_control_exp/max(common_control_exp)*MVC*160
    norm_Force = Force/max(Force)*MVC*130
    # axs.ax2.plot(time, norm_CI , label='Common input to the Motor Neuron pool')
    # axs.ax2.plot(time, norm_CC,linewidth=3, label='Common control to the Motor Neuron pool')
    axs.ax2.plot(time, norm_Force,  '-g', linewidth=3, label='Experimental transducer force')#' Whole Muscle Force '+r'$F^{MT} (t)$' )

    plt.show()
   
def plot_current_input(time, I, end_force):
    I_array_plot= [I(time[i])*10**9 for i in range (len(time))] #, THRESHOLDS, common_input, I1, G
    plt.plot(time, I_array_plot, '-g')
    plt.xlim(0,end_force)
    plt.ylim(0,max(I_array_plot)*1.1)
    plt.xlabel('Time [s]', color='k',fontsize=14)
    plt.ylabel('Common synpatic current [nA]', color='k',fontsize=14)
    plt.title('Common Synaptic Input Current', color='k',fontsize=15, fontweight='bold')
    plt.show()    
    return I_array_plot

def plot_exp_pred_FIDFs_func(time,range_start, range_stop,  FF_filt_exp, I,  t_start, t_stop, plateau_time2, S, Cm_rec, Cm_derec, step_size, ARP, i, kR):
        # Plotting exp vs simulated FIDF for each MN if not sensitivity analysis
    plt.plot (time[range_start:range_stop-200], FF_filt_exp[0:len(FF_filt_exp)-200], 'k', label='Experimental')
    plt.plot (time[range_start:range_stop-200], FF_filt_func(I,  t_start, t_stop, plateau_time2, S, Cm_rec, Cm_derec, step_size, ARP, kR)[0][0:len(FF_filt_exp)-200], label='Simulated')
    plt.xlabel('Time(s)', fontsize=12)
    plt.ylabel('FIDF', fontsize=12)
    plt.xlim(0,t_stop)
    plt.ylim(0,max(FF_filt_exp)*1.05)
    plt.title('Calibration of MN size for MN '+str(i))
    # plt.grid()
    plt.legend()
    plt.show()

# def plot_Cm_sensitivity_analysis_func(Cm_rec_array, sol): 
#     fig, axs = plt.subplots(2, 1)
#     axs[0].scatter(Cm_rec_array, sol[2])
#     axs[0].set_title('Sensitivity - Mean r2 values per value of Cm')
#     axs[0].set_ylim(0,1)
#     axs[1].scatter(Cm_rec_array, sol[1])
#     axs[1].set_title('Sensitivity - Cumulated RMS values (% max exp value) per value of Cm')
#     plt.tight_layout()
#     for ax in axs.flat:
#         ax.set(xlabel='Cm')
#         ax.set_xlim(Cm_rec_array[0],Cm_rec_array[-1])
#         ax.grid()
#     plt.show()    

def plot_Cm_derec_sensitivity_analysis_func(Cm_derec_array, r2_mean_table_derec, RMS_normalized_table_derec):
    plt.scatter(Cm_derec_array, r2_mean_table_derec, marker='x', c='k')
    plt.title('Sensitivity - Mean r2 value per value of Cm_derec')
    plt.xlabel('Identified MNs')
    plt.ylabel(r'$r^2$') 
    plt.xlim(min(Cm_derec_array), max(Cm_derec_array))
    plt.show()

    plt.scatter(Cm_derec_array, RMS_normalized_table_derec, marker='x', c='k')
    plt.xlabel('Identified MNs')
    plt.ylabel(r'$nRMSE$') 
    plt.title('Sensitivity - Cumulated RMS values (% max exp value) per value of Cm_derec')
    plt.xlim(min(Cm_derec_array), max(Cm_derec_array))
    plt.show()
    
def plot_achieved_error_S_calibration(Nb_MN,Calib_sizes, Calib_delta_tf1, Calib_RMS_table, Calib_r2_table):
    X=np.arange(Nb_MN) 

    plt.scatter(X, Calib_sizes, marker='x', c='k')
    plt.xlabel('Identified MNs') 
    plt.ylabel(r'$MN size [m^2]$')
    plt.title('Calibrated MN Sizes')
    plt.show()
    
    plt.scatter(X+1, Calib_delta_tf1, marker='x', c='k')
    plt.plot([0,Nb_MN+1], [0.25,0.25], linestyle='--', c='k')
    plt.plot([0,Nb_MN+1], [-0.25,-0.25], linestyle='--', c='k')
    plt.xlabel('Identified MNs')
    plt.xlim(0,Nb_MN+1)
    plt.ylim(-1.5,1.5)
    ylabel=r'$\Delta ft^1$' + ' [s]' 
    plt.ylabel(ylabel)
    plt.title('Onset error obtained from Size calibration')
    plt.show()
    
    plt.scatter(X+1, Calib_RMS_table, marker='x', c='k')
    plt.plot([0,Nb_MN+1], [15, 15], linestyle='--', c='k')
    plt.xlabel('Identified MNs')
    plt.ylabel(r'$nRMSE$') 
    plt.xlim(0,Nb_MN+1)
    plt.ylim(0,100)
    plt.title('Exp vs Sim FIDFs nRMSE obtained from Size calibration')
    plt.show()

    plt.scatter(X+1, Calib_r2_table, marker='x', c='k')
    plt.plot([0,Nb_MN+1], [0.8,0.8], linestyle='--', c='k')
    plt.xlabel('Identified MNs')
    plt.ylabel(r'$r^2$') 
    plt.xlim(0,Nb_MN+1)
    plt.ylim(0,1.0)
    plt.title('Exp vs Sim FIDFs r2 obtained from Size calibration')
    plt.show()    
    
def plot_size_distribution_func(Real_MN_pop, size_power, popt, a_size, c_size, r2, Calib_sizes, MN_pop):
    #Plotting the trendline of the distribution of the calibrated MN sizes
    plt.plot(np.arange(0,MN_pop,1), size_power(np.arange(0,MN_pop,1), *popt)*10**6, 'r', linewidth = 3)#, label=str(round(a_size*10**7,1))+r'$*10^{-7}*2.4^(i/N)^$'+str(round(c_size,1))+' ,'+r'$r^2 = $'+str(round(r2,2)))
    plt.scatter (Real_MN_pop, Calib_sizes*10**6, marker='x', c='k')
    plt.xlabel('MN population')
    ylabel='MN size S ' + r'$[mm^2]$'
    plt.ylabel(ylabel)
    plt.ylim(0.1, size_power(MN_pop, *popt)*10**6*1.05)
    plt.xlim(0, MN_pop)
    plt.title('Size distribution in real population')
    # plt.legend()
    plt.show()

    
# def plot_simulation_error_func(Nb_MN, MN_list, delta_tf1_end, nME_sim, RMS_table_sim, Corrcoef_table_sim):
#     plt.scatter(MN_list+1, delta_tf1_end, marker='x', c='k')
#     plt.plot([1,Nb_MN+1], [0.25,0.25], linestyle='--', c='k')
#     plt.plot([1,Nb_MN+1], [-0.25,-0.25], linestyle='--', c='k')
#     plt.xlabel('Identified MNs')
#     plt.xlim(1,Nb_MN+1)
#     plt.ylim(-1.5,1.5)
#     ylabel=r'$\Delta ft^1$' + ' [s]' 
#     plt.ylabel(ylabel)
#     plt.title('Replicating exp data - First discharge time')
#     plt.show()
    
#     plt.scatter(MN_list, nME_sim, marker='x', c='k')
#     plt.xlabel('Identified MNs')
#     plt.ylabel('nM error')
#     plt.ylim(0,100)
#     plt.title('nM error (%) between experimental and simulated virtual filtered FF')
#     plt.grid()
#     plt.show()
    
#     for i in range(len(MN_list)):
#         if RMS_table_sim[i]==None: RMS_table_sim[i]=RMS_table_sim[i-1]
#     plt.scatter(MN_list, RMS_table_sim, marker='x', c='k')
#     plt.xlabel('Identified MNs')
#     plt.ylabel('nRMSE (%)')
#     plt.plot([1,Nb_MN+1], [15,15], linestyle='--', c='k')
#     plt.xlim(1, Nb_MN+1)
#     plt.ylim(0,100)
#     plt.title('RMS error (%) between experimental and simulated virtual filtered FF')
#     plt.show()

#     for i in range(len(MN_list)):
#         if Corrcoef_table_sim[i]==None: Corrcoef_table_sim[i]=Corrcoef_table_sim[i-1]    
#     plt.scatter(MN_list, Corrcoef_table_sim**2, marker='x', c='k')
#     plt.plot([1,Nb_MN+1], [0.8, 0.8], linestyle='--', c='k')
#     plt.xlim(1, Nb_MN+1)
#     plt.ylim(0,1)
#     plt.xlabel('Identified MNs')
#     plt.ylabel(r'$r^2$')
#     plt.title('r2 between exp and sim virtual filtered FF')
#     plt.show()
    


def plot_final_results_func(Nb_MN, time, t_start, range_start, end_force, range_stop, common_control_exp, common_control_sim, common_input_exp, common_input_sim, Force):
    time_shaped = time[range_start:range_stop]
    norm_CC_exp = common_control_exp[range_start:range_stop]/max(common_control_exp)*1.12
    norm_CI_exp = common_input_exp[range_start:range_stop]/max(common_input_exp)*1.1
    if Nb_MN==14: k_cc, k_ci=0.9, 1.0
    elif Nb_MN == 27: k_cc, k_ci, end_force=1.05, 1.3, end_force*1.02
    else: k_cc, k_ci = 1.0, 1.1
    
    norm_CC_sim = common_control_sim/max(common_control_sim)*k_cc
    norm_CI_sim = common_input_sim[range_start:range_stop]/max(common_input_sim)*k_ci
    
    plt.plot(time_shaped, norm_CC_exp, linestyle='dotted', label ='common control  [0-4Hz] from the '+str(Nb_MN)+' recorded MNs')
    plt.plot(time_shaped, norm_CC_sim, linewidth=2, color='orange', label ='common control  [0-4Hz] from the 400 virtual MNs')
    plt.plot(time,(Force)/max(Force)*0.95,  '-g', linewidth=2, label='Experimental transducer force')#' Whole Muscle Force '+r'$F^{MT} (t)$' )
    plt.xlabel('Time [s]', color='k',fontsize=14)
    plt.ylabel('Normalized values', color='k',fontsize=14)
    plt.xlim(t_start,end_force)
    plt.ylim(0,1.2)
    plt.title('Approximation of the muscle neural drive (common control)')
    plt.legend()
    plt.show()

    plt.plot(time_shaped, norm_CI_exp, label ='common input  [0-10Hz] from the '+str(Nb_MN)+' recorded MNs')
    plt.plot(time_shaped, norm_CI_sim, label ='common input  [0-10Hz] from the 400 virtual MNs')
    plt.plot(time,(Force)/max(Force)*1,  '-g', linewidth=3, label='Experimental transducer force')#' Whole Muscle Force '+r'$F^{MT} (t)$' )
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized values')
    plt.xlim(t_start,end_force)
    plt.title('Approximation of the muscle neural drive (common input)')
    plt.ylim(0,1.2)
    plt.legend()
    plt.show()
    
def plot_pred_exp_FIDF_func(MN, time, range_start, range_stop, FF_filt_exp, FF_filt_sim):
    plt.plot (time[range_start:range_stop], FF_filt_exp, 'k', label='Experimental')
    plt.plot (time[range_start:range_stop], FF_filt_sim , label='LIF-simulated')
    plt.xlabel('Time [s]', color='k',fontsize=12)
    plt.ylabel('FIDFs', color='k',fontsize=12)
    plt.title('Experimental vs simulated filtered firing frequencies (MN '+str(MN) + ')', color='k',fontsize=12, fontweight='bold')
    plt.xlim(0, 30)
    plt.legend()
    plt.show()
    
def plot_exp_vs_sim_FIDF_func(time, range_start,FIDF_exp_test,  FIDF_sim_test, i, step):
    if step == 'calibration':
        plt.title('Exp vs predicted FIDFs at calibration step for MN n°' + str(i+1))
        c='tab:blue'
    if step == 'validation':
        plt.title('Exp vs predicted FIDFs at validation step for MN n°' + str(i+1)) 
        c='orange'
    plt.plot (time[range_start:len(FIDF_exp_test)-200], FIDF_sim_test[0:len(FIDF_exp_test)-200], color=c, label='Simulated')
    plt.plot (time[range_start:len(FIDF_exp_test)-200], FIDF_exp_test[0:len(FIDF_exp_test)-200],color='black', label='Experimental')

    plt.xlim(0, 30)
    plt.ylim(0, max(FIDF_exp_test)*1.1)
    plt.xlabel('Time [s]', color='k',fontsize=12)
    plt.ylabel('Filtered Discharge Frequencies [Hz]', color='k',fontsize=12)    
    plt.legend()
    plt.show()


def plot_calibrated_FIDF(ith_MN, time, FIDF_exp_list,  I,  t_start, t_stop, t_plateau_end, Calib_sizes, Cm_rec, Cm_derec, step_size, ARP_table,  kR, adapt_kR, kR_derec, fs=2048):
    Calib_FIDF = FF_filt_func(I,  t_start, t_stop, t_plateau_end, Calib_sizes[ith_MN], Cm_rec, Cm_derec, step_size, ARP_table[ith_MN],  kR, adapt_kR='n', kR_derec=kR_derec, fs=2048)[0]
    FIDF_exp = FIDF_exp_list[ith_MN]
    plt.plot(time[int(t_start*fs):int(t_stop*fs)], Calib_FIDF)
    plt.plot(time[int(t_start*fs):int(t_stop*fs)], FIDF_exp[int(t_start*fs):int(t_stop*fs)], 'k')
    plt.xlim(t_start, t_stop)
    plt.ylim(0, max(max(FIDF_exp), max(Calib_FIDF))*1.05)
    plt.title(str(ith_MN))
    plt.show()
    return Calib_FIDF
    
    
def plot_final_results_validation_func(Nb_MN, deltaf1_table_sim, nME_table_sim, RMS_table_sim, Corrcoef_table_sim, Size_distrib_matrix):
    Nb_MN=Nb_MN+2
    MN_list=np.arange(1,Nb_MN,1)
    plt.scatter(MN_list, deltaf1_table_sim, marker='x', c='k')
    plt.plot([0,Nb_MN], [0.25,0.25], linestyle='--', c='k')
    plt.plot([0,Nb_MN], [-0.25,-0.25], linestyle='--', c='k')
    plt.xlabel('Identified MNs')
    plt.xticks(np.arange(0, Nb_MN,2))
    plt.xlim(0,Nb_MN-1)
    plt.ylim(-1.5,1.5)
    ylabel=r'$\Delta ft^1$' + ' [s]' 
    plt.ylabel(ylabel)
    plt.title('Onset error (s) between experimental and simulated FIDFs')
    plt.show()
    
    if len(nME_table_sim)==1 :
        pass
    else:
        plt.scatter(MN_list, nME_table_sim/100, marker='x', c='k')
        plt.xlabel('Identified MNs')
        plt.ylabel('nM error / 100')
        plt.xlim(1,Nb_MN)
        plt.xticks(np.arange(1, Nb_MN,2))
        plt.title('nM error between experimental and simulated  FIDFs')
        plt.show()
    
    for i in range(len(MN_list)):
        if RMS_table_sim[i]==None: RMS_table_sim[i]=RMS_table_sim[i-1]
    plt.scatter(MN_list, RMS_table_sim, marker='x', c='k')
    plt.xlabel('Identified MNs')
    plt.ylabel('nRMSE (%)')
    plt.plot([0,Nb_MN], [20,20], linestyle='--', c='k')
    plt.xlim(0,Nb_MN-1)
    plt.ylim(0,100)
    plt.xticks(np.arange(0, Nb_MN,2))
    plt.title('RMS error (%) between experimental and simulated FIDFs')
    plt.show()

    for i in range(len(MN_list)):
        if Corrcoef_table_sim[i]==None: Corrcoef_table_sim[i]=Corrcoef_table_sim[i-1]    
    plt.scatter(MN_list, Corrcoef_table_sim, marker='x', c='k')
    plt.plot([0,Nb_MN], [0.8, 0.8], linestyle='--', c='k')
    plt.xlim(0,Nb_MN-1)
    # plt.xlim(0,20)
    plt.ylim(0,1)
    plt.xticks(np.arange(0, Nb_MN,2))
    plt.xlabel('Identified MNs')
    plt.ylabel('r2')
    plt.title('r2 between experimental and simulated FIDFs')
    plt.show()
    
    # try: 
    #     plt.scatter(MN_list, Size_distrib_matrix[:,0], marker='x', c='k')
    #     plt.xlabel('Identified MNs')
    #     plt.xlim(1,Nb_MN)
    #     plt.xticks(np.arange(1, Nb_MN,2))
    #     plt.ylabel('Predicted lowest MN size [m2]')
    #     plt.show()
    
    #     plt.scatter(MN_list, Size_distrib_matrix[:,1], marker='x', c='k')
    #     plt.xlim(1,Nb_MN)
    #     plt.xticks(np.arange(1, Nb_MN,2))
    #     plt.xlabel('Identified MNs')
    #     plt.ylabel('Power value in Size distribution')
    #     plt.show()
    # except: pass


def plot_onionskin_representation_func(time, Firing_times_sim, t_start, end_force, MN_pop, fs=2048):
    from IDF_MOD import IDF_func
    Firing_times_sim_sec= np.empty((len(Firing_times_sim),), dtype=object)
    smoothed_IDF_sim_sec= np.empty((len(Firing_times_sim),), dtype=object)
    stop_iter = int(MN_pop*0.8)
    for i in range (len(Firing_times_sim)):
        Firing_times_sim_sec[i] = (Firing_times_sim[i]*fs).astype(int)
    
    IDF_dt_sim, FF_FULL_sim = IDF_func(stop_iter, time, Firing_times_sim_sec[0:stop_iter], 0, end_force)
    for i in range(stop_iter):    
        smoothed_IDF_sim_sec[i] = np.poly1d(np.polyfit(time[FF_FULL_sim[i]>0].astype(float),IDF_dt_sim[i].astype(float),6))(time[FF_FULL_sim[i]>0].astype(float))
        if i%10==0:
            plt.plot(time[FF_FULL_sim[i]>0], smoothed_IDF_sim_sec[i], color=(i/stop_iter, 0.4,  (stop_iter-i)/stop_iter))# c=colors[i])
    plt.xlim(t_start, end_force)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.ylim(0, 25)
    plt.xlabel('Time [s]')
    plt.ylabel('FIDFs [Hz]')
    plt.title('FIDFs of completely reconstructed discharging MN population')
    plt.show()
