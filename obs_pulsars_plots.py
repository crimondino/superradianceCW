#%%
import numpy as np
#import sys
#from tqdm import tqdm
#import time

from utils.my_units import *

#import importlib
#importlib.reload(sys.modules['utils.analytic_estimate_pulsars'])
#from utils.analytic_estimate_pulsars import *
#from utils.load_pulsars import load_pulsars_fnc
#%%


############## Plots ##############
#%%
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib.lines as mlines
rcParams['mathtext.rm'] = 'Times New Roman' 
rcParams['text.usetex'] = True
rcParams['font.family'] = 'times' #'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':14})
#%%

#%%
### Plot flux density distribution df/dlogF
ntot = 11
dfdlogF = []
dfdlogF_lowmass = []
dfdlogF_lowspin = []
dfdlogF_pess = []
freq_GW = np.zeros(ntot)
BW = 1.4*GHz
unit_conv = (erg/Second/CentiMeter**2)/BW/(1.E-3*Jy)

for i in range(ntot):
    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_30_0_1_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogF.append(list_temp[1:])
    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_20_0_1_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogF_lowmass.append(list_temp[1:])
    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_30_0_0.5_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogF_lowspin.append(list_temp[1:])
    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_20_0_0.3_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogF_pess.append(list_temp[1:])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("mako", ntot) 

for i_fGW in range(ntot):
    if i_fGW%2==0:
        mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
        line,=ax.plot(dfdlogF[i_fGW][:, 0]*unit_conv, dfdlogF[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
        line,=ax.plot(dfdlogF_lowmass[i_fGW][:, 0]*unit_conv, dfdlogF_lowmass[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashed', linewidth=1)
        line,=ax.plot(dfdlogF_lowspin[i_fGW][:, 0]*unit_conv, dfdlogF_lowspin[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashed', linewidth=1)
        line,=ax.plot(dfdlogF_pess[i_fGW][:, 0]*unit_conv, dfdlogF_pess[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashdot', linewidth=1)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\bar{F}$ [mJy]', fontsize=font_s); ax.set_ylabel(r'$df_{h}/d\log \bar{F}$', fontsize=font_s); 
ax.set_title('Signal flux density distribution', fontsize=font_s);
legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc=(0.01, 0),  fontsize=14)
ax.set_xlim(0.014,100); ax.set_ylim(3E-6,0.3)
#ax.set_ylim(3E-6, 1); ax2.set_ylim(3E-6*facotr_y2, 1*facotr_y2)
ax.grid()

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'30 $M_{\odot}$, 1')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'20 $M_{\odot}$, 1')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'30 $M_{\odot}$, 0.5')
line_style4 = mlines.Line2D([], [], color='black', linestyle='dashdot', label=r'20 $M_{\odot}$, 0.3')
# Add the second legend for the linestyles
legend2 = ax.legend(title='$M_{\mathrm{max}}, \chi_{\mathrm{max}}$',
                    handles=[line_style1, line_style2, line_style3, line_style4], loc=(0.01, 0.25), handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

# Add the first legend back to the plot
ax.add_artist(legend)

fig.tight_layout()
fig.savefig('figs/flux_dist_disc_9M23.pdf', bbox_inches="tight")
#%%

#%%
### Plot flux density cumulative distribution \int df/dlogF : *** needs to be edited ***
ntot = 11
cum_dist = []
cum_dist_lowmass = []
cum_dist_lowspin = []
cum_dist_pess = []
freq_GW = np.zeros(ntot)
BW = 1.4*GHz
unit_conv = (erg/Second/CentiMeter**2)/BW/(1.E-3*Jy)

for i in range(ntot):
    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_30_0_1_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist.append(cum_dist_temp)

    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_20_0_1_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist_lowmass.append(cum_dist_temp)

    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_30_0_0.5_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist_lowspin.append(cum_dist_temp)

    list_temp = np.load('data/pulsars_events/dfdlogF_disc_NB_5_20_0_0.3_'+str(i)+'_9M23.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist_pess.append(cum_dist_temp)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("mako", ntot) 
# Create a second y-axis that shares the same x-axis
ax2 = ax.twinx()
facotr_y2 = 0.5E8

for i_fGW in range(ntot):
    if i_fGW%2==0:
        mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
        line,=ax.plot(cum_dist[i_fGW][:, 0]*unit_conv, cum_dist[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
        line,=ax.plot(cum_dist_lowmass[i_fGW][:, 0]*unit_conv, cum_dist_lowmass[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, linestyle='dashed')
        line,=ax.plot(cum_dist_lowspin[i_fGW][:, 0]*unit_conv, cum_dist_lowspin[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, linestyle='dotted')
        line,=ax.plot(cum_dist_pess[i_fGW][:, 0]*unit_conv, cum_dist_pess[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, linestyle='dashdot')
        ax2.plot(cum_dist[i_fGW][:, 0]*unit_conv, cum_dist[i_fGW][:, 1]*facotr_y2, color = colors[i_fGW], linewidth=0)
        
ax.set_xscale('log'); ax.set_yscale('log')
ax2.set_yscale('log'); ax2.set_yscale('log')
ax.set_xlabel(r'$\bar{F}$ [mJy]', fontsize=font_s); ax.set_ylabel(r'$P(F>\bar{F})$', fontsize=font_s); 
ax2.set_ylabel(r'$N_{\rm pulsars}(F>\bar{F})$', fontsize=font_s)
ax.set_title('Cumulative signal flux density distribution', fontsize=font_s);
ax.legend(title='$f_{\mathrm{GW}}\ [Hz]$', frameon=False, labelspacing=0.2, ncol=2, loc='lower left', fontsize=14)
ax.set_xlim(0.014,100); ax2.set_xlim(0.014,100); 
ax.set_ylim(3E-6, 1); ax2.set_ylim(3E-6*facotr_y2, 1*facotr_y2)
legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc=(0.01, 0), fontsize=14)
ax.grid()

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'30 $M_{\odot}$, 1')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'20 $M_{\odot}$, 1')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'30 $M_{\odot}$, 0.5')
line_style4 = mlines.Line2D([], [], color='black', linestyle='dashdot', label=r'20 $M_{\odot}$, 0.3')
# Add the second legend for the linestyles
legend2 = ax.legend(title='$M_{\mathrm{max}}, \chi_{\mathrm{max}}$',
                    handles=[line_style1, line_style2, line_style3, line_style4], loc=(0.01, 0.25), handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1, fontsize=14)
ax.axvline(x=1.5, color='black', linewidth=0.8)

# Add the first legend back to the plot
ax.add_artist(legend)

fig.tight_layout()
fig.savefig('figs/flux_cumulative_disc_9M23.pdf', bbox_inches="tight")
#%%