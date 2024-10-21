#%%
import numpy as np

import importlib                                                 # *** COMMENT OUT ***
import sys                                                       # *** COMMENT OUT ***
importlib.reload(sys.modules['utils.plot_functions'])  # *** COMMENT OUT ***
from utils.plot_functions import *

from utils.my_units import *

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
freq_GW = [0, 3, 10, 23]
file_name_end = '_eps1Em8'
dfdlogh, cum_dist = load_results(freq_GW, ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_'], file_name_end)

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
colors = sns.color_palette("mako", ntot) 
NBH = 1.E8

for i_fGW in range(ntot):
    #if i_fGW%2==0:
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(dfdlogh[i_fGW][0][:, 0], NBH*dfdlogh[i_fGW][0][:, 1], color = colors[i_fGW], linewidth=1, label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$h$', fontsize=font_s); ax.set_ylabel(r'$dn_{h}/d\log h$', fontsize=font_s); 
ax.set_title('Signal strain distribution', fontsize=font_s);
legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc='lower left', fontsize=14)
ax.set_xlim(5E-28,1E-24); ax.set_ylim(9,6.E6)
ax.grid()

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'30 $M_{\odot}$, 1')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'20 $M_{\odot}$, 1')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'30 $M_{\odot}$, 0.5')
line_style4 = mlines.Line2D([], [], color='black', linestyle='dashdot', label=r'20 $M_{\odot}$, 0.3')
# Add the second legend for the linestyles
legend2 = ax.legend(title='$M_{\mathrm{max}}, \chi_{\mathrm{max}}$',
                    handles=[line_style1, line_style2, line_style3, line_style4], loc='upper right', handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1, fontsize=14)
# Add the first legend back to the plot
ax.add_artist(legend)
fig.tight_layout()
#fig.savefig('figs/strain_dist_disk_9M23.pdf', bbox_inches="tight")
#fig.savefig('figs/strain_dist_eps1Em8.pdf', bbox_inches="tight")


#%%
### Plot signal strain distribution dn/dlogh
ntot = 4
nlist = [0, 3, 10, 23]
dfdlogh = []
dfdlogh_lowmass = []
dfdlogh_lowspin = []
dfdlogh_pess = []
freq_GW = np.zeros(ntot)
NBH = 1.E8
file_name = 'dndlogh_'
file_name_end = '_eps1Em8'

for i, ni in enumerate(nlist):
    list_temp = np.load('data/disc_events/'+file_name+'5_30_0_1_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh.append(list_temp[1:])
    list_temp = np.load('data/disc_events/'+file_name+'5_20_0_1_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh_lowmass.append(list_temp[1:])
    list_temp = np.load('data/disc_events/'+file_name+'5_30_0_0.5_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh_lowspin.append(list_temp[1:])
    list_temp = np.load('data/disc_events/'+file_name+'5_20_0_0.3_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh_pess.append(list_temp[1:])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
colors = sns.color_palette("mako", ntot) 

for i_fGW in range(ntot):
    #if i_fGW%2==0:
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(dfdlogh[i_fGW][:, 0], NBH*dfdlogh[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    line,=ax.plot(dfdlogh_lowmass[i_fGW][:, 0], NBH*dfdlogh_lowmass[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashed', linewidth=1)
    line,=ax.plot(dfdlogh_lowspin[i_fGW][:, 0], NBH*dfdlogh_lowspin[i_fGW][:, 1], color = colors[i_fGW], linestyle='dotted', linewidth=1)
    line,=ax.plot(dfdlogh_pess[i_fGW][:, 0], NBH*dfdlogh_pess[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashdot', linewidth=1)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$h$', fontsize=font_s); ax.set_ylabel(r'$dn_{h}/d\log h$', fontsize=font_s); 
ax.set_title('Signal strain distribution', fontsize=font_s);
legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc='lower left', fontsize=14)
ax.set_xlim(5E-28,1E-24); ax.set_ylim(9,6.E6)
ax.grid()

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'30 $M_{\odot}$, 1')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'20 $M_{\odot}$, 1')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'30 $M_{\odot}$, 0.5')
line_style4 = mlines.Line2D([], [], color='black', linestyle='dashdot', label=r'20 $M_{\odot}$, 0.3')
# Add the second legend for the linestyles
legend2 = ax.legend(title='$M_{\mathrm{max}}, \chi_{\mathrm{max}}$',
                    handles=[line_style1, line_style2, line_style3, line_style4], loc='upper right', handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1, fontsize=14)
# Add the first legend back to the plot
ax.add_artist(legend)
fig.tight_layout()
#fig.savefig('figs/strain_dist_disk_9M23.pdf', bbox_inches="tight")
fig.savefig('figs/strain_dist_eps1Em8.pdf', bbox_inches="tight")

#%%
### Plot expected number of events as a function of the strain UL
ntot = 4
nlist = [0, 3, 10, 23]
cum_dist = []
cum_dist_lowmass = []
cum_dist_lowspin = []
cum_dist_pess = []
freq_GW = np.zeros(ntot)
obs_events, obs_events_lowmass, obs_events_lowspin, obs_events_pess = np.zeros((ntot, 2)), np.zeros((ntot, 2)), np.zeros((ntot, 2)), np.zeros((ntot, 2))
NBH = 1.E8
file_name = 'dndlogh_'
file_name_end = '_eps1Em8'

for i, ni in enumerate(nlist):
    list_temp = np.load('data/disc_events/'+file_name+'5_30_0_1_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist.append(cum_dist_temp)
    #obs_events[i, 0] = freq_GW[i]
    #obs_events[i, 1] = cum_dist_temp[(np.abs(cum_dist_temp[:, 0] - h_UL[i])).argmin(), 1]*NBH

    list_temp = np.load('data/disc_events/'+file_name+'5_20_0_1_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist_lowmass.append(cum_dist_temp)
    #obs_events_lowmass[i, 0] = freq_GW[i]
    #obs_events_lowmass[i, 1] = cum_dist_temp[(np.abs(cum_dist_temp[:, 0] - h_UL[i])).argmin(), 1]*NBH

    list_temp = np.load('data/disc_events/'+file_name+'5_30_0_0.5_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist_lowspin.append(cum_dist_temp)
    #obs_events_lowspin[i, 0] = freq_GW[i]
    #obs_events_lowspin[i, 1] = cum_dist_temp[(np.abs(cum_dist_temp[:, 0] - h_UL[i])).argmin(), 1]*NBH

    list_temp = np.load('data/disc_events/'+file_name+'5_20_0_0.3_'+str(ni)+file_name_end+'.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist_pess.append(cum_dist_temp)
    #obs_events_pess[i, 0] = freq_GW[i]
    #obs_events_pess[i, 1] = cum_dist_temp[(np.abs(cum_dist_temp[:, 0] - h_UL[i])).argmin(), 1]*NBH

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
colors = sns.color_palette("mako", ntot) 

for i_fGW in range(ntot):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(cum_dist[i_fGW][:, 0], cum_dist[i_fGW][:, 1]*NBH, color = colors[i_fGW], linewidth=1, label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    line,=ax.plot(cum_dist_lowmass[i_fGW][:, 0], cum_dist_lowmass[i_fGW][:, 1]*NBH, color = colors[i_fGW], linewidth=1, linestyle='dashed')
    line,=ax.plot(cum_dist_lowspin[i_fGW][:, 0], cum_dist_lowspin[i_fGW][:, 1]*NBH, color = colors[i_fGW], linewidth=1, linestyle='dotted')
    line,=ax.plot(cum_dist_pess[i_fGW][:, 0], cum_dist_pess[i_fGW][:, 1]*NBH, color = colors[i_fGW], linewidth=1, linestyle='dashdot')
    #ax2.plot(cum_dist[i_fGW][:, 0], cum_dist[i_fGW][:, 1]*facotr_y2, color = colors[i_fGW], linewidth=0)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$h_{0}^{95\%}$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h>h_{0}^{95\%})$', fontsize=font_s); 
ax.set_title('Number of expected events', fontsize=font_s);
ax.legend(title='$f_{\mathrm{GW}}\ [Hz]$', frameon=False, labelspacing=0.2, ncol=2, loc='lower left', fontsize=font_s)
ax.set_xlim(5E-28,5E-25); ax.set_ylim(10,5.E6)
#ax2.set_xlim(5E-28,5E-25); ax2.set_ylim(1E-7*facotr_y2,0.7*facotr_y2)
legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc='lower left', fontsize=14)
ax.grid()

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'30 $M_{\odot}$, 1')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'20 $M_{\odot}$, 1')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'30 $M_{\odot}$, 0.5')
line_style4 = mlines.Line2D([], [], color='black', linestyle='dashdot', label=r'20 $M_{\odot}$, 0.3')
# Add the second legend for the linestyles
legend2 = ax.legend(title='$M_{\mathrm{max}}, \chi_{\mathrm{max}}$',
                    handles=[line_style1, line_style2, line_style3, line_style4], loc='upper right', handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

# Add the first legend back to the plot
ax.add_artist(legend)

fig.tight_layout()
#fig.savefig('figs/strain_cumulative_disk_9M23.pdf', bbox_inches="tight")
fig.savefig('figs/nevents_eps1Em8.pdf', bbox_inches="tight")

#%%
### Plot number of expected events as a function of frequency
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("mako", 4) 
ax2 = ax.twiny()

ax.plot(obs_events[:, 0], obs_events[:, 1], color = colors[0], linewidth=1)
ax.plot(obs_events_lowmass[:, 0], obs_events_lowmass[:, 1], color = colors[0], linewidth=1, linestyle='dashed')
ax.plot(obs_events_lowspin[:, 0], obs_events_lowspin[:, 1], color = colors[0], linewidth=1, linestyle='dotted')
ax.plot(obs_events_pess[:, 0], obs_events_pess[:, 1], color = colors[0], linewidth=1, linestyle='dashdot')
ax2.plot(np.pi*obs_events[:, 0]*Hz/(1.E-13*eV), obs_events[:, 1], color = colors[0], linewidth=1)
#ax.grid()

#ax.set_xscale('log'); 
ax.set_yscale('log')
ax.set_xlabel(r'$f_{\mathrm{GW}}\ [Hz]$', fontsize=font_s)
ax.set_ylabel(r'$N_{\rm pulsars}(h>h_{\mathrm{UL}})$', fontsize=font_s)
ax2.set_xlabel(r'$\mu\ [10^{-13}\ \mathrm{eV}]$', fontsize=font_s)

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'30 $M_{\odot}$, 1')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'20 $M_{\odot}$, 1')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'30 $M_{\odot}$, 0.5')
line_style4 = mlines.Line2D([], [], color='black', linestyle='dashdot', label=r'20 $M_{\odot}$, 0.3')
# Add the second legend for the linestyles
legend2 = ax.legend(title='$M_{\mathrm{max}}, \chi_{\mathrm{max}}$',
                    handles=[line_style1, line_style2, line_style3, line_style4], loc='best', handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

ax.text(120, 2.E2, r'$\varepsilon = 3\times 10^{-9}$', fontsize=font_s)
ax.text(120, 1.E2, r'$f_{r} = 10^{-5}$', fontsize=font_s)
fig.tight_layout()
fig.savefig('figs/Nhevents_disc_9M23.pdf', bbox_inches="tight")
#%%



#%%
### Plot double-differential distribution df/dloghdnu

n_freq = [0, 1, 8]
xtable = []
ytable = []
dfdloghdnu = []
freq_GW = np.zeros(len(n_freq))

for i, i_f in enumerate(n_freq):
    list_temp = np.load('data/disc_events/dfdloghdnu_disc_NB_5_30_0_1_'+str(i_f)+'.npy')
    freq_GW[i] = list_temp[0, 2]
    dim1, dim2 = int(list_temp[0, 0]), int(list_temp[0, 1])
    #list_temp[1:, [0, 1]] = list_temp[1:, [1, 0]]
    #xtable.append(list_temp[1:, 0].reshape(dim2, dim1))
    #ytable.append(list_temp[1:, 1].reshape(dim2, dim1))
    #dfdloghdnu.append(list_temp[1:, 2].reshape(dim2, dim1))
    # Determine unique x and y values to create a mesh grid
    xtable.append(np.unique(list_temp[1:, 0]))
    ytable.append(np.unique(list_temp[1:, 1]))

    list_temp[1:, 2][list_temp[1:, 2]==0] = np.nan

    dfdloghdnu.append(list_temp[1:, 2].reshape(dim2, dim1))



# Create a density plot
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

for i in range(len(n_freq)):
    #contour = ax[i].contourf(xtable[i], ytable[i], np.log10(dfdloghdnu[i]), cmap='viridis')
    contour = ax[i].pcolormesh(xtable[i], ytable[i], np.log10(dfdloghdnu[i]), cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax[i])
    ax[i].set_title(r'$f_{\mathrm{GW}}\ [Hz]$ = '+str(round(freq_GW[i],1))+' Hz', fontsize=font_s)
    #cbar.set_label('log10(dfdloghdnu)')  # Optional: Set the colorbar label
    ax[i].set_xlabel(r'M [$M_{\odot}$]', fontsize=font_s); 
    ax[i].set_yscale('log')
ax[0].set_ylabel(r'h', fontsize=font_s);

fig.tight_layout()
fig.savefig('figs/double_diff_test.pdf', bbox_inches="tight")
#%%

#%%
### Plot for the scalar field to compare with sims results
ntot = 6
dfdlogh = []
cum_dist = []
cum_dist_masha = []
masha_filenames = ['data/scalar/Fig15_'+str(i)+'_2003_03359.txt' for i in [2, 5, 8, 15, 20, 35]]
Ntot = 1.E8
freq_GW = np.zeros(ntot)

for i in range(ntot):
    list_temp = np.load('data/disc_events/dfdlogh_disc_scalar_5_20_0_1_'+str(i)+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh.append(list_temp[1:])

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = Ntot*np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist.append(cum_dist_temp)

    cum_dist_masha.append(np.loadtxt(masha_filenames[i]))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("deep", ntot) 

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(ntot):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    #line,=ax.plot(dfdlogh[i_fGW][:, 0], dfdlogh[i_fGW][:, 1], color = colors[i_fGW], label=str(int(mu_temp)))
    line,=ax.plot(cum_dist[i_fGW][:, 0], cum_dist[i_fGW][:, 1], color = colors[i_fGW], label=str(round(mu_temp, 1)))
    line,=ax.plot(cum_dist_masha[i_fGW][:, 0]*1.E-26, cum_dist_masha[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashed', linewidth=2)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$h_0$', fontsize=font_s); ax.set_ylabel(r'Number of signals above $h_0$', fontsize=font_s); 
ax.set_title('Scalar field, comparison with Sylvia et al.', fontsize=font_s);
ax.legend(title='$\mu\ [10^{-13}\ {\mathrm eV}]$', frameon=False, labelspacing=0.2)
ax.set_xlim(5E-27,3E-24); ax.set_ylim(1, 5E6)
ax.grid()
ax.text(1.2E-25, 1.5E6, r'Sylvia (dashed)', fontsize=12)
ax.text(1.2E-25, 5E5, r'Mine (solid)', fontsize=12)
#ax.text(1E-30, 1200, r'$s_{\rm min} = 0,\ s_{\rm max} = 1$', fontsize=12)

fig.tight_layout()
fig.savefig('figs/scalar_comp.pdf', bbox_inches="tight")
#%%

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("deep", 4) 

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(4):
    #line,=ax.plot(hList, Ntot*dfdlogh_center_disc[i_fGW], linestyle='dashed', color = colors[i_fGW])
    line,=ax.plot(hList, Ntot*dfdlogh_disc[i_fGW], color = colors[i_fGW], label=str(round(freq_GW[i_fGW], 1))+' Hz')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'h', fontsize=font_s); ax.set_ylabel(r'$dN/d\log h$', fontsize=font_s); 
ax.set_title('Galactic disc, Earth observer, $N_{\mathrm{tot}} =$ '+str(Ntot), fontsize=font_s);
ax.legend(title='$f_{\rm GW}$')
ax.set_ylim(0.01,1E4)
ax.grid()
ax.text(1E-30, 3000, r'$M_{\rm min} = 5\ M_{\odot},\ M_{\rm max} = 30\ M_{\odot}$', fontsize=12)
ax.text(1E-30, 1200, r'$s_{\rm min} = 0,\ s_{\rm max} = 1$', fontsize=12)

fig.tight_layout()
fig.savefig('figs/dndlogh_disc.pdf', bbox_inches="tight")
#%%

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(1):
    line,=ax.plot(hList, Ntot*dfdlogh_center_disc[i_fGW], linestyle='dashed', label='GC observer') #label=str(round(freq_GW[i_fGW], 1))+' Hz')
    line,=ax.plot(hList, Ntot*dfdlogh_disc[i_fGW], label='Earth observer')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'h', fontsize=font_s); ax.set_ylabel(r'$dN/d\log h$', fontsize=font_s); 
#ax.set_title('Number of events per unit strain, $N_{\mathrm{tot}} =$ '+str(Ntot), fontsize=font_s);
ax.set_title('Galactic disc spatial distribution, $f_{\rm GW}$ = '+str(round(freq_GW[i_fGW], 1))+' Hz');
#ax.legend(title='$f_{\rm GW}$')
ax.legend()
ax.set_ylim(0.01,1E4)
ax.grid()
ax.text(2E-31, 3000, r'$M_{\rm min} = 5\ M_{\odot}, $')
ax.text(1E-29, 3000, r'$M_{\rm max} = 30\ M_{\odot}$')
ax.text(2E-31, 1000, r'$s_{\rm min} = 0, $')
ax.text(3E-30, 1000, r'$s_{\rm max} = 1$')

fig.tight_layout()
fig.savefig('figs/dndlogh_disc_comp.pdf', bbox_inches="tight")
#%%

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
colorlist =  ['darkslateblue', 'darkred', 'orangered', 'green'] #['teal', 'orange', 'indianred']
liestyle_list = ['solid', 'dashed', 'dashdot']
font_s=16

ax.plot(freq_GW, Nevent, 's', marker='o')

ax.set_title('N events above UL for different pulsars, $N_{\mathrm{tot}} =$ '+str(Ntot),fontsize=font_s);
ax.set_xlabel(r'$f_{\mathrm{GW}}$ [Hz]', fontsize=font_s); ax.set_ylabel(r'$N_{\mathrm{ev}}(h>h_{\mathrm{UL}})$', fontsize=font_s); 
ax.set_yscale('log')
#ax.grid()
#ax.set_xlim(100,300); #ax.set_ylim(0,10)

fig.tight_layout()
fig.savefig('figs/Nexpected_new.pdf', bbox_inches="tight")
#%%