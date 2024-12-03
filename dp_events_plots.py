#%%
import numpy as np

import importlib                                               
import sys                                                      
#importlib.reload(sys.modules['utils.plot_functions']) 
from utils.plot_functions import *
from utils.load_pulsars import load_pulsars_fnc
from utils.my_units import *

#%%
freq_GW_ind = [0, 3, 10, 23] #[0, 3, 10, 23]
file_name_end = '_eps80' #'_eps65_testlarge'
#file_name_end_testEM = '_eps80_testPEM_fdot'
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
#BHpop_list = ['5_30_0_1_']
dfdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)
#dfdlogh_testEM, cum_dist_testEM = load_results(freq_GW_ind, BHpop_list, file_name_end_testEM)

### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
freq_GW = pulsars.F_GW
hUL = pulsars.upper_limits
pipelines = pulsars['suggested_pipeline'].unique() 

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
colors = sns.color_palette("mako", len(freq_GW_ind)) 
linestyles = ['solid','dashed','dashdot','dotted']
NBH = 1.E8

for i, i_fGW in enumerate(freq_GW_ind):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(dfdlogh[i][0][:, 0], NBH*dfdlogh[i][0][:, 1], color = colors[i], linewidth=1, 
                ls=linestyles[0], label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    for i_BH in range(1, len(BHpop_list)):
        line,=ax.plot(dfdlogh[i][i_BH][:, 0], NBH*dfdlogh[i][i_BH][:, 1], color = colors[i], linewidth=1, 
                      ls=linestyles[i_BH])

legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc='upper left', fontsize=14)
lines = []
for i_BH, BHp_name in enumerate(BHpop_list):
    if len(BHp_name)>9:
        max_spin = BHp_name[-4:-1]
    else:
        max_spin = BHp_name[-2:-1]
    lines.append(mlines.Line2D([], [], color='black', linestyle=linestyles[i_BH], label=BHp_name[2:4]+', '+max_spin))  ### fix this
legend2 = ax.legend(title='$M_{\mathrm{max}}\ [M_{\odot}], \chi_{\mathrm{max}}$',
                    handles=lines, loc='upper right', handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1, fontsize=14)
ax.add_artist(legend)

ax.set_xscale('log'); ax.set_yscale('log')
#ax.set_xlim(5E-27,1.E-22); ax.set_ylim(-2,10)
ax.set_xlim(5E-28,6E-24); ax.set_ylim(5,8.E3)
ax.grid()
ax.set_xlabel(r'$h$', fontsize=font_s); ax.set_ylabel(r'$dn_{h}/d\log h$', fontsize=font_s); 
ax.set_title('Signal strain distribution', fontsize=font_s);
#ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
fig.savefig('figs/strain_dist_eps80.pdf', bbox_inches="tight")


#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
colors = sns.color_palette("mako", len(freq_GW_ind)) 

for i, i_fGW in enumerate(freq_GW_ind):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(cum_dist[i][0][:, 0], NBH*cum_dist[i][0][:, 1], color = colors[i], linewidth=1, 
                ls=linestyles[0], label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    for i_BH in range(1, len(BHpop_list)):
        line,=ax.plot(cum_dist[i][i_BH][:, 0], NBH*cum_dist[i][i_BH][:, 1], color = colors[i], linewidth=1, 
                      ls=linestyles[i_BH])

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(6E-28,1E-24); ax.set_ylim(2,1.E4)
ax.grid()
ax.set_xlabel(r'$h_{0}$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h>h_{0})$', fontsize=font_s); 
ax.set_title('Number of expected events', fontsize=font_s);
#ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
fig.savefig('figs/nevents_eps80.pdf', bbox_inches="tight")





#%%
### Error bar plot for the number of events decreasing epsilon  for narrowband
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
NBH = 1.E8
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
freq_GW = pulsars.F_GW
hUL = pulsars.upper_limits

log10eps_list = [9., 8.5, 8.]#[8., 8.5, 9.]
eps_labels = ['$10^{-9}$', '$10^{-8.5}$', '$10^{-8}$']
#freq_GW_ind = np.arange(42)
freq_GW_ind, = np.where(pulsars['suggested_pipeline'].to_numpy()=='narrowband') #np.arange(42)
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
colors = sns.color_palette("flare", 7)[::-1] #sns.color_palette("crest", len(log10eps_list)) 

for i_eps, log10eps in enumerate(log10eps_list):
    #eps = np.power(10, -log10eps)
    file_name_end = '_eps'+str(round(10*log10eps))
    fdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)
    nEV_list = np.zeros((len(freq_GW_ind), 1+len(BHpop_list)))

    for i, i_fGW in enumerate(freq_GW_ind):
        nEV_list[i, 0] = freq_GW.iloc[i_fGW]
        loghUL = [np.log10(hUL.iloc[i_fGW])] #[-27, -26.3, -26, -25] #
        for i_BH in range(len(BHpop_list)):
            sel_non_zero = cum_dist[i][i_BH][:, 1]>0
            #print(log10eps, i_fGW, len(cum_dist[i][i_BH][sel_non_zero, 0]))
            if len(cum_dist[i][i_BH][sel_non_zero, 0])>0:
                log_dist = np.array([np.log10(cum_dist[i][i_BH][sel_non_zero, 0]), np.log10(NBH*cum_dist[i][i_BH][sel_non_zero, 1])]).T
                [nEV_list[i, 1+i_BH]] = np.power(10, np.interp(loghUL, log_dist[:, 0], log_dist[:, 1], left=None, right=None, period=None))
            else:
                print(log10eps, i_fGW, freq_GW.iloc[i_fGW], BHpop_list[i_BH], ' has zero')
    
    y = (np.min(nEV_list[:, 1:], axis=1) + np.max(nEV_list[:, 1:], axis=1)) / 2  # Central value
    yerr = [y - np.min(nEV_list[:, 1:], axis=1), np.max(nEV_list[:, 1:], axis=1) - y]  # Asymmetric error bars
    # Create error bar plot
    ax.errorbar(nEV_list[:, 0], y, yerr=yerr, fmt='None', elinewidth=1.5,
                capsize=1.5, capthick=1.5, ecolor=colors[i_eps], label=eps_labels[i_eps])

lines = []
for i_eps, log10eps in enumerate(log10eps_list):
    lines.append(mlines.Line2D([], [], color=colors[i_eps], label=eps_labels[i_eps]))
legend2 = ax.legend(title='$\epsilon$', handles=lines[::-1], loc='upper right', handletextpad=0.5, frameon=True, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

def forward(x):
    return np.pi*Hz/(1.E-13*eV)*x 
def inverse(x):
    return x / (np.pi*Hz/(1.E-13*eV))  # Inverse of the scaling
ax_top = ax.secondary_xaxis('top', functions=(forward, inverse))
ax_top.set_xlabel(r'$m c^2\ [10^{-13}\ {\mathrm{eV}}]$', fontsize=font_s)

ax.grid()
#ax.set_xscale('log'); ax.set_xlim(100, 1000); ax.grid(which='minor', axis='x')
ax.set_yscale('log'); ax.set_ylim(1, 2E4)
ax.set_xlabel(r'$f_{\mathrm{GW}}\ [{\mathrm{Hz}}]$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h>h_{0}^{95\%})$', fontsize=font_s); 
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
#fig.savefig('figs/nevents_analysis_loweps.pdf', bbox_inches="tight")


#%%
### Error bar plot for the number of events increasing epsilon for narrowband
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
NBH = 1.E8
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
freq_GW = pulsars.F_GW
hUL = pulsars.upper_limits

log10eps_list = [8., 8.]#[8., 7.5, 7., 6.5]#[6.5, 7., 7.5, 8.]
eps_labels = ['$10^{-8}$', '$10^{-7.5}$', '$10^{-7}$',  '$10^{-6.5}$']
freq_GW_ind, = np.where(pulsars['suggested_pipeline'].to_numpy()=='narrowband') #np.arange(42)
#freq_GW_ind = [0, 3, 10, 23, 43]
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
colors = sns.color_palette("flare", 7)[::-1] #sns.color_palette("flare", len(log10eps_list)) 

for i_eps, log10eps in enumerate(log10eps_list):
    #eps = np.power(10, -log10eps)
    file_name_end = '_eps'+str(round(10*log10eps))
    print(file_name_end)
    fdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)
    nEV_list = np.zeros((len(freq_GW_ind), 1+len(BHpop_list)))

    for i, i_fGW in enumerate(freq_GW_ind):
        nEV_list[i, 0] = freq_GW.iloc[i_fGW]
        loghUL = [np.log10(hUL.iloc[i_fGW])] #[-27, -26.3, -26, -25] #
        for i_BH in range(len(BHpop_list)):
            sel_non_zero = cum_dist[i][i_BH][:, 1]>0
            #print(log10eps, i_fGW, len(cum_dist[i][i_BH][sel_non_zero, 0]))
            if len(cum_dist[i][i_BH][sel_non_zero, 0])>0:
                log_dist = np.array([np.log10(cum_dist[i][i_BH][sel_non_zero, 0]), np.log10(NBH*cum_dist[i][i_BH][sel_non_zero, 1])]).T
                [nEV_list[i, 1+i_BH]] = np.power(10, np.interp(loghUL, log_dist[:, 0], log_dist[:, 1], left=None, right=None, period=None))
            else:
                print(log10eps, i_fGW, freq_GW.iloc[i_fGW], BHpop_list[i_BH], ' has zero')
    
    y = (np.min(nEV_list[:, 1:], axis=1) + np.max(nEV_list[:, 1:], axis=1)) / 2  # Central value
    yerr = [y - np.min(nEV_list[:, 1:], axis=1), np.max(nEV_list[:, 1:], axis=1) - y]  # Asymmetric error bars
    # Create error bar plot
    ax.errorbar(nEV_list[:, 0], y, yerr=yerr, fmt='None', elinewidth=1.5,
                capsize=1.5, capthick=1.5, ecolor=colors[i_eps+2], label=eps_labels[i_eps])

lines = []
for i_eps, log10eps in enumerate(log10eps_list):
    lines.append(mlines.Line2D([], [], color=colors[i_eps+2], label=eps_labels[i_eps]))
legend2 = ax.legend(title='$\epsilon$', handles=lines, loc='upper right', handletextpad=0.5, frameon=True, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

def forward(x):
    return np.pi*Hz/(1.E-13*eV)*x 
def inverse(x):
    return x / (np.pi*Hz/(1.E-13*eV))  # Inverse of the scaling
ax_top = ax.secondary_xaxis('top', functions=(forward, inverse))
ax_top.set_xlabel(r'$m c^2\ [10^{-13}\ {\mathrm{eV}}]$', fontsize=font_s)

ax.grid()
#ax.set_xscale('log'); ax.set_xlim(100, 1000); ax.grid(which='minor', axis='x')
ax.set_yscale('log'); ax.set_ylim(1, 2E4)
ax.set_xlabel(r'$f_{\mathrm{GW}}\ [{\mathrm{Hz}}]$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h>h_{0}^{95\%})$', fontsize=font_s); 
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
fig.savefig('figs/nevents_analysis_higheps.pdf', bbox_inches="tight")


#%%
### Error bar plot for the number of events decreasing epsilon other pipelines
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
NBH = 1.E8
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
freq_GW = pulsars.F_GW
hUL = pulsars.upper_limits

log10eps_list = [9., 8.5, 8.]#[8., 8.5, 9.]
eps_labels = ['$10^{-9}$', '$10^{-8.5}$', '$10^{-8}$']
#freq_GW_ind = np.arange(42)
freq_GW_ind, = np.where(pulsars['suggested_pipeline'].to_numpy()!='narrowband') #np.arange(42)
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
colors = sns.color_palette("flare", 7)[::-1] #sns.color_palette("crest", len(log10eps_list)) 

for i_eps, log10eps in enumerate(log10eps_list):
    #eps = np.power(10, -log10eps)
    file_name_end = '_eps'+str(round(10*log10eps))
    fdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)
    nEV_list = np.zeros((len(freq_GW_ind), 1+len(BHpop_list)))

    for i, i_fGW in enumerate(freq_GW_ind):
        nEV_list[i, 0] = freq_GW.iloc[i_fGW]
        loghUL = [np.log10(hUL.iloc[i_fGW])] #[-27, -26.3, -26, -25] #
        for i_BH in range(len(BHpop_list)):
            sel_non_zero = cum_dist[i][i_BH][:, 1]>0
            #print(log10eps, i_fGW, len(cum_dist[i][i_BH][sel_non_zero, 0]))
            if len(cum_dist[i][i_BH][sel_non_zero, 0])>0:
                log_dist = np.array([np.log10(cum_dist[i][i_BH][sel_non_zero, 0]), np.log10(NBH*cum_dist[i][i_BH][sel_non_zero, 1])]).T
                [nEV_list[i, 1+i_BH]] = np.power(10, np.interp(loghUL, log_dist[:, 0], log_dist[:, 1], left=None, right=None, period=None))
            else:
                print(log10eps, i_fGW, freq_GW.iloc[i_fGW], BHpop_list[i_BH], ' has zero')
    
    y = (np.min(nEV_list[:, 1:], axis=1) + np.max(nEV_list[:, 1:], axis=1)) / 2  # Central value
    yerr = [y - np.min(nEV_list[:, 1:], axis=1), np.max(nEV_list[:, 1:], axis=1) - y]  # Asymmetric error bars
    # Create error bar plot
    ax.errorbar(nEV_list[:, 0], y, yerr=yerr, fmt='None', elinewidth=1.5,
                capsize=1.5, capthick=1.5, ecolor=colors[i_eps], label=eps_labels[i_eps])

lines = []
for i_eps, log10eps in enumerate(log10eps_list):
    lines.append(mlines.Line2D([], [], color=colors[i_eps], label=eps_labels[i_eps]))
legend2 = ax.legend(title='$\epsilon$', handles=lines[::-1], loc='upper right', handletextpad=0.5, frameon=True, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

def forward(x):
    return np.pi*Hz/(1.E-13*eV)*x 
def inverse(x):
    return x / (np.pi*Hz/(1.E-13*eV))  # Inverse of the scaling
ax_top = ax.secondary_xaxis('top', functions=(forward, inverse))
ax_top.set_xlabel(r'$m c^2\ [10^{-13}\ {\mathrm{eV}}]$', fontsize=font_s)

ax.grid()
#ax.set_xscale('log'); ax.set_xlim(100, 1000); ax.grid(which='minor', axis='x')
ax.set_yscale('log'); ax.set_ylim(1, 5E4)
ax.set_xlabel(r'$f_{\mathrm{GW}}\ [{\mathrm{Hz}}]$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h>h_{0}^{95\%})$', fontsize=font_s); 
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
fig.savefig('figs/nevents_analysis_loweps_app.pdf', bbox_inches="tight")


#%%
### Error bar plot for the number of events increasing epsilon other pipelines
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
NBH = 1.E8
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
freq_GW = pulsars.F_GW
hUL = pulsars.upper_limits

log10eps_list = [8., 7.5, 7., 6.5]#[6.5, 7., 7.5, 8.]
eps_labels = ['$10^{-8}$', '$10^{-7.5}$', '$10^{-7}$',  '$10^{-6.5}$']
#freq_GW_ind = np.arange(42)
freq_GW_ind, = np.where(pulsars['suggested_pipeline'].to_numpy()!='narrowband') #np.arange(42)
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
colors = sns.color_palette("flare", 7)[::-1] #sns.color_palette("flare", len(log10eps_list)) 

for i_eps, log10eps in enumerate(log10eps_list):
    #eps = np.power(10, -log10eps)
    file_name_end = '_eps'+str(round(10*log10eps))
    fdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)
    nEV_list = np.zeros((len(freq_GW_ind), 1+len(BHpop_list)))

    for i, i_fGW in enumerate(freq_GW_ind):
        nEV_list[i, 0] = freq_GW.iloc[i_fGW]
        loghUL = [np.log10(hUL.iloc[i_fGW])] #[-27, -26.3, -26, -25] #
        for i_BH in range(len(BHpop_list)):
            sel_non_zero = cum_dist[i][i_BH][:, 1]>0
            #print(log10eps, i_fGW, len(cum_dist[i][i_BH][sel_non_zero, 0]))
            if len(cum_dist[i][i_BH][sel_non_zero, 0])>0:
                log_dist = np.array([np.log10(cum_dist[i][i_BH][sel_non_zero, 0]), np.log10(NBH*cum_dist[i][i_BH][sel_non_zero, 1])]).T
                [nEV_list[i, 1+i_BH]] = np.power(10, np.interp(loghUL, log_dist[:, 0], log_dist[:, 1], left=None, right=None, period=None))
            else:
                print(log10eps, i_fGW, freq_GW.iloc[i_fGW], BHpop_list[i_BH], ' has zero')
    
    y = (np.min(nEV_list[:, 1:], axis=1) + np.max(nEV_list[:, 1:], axis=1)) / 2  # Central value
    yerr = [y - np.min(nEV_list[:, 1:], axis=1), np.max(nEV_list[:, 1:], axis=1) - y]  # Asymmetric error bars
    # Create error bar plot
    ax.errorbar(nEV_list[:, 0], y, yerr=yerr, fmt='None', elinewidth=1.5,
                capsize=1.5, capthick=1.5, ecolor=colors[i_eps+2], label=eps_labels[i_eps])

lines = []
for i_eps, log10eps in enumerate(log10eps_list):
    lines.append(mlines.Line2D([], [], color=colors[i_eps+2], label=eps_labels[i_eps]))
legend2 = ax.legend(title='$\epsilon$', handles=lines, loc='upper right', handletextpad=0.5, frameon=True, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

def forward(x):
    return np.pi*Hz/(1.E-13*eV)*x 
def inverse(x):
    return x / (np.pi*Hz/(1.E-13*eV))  # Inverse of the scaling
ax_top = ax.secondary_xaxis('top', functions=(forward, inverse))
ax_top.set_xlabel(r'$m c^2\ [10^{-13}\ {\mathrm{eV}}]$', fontsize=font_s)

ax.grid()
#ax.set_xscale('log'); ax.set_xlim(100, 1000); ax.grid(which='minor', axis='x')
ax.set_yscale('log'); ax.set_ylim(1, 5E4)
ax.set_xlabel(r'$f_{\mathrm{GW}}\ [{\mathrm{Hz}}]$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h>h_{0}^{95\%})$', fontsize=font_s); 
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
fig.savefig('figs/nevents_analysis_higheps_app.pdf', bbox_inches="tight")


#%%
### Individual line plot for the number of events
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
NBH = 1.E8
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
freq_GW = pulsars.F_GW
hUL = pulsars.upper_limits

log10eps_list = [7.5, 8., 8.5, 9.]
eps_labels = ['$10^{-7.5}$', '$10^{-8}$', '$10^{-8.5}$', '$10^{-9}$']
freq_GW_ind, = np.where(pulsars['suggested_pipeline'].to_numpy()=='narrowband') #np.arange(42)
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
colors = sns.color_palette("crest", len(log10eps_list)) 

for i_eps, log10eps in enumerate(log10eps_list):
    #eps = np.power(10, -log10eps)
    file_name_end = '_eps'+str(round(10*log10eps))
    fdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)
    nEV_list = np.zeros((len(freq_GW_ind), 2))

    for i, i_fGW in enumerate(freq_GW_ind):
        nEV_list[i, 0] = freq_GW.iloc[i_fGW]
        loghUL = [np.log10(hUL.iloc[i_fGW])] #[-27, -26.3, -26, -25] #
        i_BH = 2
        sel_non_zero = cum_dist[i][i_BH][:, 1]>0
        #print(log10eps, i_fGW, len(cum_dist[i][i_BH][sel_non_zero, 0]))
        if len(cum_dist[i][i_BH][sel_non_zero, 0])>0:
            log_dist = np.array([np.log10(cum_dist[i][i_BH][sel_non_zero, 0]), np.log10(NBH*cum_dist[i][i_BH][sel_non_zero, 1])]).T
            [nEV_list[i, 1]] = np.power(10, np.interp(loghUL, log_dist[:, 0], log_dist[:, 1], left=None, right=None, period=None))
        else:
            print(log10eps, i_fGW, freq_GW.iloc[i_fGW], BHpop_list[i_BH], ' has zero')
        
    ax.plot(nEV_list[:, 0], nEV_list[:, 1], 'go', marker='v', color=colors[i_eps], label=eps_labels[i_eps])

lines = []
for i_eps, log10eps in enumerate(log10eps_list):
    lines.append(mlines.Line2D([], [], color=colors[i_eps], label=eps_labels[i_eps]))
legend2 = ax.legend(title='$\epsilon$', handles=lines, loc='upper right', handletextpad=0.5, frameon=True, 
                    labelspacing=0.2, handlelength=1, fontsize=14)

def forward(x):
    return np.pi*Hz/(1.E-13*eV)*x 
def inverse(x):
    return x / (np.pi*Hz/(1.E-13*eV))  # Inverse of the scaling
ax_top = ax.secondary_xaxis('top', functions=(forward, inverse))
ax_top.set_xlabel(r'$m c^2\ [10^{-13}\ {\mathrm{eV}}]$', fontsize=font_s)

ax.grid()
#ax.set_xscale('log'); ax.set_xlim(100, 1000); ax.grid(which='minor', axis='x')
ax.set_yscale('log')
ax.set_xlabel(r'$f_{\mathrm{GW}}\ [{\mathrm{Hz}}]$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h>h_{0}^{95\%})$', fontsize=font_s); 
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
#fig.savefig('figs/nevents_analysis.pdf', bbox_inches="tight")



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
