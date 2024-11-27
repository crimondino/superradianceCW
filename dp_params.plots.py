#%%
import numpy as np

import importlib                                               
import sys                                                      
importlib.reload(sys.modules['utils.plot_functions']) 
from utils.plot_functions import *
from utils.load_pulsars import load_pulsars_fnc
from utils.my_units import *

# %%

### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
freq_GW = pulsars.F_GW
hUL = pulsars.upper_limits
NBH = 1.E8

log10eps_list = np.array([9.7, 9.5, 9.3, 9., 8.7, 8.5, 8., 7.5, 7., 6.5])
freq_GW_ind = np.arange(42) #freq_GW_ind, = np.where(pulsars['suggested_pipeline'].to_numpy()=='narrowband')
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
colors = sns.color_palette("crest", len(log10eps_list)) 
nEV_list = np.zeros((len(log10eps_list), len(freq_GW_ind), 1+len(BHpop_list)))

for i_eps, log10eps in enumerate(log10eps_list):
    #eps = np.power(10, -log10eps)
    file_name_end = '_eps'+str(round(10*log10eps))
    fdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)

    for i, i_fGW in enumerate(freq_GW_ind):
        nEV_list[i_eps, i, 0] = freq_GW.iloc[i_fGW]
        loghUL = [np.log10(hUL.iloc[i_fGW])] 
        for i_BH in range(len(BHpop_list)):
            sel_non_zero = cum_dist[i][i_BH][:, 1]>0
            #print(log10eps, i_fGW, len(cum_dist[i][i_BH][sel_non_zero, 0]))
            if len(cum_dist[i][i_BH][sel_non_zero, 0])>0:
                log_dist = np.array([np.log10(cum_dist[i][i_BH][sel_non_zero, 0]), np.log10(NBH*cum_dist[i][i_BH][sel_non_zero, 1])]).T
                [nEV_list[i_eps, i, 1+i_BH]] = np.power(10, np.interp(loghUL, log_dist[:, 0], log_dist[:, 1], left=None, right=None, period=None))



#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
min_lognev = -1.; max_lognev = 4.
logNev_th = 1. #np.log10(3.)
dp_limits = np.loadtxt('data/DPlimits.txt')
dp_limits_Planck = np.loadtxt('data/DPlimits_Planck.txt')
ax.fill_between(dp_limits_Planck[:, 0], dp_limits_Planck[:, 1], 1, color='k', alpha=0.3)    
ax.fill_between(dp_limits[:, 0], dp_limits[:, 1], 1, color='k', alpha=0.2)    

for i, i_fGW in enumerate(freq_GW_ind):
    # Define the x-position for the vertical line
    x_pos = freq_GW.iloc[i_fGW]*np.pi*Hz/(eV)
    y_values = np.geomspace(10**(-9.5), 10**(-6.5), 100)
    color_values = np.interp(np.log10(y_values), -log10eps_list, np.log10(np.min(nEV_list[:, i, 1:], axis=1)), left=None, right=None, period=None)
    #y_values = [np.power(10, -log10eps_list[j]) for j in range(len(log10eps_list))]
    #color_values = np.log10(np.min(nEV_list[:, i, 1:], axis=1), where=(np.min(nEV_list[:, i, 1:], axis=1)>0), out=(np.zeros(len(log10eps_list))-2.))  # You can replace this with your own array of values
    #segments = np.array([[[x_pos, y_values[i]], [x_pos, y_values[i+1]]] for i in range(len(y_values) - 1)])

    segments = []
    segment_colors = []
    for j in range(len(y_values) - 1):
        # Only include segments where color value meets the threshold
        if color_values[j] >= logNev_th:
            segments.append([[x_pos, y_values[j]], [x_pos, y_values[j+1]]])
            segment_colors.append(color_values[j])

    if segments:
        lc = LineCollection(segments, cmap="viridis", norm=Normalize(vmin=-min_lognev, vmax=max_lognev))
        lc.set_array(np.array(segment_colors))  # Set the values for colormap scaling
        ax.add_collection(lc)

ax.set_xlim(1E-13, 3E-12)
ax.set_ylim(1E-10, 1E-6)
ax.set_xscale('log'); 
ax.set_yscale('log')
ax.text(1.05E-13, 6E-7, r'COBE/', fontsize=12)
ax.text(1.05E-13, 4.3E-7, r'FIRAS', fontsize=12)
ax.text(1.6E-12, 2E-7, r'Planck+', fontsize=12)
ax.text(1.6E-12, 1.4E-7, r'unWISE', fontsize=12)

plt.colorbar(lc, ax=ax, label=r'$\log_{10}(N_{\rm events})$',shrink=0.6)  # Colorbar to show the value scaling
plt.xlabel(r'$m c^2\ [{\mathrm{eV}}]$', fontsize=font_s)
plt.ylabel(r'$\varepsilon$', fontsize=font_s)
plt.title("Pessimistic BH population")
plt.show()

fig.tight_layout()
#fig.savefig('figs/meps_pessimistic.pdf', bbox_inches="tight")

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
min_lognev = -1.; max_lognev = 4.
logNev_th = 1. #np.log10(3.)
dp_limits = np.loadtxt('data/DPlimits.txt')
dp_limits_Planck = np.loadtxt('data/DPlimits_Planck.txt')
ax.fill_between(dp_limits_Planck[:, 0], dp_limits_Planck[:, 1], 1, color='k', alpha=0.3)    
ax.fill_between(dp_limits[:, 0], dp_limits[:, 1], 1, color='k', alpha=0.2)    

for i, i_fGW in enumerate(freq_GW_ind):
    # Define the x-position for the vertical line
    x_pos = freq_GW.iloc[i_fGW]*np.pi*Hz/(eV)
    y_values = np.geomspace(10**(-9.5), 10**(-6.5), 100)
    color_values = np.interp(np.log10(y_values), -log10eps_list, np.log10(np.max(nEV_list[:, i, 1:], axis=1)), left=None, right=None, period=None)
    #y_values = [np.power(10, -log10eps_list[j]) for j in range(len(log10eps_list))]
    #color_values = np.log10(np.max(nEV_list[:, i, 1:], axis=1), where=(np.max(nEV_list[:, i, 1:], axis=1)>0), out=(np.zeros(len(log10eps_list))-2.))  # You can replace this with your own array of values
    #segments = np.array([[[x_pos, y_values[i]], [x_pos, y_values[i+1]]] for i in range(len(y_values) - 1)])

    segments = []
    segment_colors = []
    for j in range(len(y_values) - 1):
        # Only include segments where color value meets the threshold
        if color_values[j] >= logNev_th:
            segments.append([[x_pos, y_values[j]], [x_pos, y_values[j+1]]])
            segment_colors.append(color_values[j])

    if segments:
        lc = LineCollection(segments, cmap="viridis", norm=Normalize(vmin=-min_lognev, vmax=max_lognev))
        lc.set_array(np.array(segment_colors))  # Set the values for colormap scaling
        ax.add_collection(lc)

ax.set_xlim(1E-13, 3E-12)
ax.set_ylim(1E-10, 1E-6)
ax.set_xscale('log'); 
ax.set_yscale('log')
ax.text(1.05E-13, 6E-7, r'COBE/', fontsize=12)
ax.text(1.05E-13, 4.3E-7, r'FIRAS', fontsize=12)
ax.text(1.6E-12, 2E-7, r'Planck+', fontsize=12)
ax.text(1.6E-12, 1.4E-7, r'unWISE', fontsize=12)

plt.colorbar(lc, ax=ax, label=r'$\log_{10}(N_{\rm events})$',shrink=0.6)  # Colorbar to show the value scaling
plt.xlabel(r'$m c^2\ [{\mathrm{eV}}]$', fontsize=font_s)
plt.ylabel(r'$\varepsilon$', fontsize=font_s)
plt.title("Optimistic BH population")
plt.show()

fig.tight_layout()
fig.savefig('figs/meps_optimistic.pdf', bbox_inches="tight")

# %%
