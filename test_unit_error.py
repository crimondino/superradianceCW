#%%
import numpy as np

import importlib                                               
import sys                                                      
#importlib.reload(sys.modules['utils.plot_functions']) 
from utils.plot_functions import *
from utils.load_pulsars import load_pulsars_fnc
from utils.my_units import *


#%%
def load_results_2(freqGWi_list, BHp_list, file_name_end):

    dfdlogh = []
    cum_dist = []

    for i, ni in enumerate(freqGWi_list):
        dfdlogh_temp = []
        cum_dist_fGW_temp = []
        for BHp_name in BHp_list:
            list_temp = np.load('data/disc_events/dndlogh_'+BHp_name+str(ni)+file_name_end+'.npy')
            dfdlogh_temp.append(list_temp[1:])

            cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )   
            for i_h in range(len(list_temp[1:, 0]-1)):
                cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
                cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
            cum_dist_fGW_temp.append(cum_dist_temp)

        dfdlogh.append(dfdlogh_temp)
        cum_dist.append(cum_dist_fGW_temp)

    return dfdlogh, cum_dist

#%%
freq_GW_ind = [43] #[0, 3, 10, 23]
file_name_end = '_eps87' #'_eps65_testlarge'
BHpop_list = ['5_30_0_1_', '5_20_0_0.3_']
dfdlogh, cum_dist = load_results(freq_GW_ind, BHpop_list, file_name_end)
dfdlogh_2, cum_dist_2 = load_results_2(freq_GW_ind, BHpop_list, file_name_end)

# %%
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
freq_GW = pulsars['F_GW'].to_numpy()
hUL = pulsars['upper_limits'].to_numpy()
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
    line,=ax.plot(dfdlogh_2[i][0][:, 0], NBH*dfdlogh_2[i][0][:, 1], color = 'k', linewidth=1, 
                ls=linestyles[0], label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    for i_BH in range(1, len(BHpop_list)):
        line,=ax.plot(dfdlogh[i][i_BH][:, 0], NBH*dfdlogh[i][i_BH][:, 1], color = colors[i], linewidth=1, 
                      ls=linestyles[i_BH])
        line,=ax.plot(dfdlogh_2[i][i_BH][:, 0], NBH*dfdlogh_2[i][i_BH][:, 1], color = 'k', linewidth=1, 
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
                    labelspacing=0.2, handlelength=1.5, fontsize=14)
ax.add_artist(legend)

ax.set_xscale('log'); ax.set_yscale('log')
#ax.set_xlim(5E-27,1.E-22); ax.set_ylim(-2,10)
#ax.set_xlim(5E-28,6E-24); ax.set_ylim(5,8.E3)
ax.grid()
ax.set_xlabel(r'$h_0$', fontsize=font_s); ax.set_ylabel(r'$dn_{h}/d\log h_0$', fontsize=font_s); 
ax.set_title('Signal strain distribution', fontsize=font_s);
#ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
# %%


#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
colors = sns.color_palette("mako", len(freq_GW_ind)) 

for i, i_fGW in enumerate(freq_GW_ind):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(cum_dist[i][0][:, 0], NBH*cum_dist[i][0][:, 1], color = colors[i], linewidth=1, 
                ls=linestyles[0], label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    line,=ax.plot(cum_dist_2[i][0][:, 0], NBH*cum_dist_2[i][0][:, 1], color = 'k', linewidth=1, 
                ls=linestyles[0], label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    for i_BH in range(1, len(BHpop_list)):
        line,=ax.plot(cum_dist[i][i_BH][:, 0], NBH*cum_dist[i][i_BH][:, 1], color = colors[i], linewidth=1, 
                      ls=linestyles[i_BH])
        line,=ax.plot(cum_dist_2[i][i_BH][:, 0], NBH*cum_dist_2[i][i_BH][:, 1], color = 'k', linewidth=1, 
                      ls=linestyles[i_BH])

ax.set_xscale('log'); ax.set_yscale('log')
#ax.set_xlim(6E-28,1E-24); ax.set_ylim(2,1.E4)
ax.grid()
ax.set_xlabel(r'$h_{0}^{95\%}$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(h_0>h_{0}^{95\%})$', fontsize=font_s); 
ax.set_title('Number of expected events', fontsize=font_s);
#ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
#fig.savefig('figs/nevents_eps80.pdf', bbox_inches="tight")

# %%
