#%%
import numpy as np

import importlib                                               
import sys                                                      
#importlib.reload(sys.modules['utils.plot_functions']) 
from utils.plot_functions import *
from utils.load_pulsars import load_pulsars_fnc
from utils.my_units import *

#%%
freq_GW_ind = [0, 3, 10, 23] 
file_name_end = '_eps80'
BHpop_list = ['5_30_0_1_', '5_20_0_1_', '5_30_0_0.5_', '5_20_0_0.3_']
#BHpop_list = ['5_30_0_1_']
dfdlogF, cum_dist = load_results_pulsars(freq_GW_ind, BHpop_list, file_name_end)

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
BW = 1.4*GHz
conv_f = (erg/Second/CentiMeter**2)/(1.E-3*Jy*BW)

for i, i_fGW in enumerate(freq_GW_ind):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(conv_f*dfdlogF[i][0][:, 0], NBH*dfdlogF[i][0][:, 1], color = colors[i], linewidth=1, 
                ls=linestyles[0], label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    for i_BH in range(1, len(BHpop_list)):
        line,=ax.plot(conv_f*dfdlogF[i][i_BH][:, 0], NBH*dfdlogF[i][i_BH][:, 1], color = colors[i], linewidth=1, 
                      ls=linestyles[i_BH])

legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc='upper right', fontsize=14)
lines = []
for i_BH, BHp_name in enumerate(BHpop_list):
    if len(BHp_name)>9:
        max_spin = BHp_name[-4:-1]
    else:
        max_spin = BHp_name[-2:-1]
    lines.append(mlines.Line2D([], [], color='black', linestyle=linestyles[i_BH], label=BHp_name[2:4]+', '+max_spin))  ### fix this
legend2 = ax.legend(title='$M_{\mathrm{max}}\ [M_{\odot}], \chi_{\mathrm{max}}$',
                    handles=lines, loc='upper left', handletextpad=0.5, frameon=False, 
                    labelspacing=0.2, handlelength=1.5, fontsize=14)
ax.add_artist(legend)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(2.E-1,1E6); 
ax.set_ylim(1E-1,2.E4)
ax.grid()
ax.set_xlabel(r'$F_r\ [{\rm mJy}]$', fontsize=font_s); ax.set_ylabel(r'$dn_{F_r}/d\log F_r$', fontsize=font_s); 
ax.set_title('Radio flux distribution', fontsize=font_s);
ax.xaxis.set_major_formatter(FuncFormatter(log_format_func))
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
fig.savefig('figs/radio_dist_eps80.pdf', bbox_inches="tight")


#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=18
colors = sns.color_palette("mako", len(freq_GW_ind)) 

for i, i_fGW in enumerate(freq_GW_ind):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(conv_f*cum_dist[i][0][:, 0], NBH*cum_dist[i][0][:, 1], color = colors[i], linewidth=1, 
                  ls=linestyles[0], label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    for i_BH in range(1, len(BHpop_list)):
        line,=ax.plot(conv_f*cum_dist[i][i_BH][:, 0], NBH*cum_dist[i][i_BH][:, 1], color = colors[i], linewidth=1, 
                      ls=linestyles[i_BH])

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(2.E-1,1E6); ax.set_ylim(1.,1.E4)
ax.grid()
ax.axvline(1.5, linewidth=1, color='gray')
ax.set_xlabel(r'$F_{r,0}\ [{\rm mJy}]$', fontsize=font_s); ax.set_ylabel(r'$N_{\rm events}(F_r>F_{r, 0})$', fontsize=font_s); 
ax.set_title('Number of expected anomalous pulsars', fontsize=font_s);
ax.xaxis.set_major_formatter(FuncFormatter(log_format_func))
ax.yaxis.set_major_formatter(FuncFormatter(log_format_func))

fig.tight_layout()
fig.savefig('figs/nevents_radio_eps80.pdf', bbox_inches="tight")
# %%
