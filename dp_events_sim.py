#%%
import numpy as np
import importlib
import sys
import os 
import pandas as pd
from tqdm import tqdm

from utils.my_units import *

importlib.reload(sys.modules['utils.sim_estimate_events'])
from utils.sim_estimate_events import *
from utils.load_pulsars import load_pulsars_fnc
from superrad import ultralight_boson as ub

SIM_DIR = '/Users/crimondino/Dropbox (PI)/myplasma/galactic_BH_sim'
#%%

#%%
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print(len(pulsars))
test, pulsars_ind = np.unique(np.round(pulsars['F_GW'].to_numpy(), -1), return_index=True)
freq_GW = (pulsars['F_GW'].iloc[pulsars_ind]).to_numpy()
h_UL = (pulsars['upper_limits'].iloc[pulsars_ind]).to_numpy()
print(len(freq_GW))
#%%

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 
#%%

#%%
sim_params = '/Mmax20_spinMax0p5'
sim_name =  os.listdir(SIM_DIR+sim_params)
mu_list = np.zeros(len(sim_name))
for i_f, file_name in enumerate(sim_name):
    mu_list[i_f] = float(file_name[:8])
#mu_list = np.sort(mu_list)

#%%
h_sim = np.zeros((len(freq_GW), 137838))
print(h_sim.shape, h_sim[0].shape)
#%%

#%%
for i_fGW in tqdm(range(len(freq_GW))):
    mu = np.pi*freq_GW[i_fGW]*Hz
    sel_sim =sim_name[np.argmin(np.abs(1-mu_list/(mu/eV) ))]
    sim_data = np.loadtxt(SIM_DIR+'/Mmax20_spinMax0p5/'+sel_sim)

    # 3 -> d(kpc), 7 -> age_BH(Gyr), 10 -> MBH_i, 11 -> chi_i 
    df_sim = pd.DataFrame(sim_data[:,[3, 7, 10, 11]], columns=['d(kpc)', 'age_BH(Gyr)', 'MBH_i', 'chi_i'])

    for i_s in range(len(df_sim)):

        mbh = df_sim['MBH_i'].iloc[i_s]; alpha = GN*mu*mbh*MSolar
        abh0 = df_sim['chi_i'].iloc[i_s]
        dBH = df_sim['d(kpc)'].iloc[i_s]
        age_BH = df_sim['age_BH(Gyr)'].iloc[i_s]

        try:
            wf = bc.make_waveform(mbh, abh0, alpha, units="physical+alpha")
            if wf.azimuthal_num()==1:
                tSR = wf.cloud_growth_time()*Second
                h_sim[i_fGW, i_s] = wf.strain_char( (age_BH*1E9*Year - tSR - dBH*kpc)/Second , dObs=(dBH*kpc/Mpc) )
            else:
                h_sim[i_fGW, i_s] = np.nan
        except ValueError:
            h_sim[i_fGW, i_s] = np.nan
            pass 

#%%

#%%
sim_size = np.zeros(len(sim_name))
for i_f, file_name in enumerate(sim_name):
    sim_size[i_f] = np.loadtxt(SIM_DIR+'/Mmax20_spinMax0p5/'+file_name).shape[0]

np.max(sim_size)
#%%


############## Plots ##############
#%%
import matplotlib.pyplot as plt
plt.rcdefaults()
from matplotlib import font_manager
from matplotlib import rcParams
rcParams['mathtext.rm'] = 'Times New Roman' 
rcParams['text.usetex'] = True
rcParams['font.family'] = 'times' #'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':14})
#%%

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
h_bins = np.geomspace(1e-30, 1e-24, 30)
Ntot = 3000

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(0, 4):
    h_temp = h_sim[i_fGW][h_sim[i_fGW]>0]
#    n_hist, b_hist, p_hist = ax.hist(h_temp, bins=h_bins, density=False, weights=np.full(len(h_temp), 1000/len(h_temp)),
#                                     histtype='step', label=str(round(freq_GW[i_fGW], 1))+' Hz');
    n_hist, b_hist = np.histogram(h_temp, bins=h_bins, density=True) #weights=np.full(len(h_temp), 1000/len(h_temp)))
    ax.plot(b_hist[:-1], Ntot*n_hist*b_hist[:-1], marker='.', label=str(round(freq_GW[i_fGW], 1))+' Hz')
    #ax.axvline(x=h_UL[i_fGW], linewidth=1, alpha=0.8, c=line.get_color(), linestyle='dashed')

ax.set_xscale('log'); ax.set_yscale('log')
#ax.set_xlabel(r'h', fontsize=font_s); ax.set_ylabel(r'$dN/d\log h$', fontsize=font_s); 
#ax.set_title('Number of events per unit strain, $N_{\mathrm{tot}} =$ '+str(Ntot), fontsize=font_s);
#ax.legend(title='$f_{GW}$')
ax.set_xlim(1e-27,1e-24);ax.set_ylim(0.0001,1000)
#ax.grid()

fig.tight_layout()
#fig.savefig('figs/dndlogh_pulsars_new.pdf', bbox_inches="tight")
# %%
