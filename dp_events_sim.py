#%%
import numpy as np
import importlib
import sys
import os 
import pandas as pd

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

#%%
mu = np.pi*freq_GW[0]*Hz/eV
sel_sim =sim_name[np.argmin(np.abs(1-mu_list/mu))]
#%%

#%%
#for i_fGW in tqdm(range(len(freq_GW))):
#    mu = np.pi*freq_GW[i_fGW]*Hz




sim_data = np.loadtxt(SIM_DIR+'/Mmax20_spinMax0p5/'+sel_sim)

# 3 -> d(kpc), 7 -> age_BH(Gyr), 10 -> MBH_i, 11 -> chi_i 
df_sim = pd.DataFrame(sim_data[:,[3, 7, 10, 11]], columns=['d(kpc)', 'age_BH(Gyr)', 'MBH_i', 'chi_i'])

#%%

#%%
h_sim = np.zeros((len(df_sim)))

for i_s in range(len(df_sim)):

    mbh = df_sim['MBH_i'].iloc[i_s]
    alpha = GN*mu*mbh*MSolar
    abh0 = GN*mu*df_sim['chi_i'].iloc[i_s]*MSolar
    dBH = df_sim['d(kpc)'].iloc[i_s]
    age_BH = df_sim['age_BH(Gyr)'].iloc[i_s]

    try:
        wf = bc.make_waveform(mbh, abh0, alpha, units="physical+alpha")
        if wf.azimuthal_num()==1:
            tSR = wf.cloud_growth_time()*Second
            h_sim[i_s] = wf.strain_char( (age_BH*1E9*Year - tSR - dBH*kpc)/Second , dObs=(dBH*kpc/Mpc) )
        else:
            h_sim[i_s] = np.nan
    except ValueError:
        h_sim[i_s] = np.nan
        pass 

#%%