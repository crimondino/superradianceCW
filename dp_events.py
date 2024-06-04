#%%
import numpy as np
import importlib
import sys
from tqdm import tqdm

from utils.my_units import *

importlib.reload(sys.modules['utils.analytic_estimate_events'])
from utils.analytic_estimate_events import *
from utils.load_pulsars import load_pulsars_fnc
from superrad import ultralight_boson as ub
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
# Grid of values of M, a and t
MList = np.geomspace(Mmin, Mmax, 108) 
aList = np.linspace(0, 1, 103)

# Parameters used for Galactic BH distribution
RMW=20*kpc; tMW = 13.6*1E9*Year
#%%

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 
#%%

#%%
hList = np.geomspace(1E-27, 1E-24, 300)
dfdlogh = np.zeros((len(freq_GW), len(hList)))
Nevent = np.zeros((len(freq_GW)))
Ntot = 3000

for i_fGW in tqdm(range(len(freq_GW))):
    t_SR, t_GW, h_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, RMW)

    for i_h, hVal in enumerate(hList):    
        dfdlogh[i_fGW, i_h] = get_dfdlogh(t_SR, t_GW, h_peak, MList, aList, hVal)

    selh = (hList > h_UL[i_fGW])
    Nevent[i_fGW] = Ntot*np.trapz(dfdlogh[i_fGW][selh]/hList[selh], x=hList[selh])
#%%

for i_fGW in tqdm(range(len(freq_GW))):

    selh = (hList > h_UL[i_fGW])
    Nevent[i_fGW] = Ntot*np.trapz(dfdlogh[i_fGW][selh]/hList[selh], x=hList[selh])


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

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(10):
    line,=ax.plot(hList, Ntot*dfdlogh[i_fGW], label=str(round(freq_GW[i_fGW], 1))+' Hz')
    ax.axvline(x=h_UL[i_fGW], linewidth=1, alpha=0.8, c=line.get_color(), linestyle='dashed')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'h', fontsize=font_s); ax.set_ylabel(r'$dN/d\log h$', fontsize=font_s); 
ax.set_title('Number of events per unit strain, $N_{\mathrm{tot}} =$ '+str(Ntot), fontsize=font_s);
ax.legend(title='$f_{GW}$')
ax.set_ylim(0.0001,1000)
ax.grid()

fig.tight_layout()
fig.savefig('figs/dndlogh_pulsars_new.pdf', bbox_inches="tight")
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


