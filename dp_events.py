#%%
import numpy as np
import importlib
import sys
from tqdm import tqdm
import time

from utils.my_units import *

importlib.reload(sys.modules['utils.analytic_estimate_events'])
importlib.reload(sys.modules['utils.load_pulsars'])
from utils.analytic_estimate_events import *
from utils.load_pulsars import load_pulsars_fnc
from superrad import ultralight_boson as ub
#%%

#%%
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
pulsars = pulsars[pulsars['suggested_pipeline'] == 'NB'] # select only those with narrow band search
print(len(pulsars))
# Removing frequency that are close to each over because they give similar results
test, pulsars_ind = np.unique(np.round(pulsars['F_GW'].to_numpy(), -1), return_index=True)
freq_GW = (pulsars['F_GW'].iloc[pulsars_ind]).to_numpy()
h_UL = (pulsars['upper_limits'].iloc[pulsars_ind]).to_numpy()
print(len(freq_GW))
#%%

#%%
# Grid of values of M, a and t
#MList = np.geomspace(Mmin, Mmax, 108) 
#aList = np.linspace(0, 1, 103)
MList = np.geomspace(Mmin, Mmax, 205) #113) 
aList = np.linspace(0, 0.99, 198) #105)

# Parameters used for Galactic BH distribution
RMW = 20*kpc; tMW = 13.6*1E9*Year
d_crit = 5*kpc; r_d = 2.15*kpc; z_max = 75.*pc; rho_obs = 8.*kpc
tIn = 10.*1E9*Year
#%%

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 
#%%


#%%
hList = np.geomspace(9E-31, 5E-24, 70)
#dfdlogh_center_sphere = np.zeros((len(freq_GW), len(hList)))
#dfdlogh_sphere = np.zeros((len(freq_GW), len(hList)))
dfdlogh_center_disc = np.zeros((len(freq_GW), len(hList)))
dfdlogh_disc = np.zeros((len(freq_GW), len(hList)))
Nevent = np.zeros((len(freq_GW)))
Ntot = 3000
#%%

#%%
start_time = time.time()
for i_fGW in tqdm(range(4)):
    t_SR, t_GW, h_peak, fdot_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, d_crit)

    for i_h, hVal in enumerate(tqdm(hList)):  
        #dfdlogh_center_sphere[i_fGW, i_h] = get_dfdlogh_const_center_sphere(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, RMW, tIn)
        #dfdlogh_sphere[i_fGW, i_h] = get_dfdlogh_const_sphere(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, rho_obs, RMW, tIn)
        dfdlogh_disc[i_fGW, i_h] = get_dfdlogh_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, rho_obs, tIn)
        #dfdlogh_center_disc[i_fGW, i_h] = get_dfdlogh_center_thin_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, tIn)
        dfdlogh_center_disc[i_fGW, i_h] = get_dfdlogh_center_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, z_max, tIn)

#    selh = (hList > h_UL[i_fGW])
#    Nevent[i_fGW] = Ntot*np.trapz(dfdlogh[i_fGW][selh]/hList[selh], x=hList[selh])
print("--- %s seconds ---" % (time.time() - start_time))
#%%

#%%
for i_fGW in tqdm(range(4)):
    print('Tot integral = ', np.trapz(dfdlogh_center_disc[i_fGW]/hList, x=hList))
#%%


############## Plots ##############
#%%
import matplotlib.pyplot as plt
import seaborn as sns
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


######### OLD CODE: uniform spatial distribution #########

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

