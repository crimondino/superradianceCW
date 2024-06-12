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
#MList = np.geomspace(Mmin, Mmax, 108) 
#aList = np.linspace(0, 1, 103)
MList = np.geomspace(Mmin, Mmax, 47) 
aList = np.linspace(0.7, 1, 22)

# Parameters used for Galactic BH distribution
RMW = 20*kpc; tMW = 13.6*1E9*Year
d_crit = 5*kpc; r_d = 2.15*kpc; z_max = 75.*pc; rho_obs = 8.*kpc
tIn = 10.*1E9*Year
#%%

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 
#%%

#%%
#t_SR, t_GW, h_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, d_crit)
#dmin, dmax = get_d_limits(t_SR, t_GW, h_peak, hVal, d_crit, tIn)
#R_min, R_max = get_R_limits(t_SR, t_GW, h_peak, hVal, RMW)
#%%

#%%
#vol_int = get_int_over_vol(r_d, rho_obs, d_crit, z_max, dmin, dmax) 
#vol_int_sphere = 3./2.*(R_max**2 - R_min**2)/RMW**2
#%%

#%%
i_fGW = 2
hVal = 1E-26
z_max = 3000*pc; 

t_SR, t_GW, h_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, d_crit)
dmin, dmax = get_d_limits(t_SR, t_GW, h_peak, hVal, d_crit, tIn)
vol_comp = get_vol_comp(r_d, z_max, dmin, dmax)
vol_sphere = get_int_over_vol_sphere(RMW, rho_obs, dmin, dmax)
#%%

#%%
hList = np.geomspace(1E-27, 1E-24, 30)
#dfdlogh_const = np.zeros((len(freq_GW), len(hList)))
#dfdlogh_const_small = np.zeros((len(freq_GW), len(hList)))
#dfdlogh = np.zeros((len(freq_GW), len(hList)))
dfdlogh_disc_thin = np.zeros((len(freq_GW), len(hList)))
dfdlogh_disc = np.zeros((len(freq_GW), len(hList)))
Nevent = np.zeros((len(freq_GW)))
Ntot = 3000
#%%

#%%
for i_fGW in tqdm(range(4)):
    t_SR, t_GW, h_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, d_crit)

    for i_h, hVal in enumerate(hList):  
        #print('Computing number of events from thin disc ...')  
        dfdlogh_disc_thin[i_fGW, i_h] = get_dfdlogh_center_thin_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, tIn)
        #dfdlogh_const[i_fGW, i_h] = get_dfdlogh_const_sphere(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, RMW, tIn)
        #dfdlogh_const_small[i_fGW, i_h] = get_dfdlogh_const_sphere(t_SR, t_GW, h_peak, MList, aList, hVal, 500*kpc)
        #print('Computing number of events from disk ...')  
        dfdlogh_disc[i_fGW, i_h] = get_dfdlogh_center_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, z_max, tIn)
        #dfdlogh[i_fGW, i_h] = get_dfdlogh(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, tIn, r_d, rho_obs, z_max)

#    selh = (hList > h_UL[i_fGW])
#    Nevent[i_fGW] = Ntot*np.trapz(dfdlogh[i_fGW][selh]/hList[selh], x=hList[selh])
#%%

#%%
#dfdlogh_const_small = np.zeros((len(freq_GW), len(hList)))
#hList = np.geomspace(1E-27, 1E-24, 200)
dfdlogh_center_disc = np.zeros((len(freq_GW), len(hList)))

for i_fGW in tqdm(range(4)):
    t_SR, t_GW, h_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, d_crit)

    for i_h, hVal in enumerate(hList):  
        #dfdlogh_const_small[i_fGW, i_h] = get_dfdlogh_const_sphere(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, 5*kpc, tIn)
        dfdlogh_center_disc[i_fGW, i_h] = get_dfdlogh_center_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, tIn)

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
for i_fGW in range(4):
    #line,=ax.plot(hList, 100/0.85*Ntot*dfdlogh[i_fGW], label=str(round(freq_GW[i_fGW], 1))+' Hz')
    line,=ax.plot(hList, Ntot*dfdlogh_disc_thin[i_fGW], linestyle='dashed', label=str(round(freq_GW[i_fGW], 1))+' Hz')
    #line,=ax.plot(hList, Ntot*dfdlogh_const_small[i_fGW], label=str(round(freq_GW[i_fGW], 1))+' Hz')
    line,=ax.plot(hList, Ntot*dfdlogh_disc[i_fGW], label=str(round(freq_GW[i_fGW], 1))+' Hz')
    #ax.axvline(x=h_UL[i_fGW], linewidth=1, alpha=0.8, c=line.get_color(), linestyle='dashed')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'h', fontsize=font_s); ax.set_ylabel(r'$dN/d\log h$', fontsize=font_s); 
ax.set_title('Number of events per unit strain, $N_{\mathrm{tot}} =$ '+str(Ntot), fontsize=font_s);
ax.legend(title='$f_{GW}$')
ax.set_ylim(0.001,10000)
ax.grid()

fig.tight_layout()
#fig.savefig('figs/dndlogh_center_disc.pdf', bbox_inches="tight")
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


