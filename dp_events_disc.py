#%%
import numpy as np
import sys

import importlib                                                 # *** COMMENT OUT ***
import sys                                                       # *** COMMENT OUT ***
importlib.reload(sys.modules['utils.analytic_estimate_events'])  # *** COMMENT OUT ***
from utils.analytic_estimate_events import *
from utils.load_pulsars import load_pulsars_fnc
from superrad import ultralight_boson as ub
from utils.my_units import *

#%%
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
print('\n Total number of pulsars =',len(pulsars), flush=True)
# Removing frequency that are close to each other because they give similar results
#test, pulsars_ind = np.unique(np.round(pulsars['F_GW'].to_numpy(), -1), return_index=True)
freq_GW = pulsars['F_GW'].to_numpy()
fdot_range = pulsars['fdot range or resolution [Hz/s]'].to_numpy()
#print('Total number of NB pulsars with different frequency =',len(freq_GW), flush=True)

#%%
#i_fGW = int(sys.argv[1]) 
i_fGW = 0 # *** COMMENT OUT ***
print('\nComputing pulsar number =', i_fGW, ', GW freq. = ', freq_GW[i_fGW], ' Hz', flush=True)
#Mmin, Mmax = float(sys.argv[2]), float(sys.argv[3]) 
#aMin, aMax = float(sys.argv[4]), float(sys.argv[5])
Mmin, Mmax = 5, 30 # *** COMMENT OUT ***
aMin, aMax = 0, 1  # *** COMMENT OUT ***
print('\nMmin, Mmax =', Mmin, ' ', Mmax, 'aMin, aMax =', aMin, ' ', aMax, flush=True)

#%%
# Grid of values of M, a and t
#MList = np.geomspace(Mmin, Mmax, 205)
#aList = np.linspace(aMin, aMax, 198) 
MList = np.geomspace(Mmin, Mmax, 98) # *** COMMENT OUT ***
aList = np.linspace(aMin, aMax, 101)  # *** COMMENT OUT ***

mu = np.pi*freq_GW[i_fGW]*Hz
alpha_grid = (GN*mu*MList*MSolar)[:, None]

# Parameters used for Galactic BH distribution
d_crit = 5*kpc; r_d = 2.15*kpc; z_max = 75.*pc; rho_obs = 8.*kpc
tIn = 10.*1E9*Year
x_disc, x_bulge = 0.85, 0.15

norm_mass= 1.35*(Mmax*Mmin)**(1.35)/(Mmax**(1.35)-Mmin**(1.35))
norm_disc = 1/(aMax-aMin)*norm_mass

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 

#%%
#t_SR, t_GW, h_Tilde, fGW_dot, F_peak, M_peak, Power_ratio_peak = get_hTilde_peak(bc, alpha_grid, MList, aList, d_crit)
t_SR, t_GW, h_Tilde, fGW_dot, F_peak, M_peak, tEMtGW_Ratio = get_hTilde_peak(bc, alpha_grid, MList, aList, d_crit)

#%%
#hList = np.geomspace(5E-28, 1E-24, 121)
#hList = np.geomspace(5.E-27, 6.E-24, 121)
hList = np.geomspace(5.E-26, 1.E-23, 20) # *** COMMENT OUT ***
dfdlogh_disc = np.zeros((len(hList)+1, 2))
dfdlogh_disc[0, :] = [0, freq_GW[i_fGW]]
dfdlogh_disc[1:, 0] = hList

log10eps = 6.5
eps, fr = np.power(10., -log10eps), 1.E-5
BW = 1.4*GHz
FTh = 1.5*(1E-3)*Jy*BW/(erg/Second/CentiMeter**2)

F_peak_rescaled = (eps**2. * fr)*F_peak/FTh

if np.isnan(fdot_range[i_fGW]):
    fGW_dot_rescaled = fGW_dot/1.
else:
    fGW_dot_rescaled = fGW_dot/fdot_range[i_fGW]


for i_h, hVal in enumerate(hList):
    if (i_h%20 == 0):
        print('Computing h value number =',i_h, flush=True)
    dfdlogh_disc[i_h+1, 1] = norm_disc*get_dfdlogh_disc(mu, eps, M_peak, alpha_grid, tEMtGW_Ratio, t_SR, t_GW, h_Tilde, MList, aList, hVal, 
                                                        fGW_dot_rescaled, F_peak_rescaled, d_crit, r_d, rho_obs, tIn, x_disc)
    

#old_dfdlogh_disc = np.array([[0.00000000e+00, 1.13076026e+02], [5.00000000e-28, 1.70146187e-01], [3.34370152e-27, 1.07074650e-01], 
#                             [2.23606798e-26, 7.21744878e-03], [1.49534878e-25, 2.68635069e-04], [1.00000000e-24, 1.00005841e-05]]) # *** COMMENT OUT ***
#print(dfdlogh_disc, '\n')                                                                                                           # *** COMMENT OUT ***
#print(dfdlogh_disc[1:,:]/old_dfdlogh_disc[1:,:])                                                                                    # *** COMMENT OUT ***


#%%
# Save results
#np.save('data/disc_events/dndlogh_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[4])+'_'+str(sys.argv[5])+'_'+str(i_fGW)+'_eps'+str(round(10*log10eps))+'_testPEM.npy', dfdlogh_disc)


#%%
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
rcParams['mathtext.rm'] = 'Times New Roman' 
rcParams['text.usetex'] = True
rcParams['font.family'] = 'times' #'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':14})

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))

ax.plot(dfdlogh_disc[1:, 0], dfdlogh_disc[1:, 1])

ax.set_xscale('log'); ax.set_yscale('log')
