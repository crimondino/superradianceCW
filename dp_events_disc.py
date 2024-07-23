#%%
import numpy as np
import sys

from utils.my_units import *

from utils.analytic_estimate_events import *
from utils.load_pulsars import load_pulsars_fnc
from superrad import ultralight_boson as ub
#%%

#%%
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
pulsars = pulsars[pulsars['suggested_pipeline'] == 'NB'] # select only those with narrow band search
print('Total number of NB pulsars =',len(pulsars))
# Removing frequency that are close to each over because they give similar results
test, pulsars_ind = np.unique(np.round(pulsars['F_GW'].to_numpy(), -1), return_index=True)
freq_GW = (pulsars['F_GW'].iloc[pulsars_ind]).to_numpy()
fdot_range = (pulsars['fdot range or resolution [Hz/s]'].iloc[pulsars_ind]).to_numpy()
h_UL = (pulsars['upper_limits'].iloc[pulsars_ind]).to_numpy()
print('Total number of NB pulsars with different frequency =',len(freq_GW))
#%%

#%%
i_fGW = int(sys.argv[1]) #1
print('\nComputing pulsar number =', i_fGW)
Mmin, Mmax = float(sys.argv[2]), float(sys.argv[3]) #5, 30
aMin, aMax = float(sys.argv[4]), float(sys.argv[5]) #0, 1
print('\nMmin, Mmax =', Mmin, ' ', Mmax, 'aMin, aMax =', aMin, ' ', aMax)
#%%

#%%
# Grid of values of M, a and t
MList = np.geomspace(Mmin, Mmax, 205) #113) 
aList = np.linspace(aMin, aMax, 198) #105)

# Parameters used for Galactic BH distribution
d_crit = 5*kpc; r_d = 2.15*kpc; z_max = 75.*pc; rho_obs = 8.*kpc
tIn = 10.*1E9*Year
x_disc, x_bulge = 0.85, 0.15

norm_mass= 1.35*(Mmax*Mmin)**(1.35)/(Mmax**(1.35)-Mmin**(1.35))
norm_disc = 1/(aMax-aMin)*norm_mass
#%%dp

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 
#%%

#%%
t_SR, t_GW, h_Tilde, fGW_dot, F_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, d_crit)
#%%

#%%
hList = np.geomspace(5E-28, 1E-24, 121)
dfdlogh_disc = np.zeros((len(hList)+1, 2))

dfdlogh_disc[0, :] = [0, freq_GW[i_fGW]]

epsSq_fr = (3.E-9)**2 * 1.E-5
BW = 1.4*GHz
FTh = 1.5*(1E-3)*Jy*BW/(erg/Second/CentiMeter**2)

for i_h, hVal in enumerate(hList):
    if (i_h%20 == 0):
        print('Computing h value number =',i_h)
    dfdlogh_disc[i_h+1, 0] = hVal 
    dfdlogh_disc[i_h+1, 1]  = norm_disc*get_dfdlogh_disc(t_SR, t_GW, h_Tilde, MList, aList, hVal, fGW_dot, fdot_range[i_fGW], 
                                                         epsSq_fr*F_peak, FTh,
                                                         d_crit, r_d, rho_obs, tIn, x_disc)

# Save results
np.save('data/disc_events/dfdlogh_disc_NB_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[4])+'_'+str(sys.argv[5])+'_'+str(i_fGW)+'_9M23.npy', dfdlogh_disc)
#%%