#%%
import numpy as np
import sys

from utils.my_units import *

#import importlib
#importlib.reload(sys.modules['utils.analytic_estimate_pulsars'])
from utils.analytic_estimate_pulsars import *
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
print('Total number of NB pulsars with different frequency =',len(freq_GW))
#%%

#%%
i_fGW = int(sys.argv[1])
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
r_d = 2.15*kpc; z_max = 75.*pc; rho_obs = 8.*kpc
tIn = 10.*1E9*Year
x_disc, x_bulge = 0.85, 0.15

norm_mass= 1.35*(Mmax*Mmin)**(1.35)/(Mmax**(1.35)-Mmin**(1.35))
norm_disc = 1/(aMax-aMin)*norm_mass
#%%

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 
#%%

#%%
t_SR, t_GW, F_peak, fGW_dot = get_F_peak(bc, freq_GW[i_fGW], MList, aList, r_d)
#%%

#%%
epsSq_fr = (1.E-8)**2 * 1.E-5
BW = 1.4*GHz
FTh = 1.5*(1E-3)*Jy*BW/(erg/Second/CentiMeter**2)

FList = np.geomspace(1.E-2, 1.E3, 121)*FTh
dfdlogF_disc = np.zeros((len(FList)+1, 2))

dfdlogF_disc[0, :] = [epsSq_fr, freq_GW[i_fGW]]

for i_F, FVal in enumerate(FList):
    if (i_F%20 == 0):
        print('Computing F value number =',i_F)
    dfdlogF_disc[i_F+1, 0] = FVal 
    dfdlogF_disc[i_F+1, 1] = norm_disc*get_dfdlogF_disc(t_SR, t_GW, epsSq_fr*F_peak, MList, aList, FVal, fGW_dot, fdot_range[i_fGW],
                                                        r_d, rho_obs, tIn, x_disc)

# Save results
np.save('data/pulsars_events/dfdlogF_disc_NB_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[4])+'_'+str(sys.argv[5])+'_'+str(i_fGW)+'.npy', dfdlogF_disc)
#%%