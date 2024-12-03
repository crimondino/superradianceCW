#%%
import numpy as np
import sys

#import importlib    # *** COMMENT OUT ***
#importlib.reload(sys.modules['utils.analytic_estimate_pulsars'])    # *** COMMENT OUT ***
from utils.analytic_estimate_pulsars import *
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

#%%
i_fGW = int(sys.argv[1]) 
#i_fGW = 35 # *** COMMENT OUT ***
print('\nComputing pulsar number =', i_fGW, ', GW freq. = ', freq_GW[i_fGW], ' Hz', flush=True)
Mmin, Mmax = float(sys.argv[2]), float(sys.argv[3]) 
aMin, aMax = float(sys.argv[4]), float(sys.argv[5])
#Mmin, Mmax = 5, 30 # *** COMMENT OUT ***
#aMin, aMax = 0, 1  # *** COMMENT OUT ***
print('\nMmin, Mmax =', Mmin, ' ', Mmax, 'aMin, aMax =', aMin, ' ', aMax, flush=True)


#%%
# Grid of values of M, a and t
MList = np.geomspace(Mmin, Mmax, 205)
aList = np.linspace(aMin, aMax, 198) 
#MList = np.geomspace(Mmin, Mmax, 98) # *** COMMENT OUT ***
#aList = np.linspace(aMin, aMax, 101)  # *** COMMENT OUT ***

mu = np.pi*freq_GW[i_fGW]*Hz
alpha_grid = (GN*mu*MList*MSolar)[:, None]

# Parameters used for Galactic BH distribution
r_d = 2.15*kpc; z_max = 75.*pc; rho_obs = 8.*kpc
tIn = 10.*1E9*Year
x_disc, x_bulge = 0.85, 0.15

norm_mass= 1.35*(Mmax*Mmin)**(1.35)/(Mmax**(1.35)-Mmin**(1.35))
norm_disc = 1/(aMax-aMin)*norm_mass

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 

#%%
t_SR, t_GW, F_peak, M_peak, tEMtGW_Ratio = get_F_peak(bc, alpha_grid, MList, aList, r_d)


#%%
BW = 1.4*GHz
FTh = 1.5*(1E-3)*Jy*BW/(erg/Second/CentiMeter**2)
FList = np.geomspace(1.E-1, 5.E6, 121)*FTh 
#FList = np.geomspace(1., 1.E4, 42)*FTh  # *** COMMENT OUT ***
dfdlogF_disc = np.zeros((len(FList)+1, 2))
dfdlogF_disc[0, :] = [0, freq_GW[i_fGW]]
dfdlogF_disc[1:, 0] = FList

log10eps = 8.0
eps, fr = np.power(10., -log10eps), 1.E-5

Fpeak_fr_eps = (eps**2. * fr)*F_peak
Fpeak_temp = (eps**2. * fr)*F_peak/FTh

for i_F, FVal in enumerate(FList):
    if (i_F%20 == 0):
        print('Computing F value number =',i_F, flush=True)
    Fpeak_fr_eps_FVal = Fpeak_fr_eps/FVal
    dfdlogF_disc[i_F+1, 1] = norm_disc*get_dfdlogF_disc(mu, eps, M_peak, alpha_grid, tEMtGW_Ratio, t_SR, t_GW, Fpeak_fr_eps_FVal, 
                                                        MList, aList, r_d, rho_obs, tIn, x_disc)
                                             

# Save results
np.save('data/pulsars_events/dndlogF_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[4])+'_'+str(sys.argv[5])+'_'+str(i_fGW)+'_eps'+str(round(10*log10eps))+'.npy', dfdlogF_disc)


#%%  # *** COMMENT OUT ***
#import matplotlib.pyplot as plt

#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
#conv_f = (erg/Second/CentiMeter**2)/(1.E-3*Jy*BW)
#NBH = 1.E8

#ax.plot(dfdlogF_disc[1:, 0]/FTh, NBH*dfdlogF_disc[1:, 1])
#ax.set_xscale('log'); ax.set_yscale('log')
#ax.set_xlim(1.5E-5,1.E3); ax.set_ylim(1E-2,1.E4)
#ax.grid()
#ax.set_xlabel(r'$F_r\ [{\rm mJy}]$', fontsize=12); ax.set_ylabel(r'$dn_{F_r}/d\log F_r$', fontsize=12); 
#ax.set_title('Radio flux distribution', fontsize=12);


# %%
#cum_dist = np.zeros( (len(dfdlogF_disc[1:, 0]-1), 2) )
#for i_h in range(len(dfdlogF_disc[1:, 0]-1)):
#    cum_dist[i_h, 0] = dfdlogF_disc[1+i_h, 0]
#    cum_dist[i_h, 1] = np.trapz(dfdlogF_disc[1+i_h:, 1]/dfdlogF_disc[1+i_h:, 0], x=dfdlogF_disc[1+i_h:, 0])
#print(cum_dist[:11, 1]/x_disc)

##fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
conv_f = (erg/Second/CentiMeter**2)/(1.E-3*Jy*BW)

#ax.plot(conv_f*cum_dist[:, 0], cum_dist[:, 1]/x_disc)
#ax.set_xscale('log'); ax.set_yscale('log')
#ax.set_xlim(1.5E-1,1.E6); ax.set_ylim(1E-2,1.E4)

# %%
