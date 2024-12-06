#%%
import numpy as np
import sys

#import importlib                                                 # *** COMMENT OUT ***
#importlib.reload(sys.modules['utils.analytic_estimate_events'])  # *** COMMENT OUT ***
#import time                                                      # *** COMMENT OUT ***
#import matplotlib.pyplot as plt                                  # *** COMMENT OUT ***
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

#%%
testing = False                                   # *** COMMENT OUT *** change to False
if testing:
    i_fGW = 0                   
    Mmin, Mmax = 5, 30           
    aMin, aMax = 0, 1            
    MValues, aValues, hValues = 98, 101, 20
    log10eps = 8.6
else:
    i_fGW = int(sys.argv[1]) 
    Mmin, Mmax = float(sys.argv[2]), float(sys.argv[3]) 
    aMin, aMax = float(sys.argv[4]), float(sys.argv[5])
    log10eps = float(sys.argv[6])
    MValues, aValues, hValues = 205, 198, 121                 
print('\nComputing pulsar number =', i_fGW, ', GW freq. = ', freq_GW[i_fGW], ' Hz', flush=True)
print('\nMmin, Mmax =', Mmin, ' ', Mmax, 'aMin, aMax =', aMin, ' ', aMax, flush=True)

#%%
# Grid of values of M, a and t
MList = np.geomspace(Mmin, Mmax, MValues)
aList = np.linspace(aMin, aMax, aValues) 

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
t_SR, t_GW, h_Tilde, fGW_dot, F_peak, M_peak, tEMtGW_Ratio = get_hTilde_peak(bc, alpha_grid, MList, aList, r_d)

#%%
BW = 1.4*GHz
FTh = 1.5*(1E-3)*Jy*BW/(erg/Second/CentiMeter**2)
fr = 1.E-5

hMin = 1.E-26  # large range for the plot of the distributions: 5E-28, 6.E-24

if log10eps <= 7.:
    if freq_GW[i_fGW] < 250:
        hMax = 5.E-19
    elif freq_GW[i_fGW] < 400:
        hMax = 1.E-20
    else:
        hMax = 3.E-21
elif log10eps <= 8.5:
    if freq_GW[i_fGW] < 250:
        hMax = 1.E-21
    elif freq_GW[i_fGW] < 400:
        hMax = 3.E-22
    else:
       hMax = 1.E-22
else:
    hMax = 1.E-22

print('\nlog10(eps) =', log10eps, ' hMin, hMax = ', hMin, hMax, flush=True)


#%%
eps = np.power(10., -log10eps)
F_peak_rescaled = (eps**2. * fr)*F_peak/FTh

hList = np.geomspace(hMin, hMax, hValues)
dfdlogh_disc = np.zeros((hValues+1, 2))
dfdlogh_disc[0, :] = [0, freq_GW[i_fGW]]
dfdlogh_disc[1:, 0] = hList

if np.isnan(fdot_range[i_fGW]):
    fGW_dot_rescaled = fGW_dot/1.
else:
    fGW_dot_rescaled = fGW_dot/fdot_range[i_fGW]

Hplasma = get_Hplasma(mu, eps, M_peak, alpha_grid)
h_Tilde = np.where(Hplasma == 0., np.nan, h_Tilde)

#start_time = time.time()                                     # *** COMMENT OUT ***
for i_h, hVal in enumerate(hList):
    if (i_h%20 == 0):
        print('Computing h value number =',i_h, flush=True)
    dfdlogh_disc[i_h+1, 1] = norm_disc*get_dfdlogh_disc(eps, tEMtGW_Ratio, t_SR, t_GW, h_Tilde, MList, aList, hVal, 
                                                        fGW_dot_rescaled, F_peak_rescaled, r_d, rho_obs, tIn, x_disc)
        
#end_time = time.time()                                       # *** COMMENT OUT ***
#print(f"Execution time: {end_time - start_time} seconds")    # *** COMMENT OUT ***

#%%
# Save results
if not testing:
    np.save('data/disc_events/dndlogh_'+str(sys.argv[2])+'_'+str(sys.argv[3])+'_'+str(sys.argv[4])+'_'+str(sys.argv[5])+'_'+str(i_fGW)+'_eps'+str(round(10*log10eps))+'.npy', dfdlogh_disc)


#%%
if testing:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
    ax.plot(dfdlogh_disc[1:, 0], dfdlogh_disc[1:, 1])

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(hMin,hMax); 
    #ax.set_ylim(1E-8,1.E-5)
    ax.grid()
    ax.set_xlabel(r'$h$', fontsize=12); ax.set_ylabel(r'$dn_{h}/d\log h$', fontsize=12); 

# %%
if testing:
    cum_dist = np.zeros( (len(dfdlogh_disc[1:, 0]-1), 2) )
    for i_h in range(len(dfdlogh_disc[1:, 0]-1)):
        cum_dist[i_h, 0] = dfdlogh_disc[1+i_h, 0]
        cum_dist[i_h, 1] = np.trapz(dfdlogh_disc[1+i_h:, 1]/dfdlogh_disc[1+i_h:, 0], x=dfdlogh_disc[1+i_h:, 0])
    print(cum_dist[:11, 1]/x_disc)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))

    ax.plot(cum_dist[:, 0], cum_dist[:, 1]/x_disc)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(hMin,hMax); #ax.set_ylim(1E-2,1.E4)
# %%
