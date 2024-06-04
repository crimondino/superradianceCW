#%%
import numpy as np
#%%

#%%
from os.path import dirname, abspath, join
import sys

THIS_DIR = dirname(__file__)
sys.path.append(THIS_DIR)
MAIN_DIR = abspath(join(THIS_DIR, '..'))
sys.path.append(MAIN_DIR)

from my_units import *
#from superra#d import ultralight_boson as ub
#%%

#%%
Mmin, Mmax = 5, 30
Norm = 0.740741*(Mmax**(1.35)-Mmin**(1.35))/( (Mmax*Mmin)**(1.35))

def dndM(m):
    # Black hole mass distribution from 2003.03359
    return 1/Norm*1/m**(2.35)
#%%

#%%
def get_hTilde_peak(bc, freq_GW, MList, aList, RMW=20*kpc):
    '''Function to compute the strain hTilde as a function of mass, spin at the peak of the SR cloud growth'''

    tSR = np.zeros((len(MList), len(aList)))
    tGW = np.zeros((len(MList), len(aList)))
    hTilde = np.zeros((len(MList), len(aList)))

    mu = np.pi*freq_GW*Hz
    
#    for i_m, mbh in enumerate(tqdm(MList)):
    for i_m, mbh in enumerate(MList):
        alpha = GN*mu*mbh*MSolar

        for i_a, abh0 in enumerate(aList):
            try:
                wf = bc.make_waveform(mbh, abh0, alpha, units="physical+alpha")
                if wf.azimuthal_num()==1:
                    tSR[i_m, i_a] = wf.cloud_growth_time()*Second
                    tGW[i_m, i_a] = wf.gw_time()*Second
                    hTilde[i_m, i_a] = wf.strain_char(0., dObs=(RMW/Mpc) )
                else:
                    tSR[i_m, i_a] = np.nan
                    tGW[i_m, i_a] = np.nan
                    hTilde[i_m, i_a] = np.nan
            except ValueError:
                tSR[i_m, i_a] = np.nan
                tGW[i_m, i_a] = np.nan
                hTilde[i_m, i_a] = np.nan
                pass 
            
    return tSR, tGW, hTilde
#%%

#%%
def get_R_limits(tSR, tGW, hTilde, hVal, RMW=20*kpc, tMW = 13.6*1E9*Year):

    RUL = hTilde/hVal*RMW

    deltaT = tMW - tSR + tGW
    xx = np.sqrt( deltaT**2. - 4.*RUL*tGW )

    Rmin = 1/2.*( deltaT - xx )
    Rmax = 1/2.*( deltaT + xx )
    Rmax_temp = np.array([Rmax, RUL, np.full(Rmax.shape, RMW), tMW-tSR]).T  
    
    return Rmin, np.min(Rmax_temp, axis=2).T
#%%

#%%
#def get_dfdlogh(bc, f_GW, MList, aList, hVal, RMW=20*kpc, tMW = 13.6*1E9*Year): 
def get_dfdlogh(tSR, tGW, hTilde, MList, aList, hVal, RMW=20*kpc, tMW = 13.6*1E9*Year): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    R_min, R_max = get_R_limits(tSR, tGW, hTilde, hVal, RMW, tMW)

    fMList = dndM(MList)
    integrand = 3./2. * (R_max**2. - R_min**2.)/(RMW**2.) * hTilde/hVal * tGW/tMW
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)
    res = np.trapz(fMList*intOvera, x=MList)

    return res
#%%

