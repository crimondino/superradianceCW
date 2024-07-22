#%%
import numpy as np
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
#Mmin, Mmax = 5, 30
#Norm = 1/1.35*(Mmax**(1.35)-Mmin**(1.35))/( (Mmax*Mmin)**(1.35))
def dndM(m):
    # Black hole mass distribution from 2003.03359
#    return 1/Norm*1/m**(2.35)
    return 1/m**(2.35)
#%%

#%%
def get_F_peak(bc, freq_GW, MList, aList, dcrit):
    '''Function to compute the radio flux density as a function of mass, spin at the peak of the SR cloud growth'''

    tSR = np.zeros((len(MList), len(aList)))
    tGW = np.zeros((len(MList), len(aList)))
    Fpeak = np.zeros((len(MList), len(aList)))
    fGWdot = np.zeros((len(MList), len(aList)))

    mu = np.pi*freq_GW*Hz
    
    for i_m, mbh in enumerate(MList):
        alpha = GN*mu*mbh*MSolar

        for i_a, abh0 in enumerate(aList):
            try:
                wf = bc.make_waveform(mbh, abh0, alpha, units="physical+alpha")
                if wf.azimuthal_num()==1:
                    tSR[i_m, i_a] = wf.cloud_growth_time()*Second
                    tGW[i_m, i_a] = wf.gw_time()*Second
                    Fpeak[i_m, i_a] = (0.13*alpha-0.19*alpha**2)*wf.mass_cloud(0)/(GN*mbh)/(4*np.pi*dcrit**2)/(erg/Second/CentiMeter**2)
                    fGWdot[i_m, i_a] = wf.freqdot_gw(0)*(Hz/Second)
                else:
                    tSR[i_m, i_a] = np.nan
                    tGW[i_m, i_a] = np.nan
                    Fpeak[i_m, i_a] = np.nan
                    fGWdot[i_m, i_a] = np.nan
            except ValueError:
                #print("ValueError", mbh, abh0, alpha)
                tSR[i_m, i_a] = np.nan
                tGW[i_m, i_a] = np.nan
                Fpeak[i_m, i_a] = np.nan
                fGWdot[i_m, i_a] = np.nan
                pass 
            
    return tSR, tGW, Fpeak, fGWdot
#%%

#%%
def get_fdot_tobs(fdot0, tobs_over_tGW):
    return fdot0/(1. + tobs_over_tGW)**2
#%%

########## FUNCTIONS FOR BHs IN THE DISC (THIN DISC APPROX) ##########

#%%
def get_vol_int_disc(rd, robs, tSR, tGW, Fpeak, FVal, fdot0, fdotMax, tIn, n_phi = 101, n_rho = 99):

    int_over_vol = np.zeros( Fpeak.shape )
    phi_list = np.linspace(0, 2*np.pi, n_phi)
    robs_tilde = robs/rd
    robs_tilde_sq = robs_tilde**2

    #rho_ll = 1#0.1 * (dmin / rd - robs_tilde)
    #rho_ll[dmin / rd <= robs_tilde] = 1E-4
    #rho_ul = 10.*(dmax/rd + robs_tilde)
    rho_list = np.concatenate( [ [0], np.geomspace( 1.E-3, 1.E3, n_rho) ])
    rho_grid = rho_list[np.newaxis, :]

    cos_phi_grid = np.cos(phi_list)[:, np.newaxis]

    for i_m in range(Fpeak.shape[0]):
        for i_s in range(Fpeak.shape[1]):

            dist_grid_sq = (rho_grid)**2 + robs_tilde_sq - 2*robs_tilde*rho_grid*cos_phi_grid

            # require fdot to be small enough
            #tobs_over_tGW = (Fpeak[i_m, i_s]/FVal / dist_grid_sq - 1.)
            H = 1. #np.heaviside(fdotMax/get_fdot_tobs(fdot0[i_m, i_s], tobs_over_tGW)-1., 1.) 

            integrand = H*rho_grid*np.exp(-rho_grid)/dist_grid_sq
            cond1 = ( dist_grid_sq > (Fpeak[i_m, i_s]/FVal) )
            cond2 = ( (dist_grid_sq**(3/2) + (tSR[i_m, i_s] - tGW[i_m, i_s] - tIn)/rd*dist_grid_sq + Fpeak[i_m, i_s]/FVal*tGW[i_m, i_s]/rd) > 0 )
            integrand[cond1] = 0.
            integrand[cond2] = 0.
            integrand[np.isnan(integrand)] = 0.

            int_over_rho = np.trapz(integrand, rho_list, axis=1)
            int_over_vol[i_m, i_s] = np.trapz(int_over_rho, phi_list, axis=0)

    return int_over_vol/(2.*np.pi)
#%%

#%%
def get_dfdlogF_disc(tSR, tGW, Fpeak, MList, aList, FVal, fdot0, fdotMax, rd, robs, tIn, xdisc): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    int_over_vol = get_vol_int_disc(rd, robs, tSR, tGW, Fpeak, FVal, fdot0, fdotMax, tIn)

    fMList = dndM(MList)
    integrand = (int_over_vol) * Fpeak/FVal * tGW/tIn
    integrand[np.isnan(integrand)] = 0.

    intOvera = np.trapz(integrand, x=aList, axis=1)

    return xdisc*np.trapz(fMList*intOvera, x=MList)
#%%