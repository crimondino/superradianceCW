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
Norm = 1/1.35*(Mmax**(1.35)-Mmin**(1.35))/( (Mmax*Mmin)**(1.35))
def dndM(m):
    # Black hole mass distribution from 2003.03359
    return 1/Norm*1/m**(2.35)
#%%

#%%
def get_hTilde_peak(bc, freq_GW, MList, aList, dcrit):
    '''Function to compute the strain hTilde as a function of mass, spin at the peak of the SR cloud growth'''

    tSR = np.zeros((len(MList), len(aList)))
    tGW = np.zeros((len(MList), len(aList)))
    hTilde = np.zeros((len(MList), len(aList)))
    fGWdot = np.zeros((len(MList), len(aList)))

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
                    hTilde[i_m, i_a] = wf.strain_char(0., dObs=(dcrit/Mpc) )
                    fGWdot[i_m, i_a] = wf.freqdot_gw(0)*(Hz/Second)
                else:
                    tSR[i_m, i_a] = np.nan
                    tGW[i_m, i_a] = np.nan
                    hTilde[i_m, i_a] = np.nan
                    fGWdot[i_m, i_a] = np.nan
            except ValueError:
                #print("ValueError", mbh, abh0, alpha)
                tSR[i_m, i_a] = np.nan
                tGW[i_m, i_a] = np.nan
                hTilde[i_m, i_a] = np.nan
                fGWdot[i_m, i_a] = np.nan
                pass 
            
    return tSR, tGW, hTilde, fGWdot
#%%

#%%
def get_d_limits(tSR, tGW, hTilde, hVal, dcrit, tIn = 10.*1E9*Year):

    dUL = hTilde/hVal*dcrit

    deltaT = tIn - tSR + tGW
    xx = np.sqrt( deltaT**2. - 4.*dUL*tGW )

    dminus = 1/2.*( deltaT - xx )
    dplus = 1/2.*( deltaT + xx )
    dmax_temp = np.array([dplus, dUL, tIn-tSR]).T  
    
    return dminus, np.min(dmax_temp, axis=2).T
#%%

#%%
def get_d_plusminus(tSR, tGW, hTilde, hVal, dcrit, tInEn):

    dUL = hTilde/hVal*dcrit

    deltaT = tInEn - tSR + tGW
    xx = np.sqrt( deltaT**2. - 4.*dUL*tGW )

    dminus = 1/2.*( deltaT - xx )
    dplus = 1/2.*( deltaT + xx )
    
    return dminus, dplus
#%%

#%%
def get_fdot_tobs(fdot0, tobs_over_tGW):
    return fdot0/(1. + tobs_over_tGW)**2
#%%

########## FUNCTIONS FOR BHs IN THE DISC (THIN DISC APPROX) ##########

#%%
def get_vol_int_disc(rd, robs, dmin, dmax, tGW, hTilde, hVal, fdot0, fdotMax, n_phi = 101, n_rho = 99):

    int_over_vol = np.zeros( dmin.shape )
    phi_list = np.linspace(0, 2*np.pi, n_phi)
    robs_tilde = robs/rd
    robs_tilde_sq = robs_tilde**2

    rho_ll = 0.1 * (dmin / rd - robs_tilde)
    rho_ll[dmin / rd <= robs_tilde] = 1E-4
    rho_ul = 10.*(dmax/rd + robs_tilde)

    #phi_grid = phi_list[:, np.newaxis]
    #cos_phi_grid = np.cos(phi_grid)
    cos_phi_grid = np.cos(phi_list)[:, np.newaxis]

    for i_m in range(dmin.shape[0]):
        for i_s in range(dmin.shape[1]):

            rho_list = np.concatenate( [ [0], np.geomspace( rho_ll[i_m, i_s], rho_ul[i_m, i_s], n_rho) ])
            rho_grid = rho_list[np.newaxis, :]

            dist_grid = np.sqrt( (rho_grid)**2 + robs_tilde_sq - 2*robs_tilde*rho_grid*cos_phi_grid )

            # require fdot to be small enough
            tobs_over_tGW = (hTilde[i_m, i_s]/hVal / dist_grid - 1.)
            H = np.heaviside(fdotMax/get_fdot_tobs(fdot0[i_m, i_s], tobs_over_tGW)-1., 1.) 

            integrand = H*rho_grid*np.exp(-rho_grid)/dist_grid
            integrand[dist_grid < (dmin[i_m, i_s]/rd)] = 0.
            integrand[dist_grid > (dmax[i_m, i_s]/rd)] = 0.

            int_over_rho = np.trapz(integrand, rho_list, axis=1)
            int_over_vol[i_m, i_s] = np.trapz(int_over_rho, phi_list, axis=0)

    return int_over_vol/(2*np.pi)
#%%

#%%
def get_dfdlogh_disc(tSR, tGW, hTilde, MList, aList, hVal, dcrit, rd, robs, tIn): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    dmin, dmax = get_d_limits(tSR, tGW, hTilde, hVal, dcrit, tIn)
    int_over_vol = get_vol_int_disc(rd, robs, dmin, dmax)

    fMList = dndM(MList)
    integrand = dcrit/rd * (int_over_vol) * hTilde/hVal * tGW/tIn
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)
    res = np.trapz(fMList*intOvera, x=MList)

    return res
#%%



########## FUNCTIONS FOR BHs IN THE BULGE ##########

#%%
def get_vol_int_sphere(RMW, robs, dmin, dmax):

    int_over_vol = np.zeros( dmin.shape )
    phi_list = np.linspace(0, 2*np.pi, 77)
    theta_list = np.linspace(0, np.pi, 68)
    robs_tilde = robs/RMW

    r_list = np.geomspace(1E-5, 1, 78)
    r_grid, phi_grid, theta_grid = np.meshgrid(phi_list, theta_list, r_list)

    for i_m in range(dmin.shape[0]):
        for i_s in range(dmin.shape[1]):

            dist_grid = np.sqrt( (r_grid)**2 + np.full(r_grid.shape, robs_tilde**2) 
                                - 2*robs_tilde*r_grid*np.cos(phi_grid)*np.sin(theta_grid) )

            integrand = r_grid**2/dist_grid
            integrand[dist_grid < (dmin[i_m, i_s]/RMW)] = 0.
            integrand[dist_grid > (dmax[i_m, i_s]/RMW)] = 0.

            int_over_r = np.trapz(integrand, r_list, axis=2)
            int_over_phi = np.trapz(int_over_r, phi_list, axis=1)
            int_over_vol[i_m, i_s] = np.trapz(int_over_phi, theta_list, axis=0)

    return int_over_vol/(2.*np.pi)
#%%

#%%
def get_dfdlogh_const_sphere(tSR, tGW, hTilde, MList, aList, hVal, dcrit, robs, RMW=20*kpc, tMW = 13.6*1E9*Year): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    dmin, dmax = get_d_limits(tSR, tGW, hTilde, hVal, dcrit, tMW)
    int_over_vol = get_vol_int_sphere(RMW, robs, dmin, dmax)

    fMList = dndM(MList)
    integrand = 3./2. * dcrit/RMW * int_over_vol * hTilde/hVal * tGW/tMW
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)
    res = np.trapz(fMList*intOvera, x=MList)

    return res
#%%


