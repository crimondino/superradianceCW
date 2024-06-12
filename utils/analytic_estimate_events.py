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
def get_dfdlogh(tSR, tGW, hTilde, MList, aList, hVal, d_crit, tIn, r_d, rho_obs, z_max): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    dmin, dmax = get_d_limits(tSR, tGW, hTilde, hVal, d_crit, tIn)
    int_over_vol = get_int_over_vol(r_d, rho_obs, d_crit, z_max, dmin, dmax)

    fMList = dndM(MList)
    integrand = int_over_vol * hTilde/hVal * tGW/tIn
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)
    res = np.trapz(fMList*intOvera, x=MList)

    return res
#%%

########## FUNCTION FOR CONSTANT DISTRIBUTION OVER A SPHERE ##########
#%%
def get_R_limits(tSR, tGW, hTilde, hVal, dcrit, RMW=20*kpc, tMW = 13.6*1E9*Year):

    RUL = hTilde/hVal*dcrit, 

    deltaT = tMW - tSR + tGW
    xx = np.sqrt( deltaT**2. - 4.*RUL*tGW )

    Rmin = 1/2.*( deltaT - xx )
    Rmax = 1/2.*( deltaT + xx )
    Rmax_temp = np.array([Rmax, RUL, np.full(Rmax.shape, RMW), tMW-tSR]).T  
    
    return Rmin, np.min(Rmax_temp, axis=2).T
#%%

#%%
def get_int_over_vol_sphere(RMW, robs, dmin, dmax):

    int_over_vol = np.zeros( dmin.shape )
    phi_list = np.linspace(0, 2*np.pi, 201)
    theta_list = np.linspace(0, np.pi, 204)
    robs_tilde = robs/RMW

    for i_m in range(dmin.shape[0]):
        for i_s in range(dmin.shape[1]):

#            rho_list = np.geomspace(0.1*np.sqrt( (dmin[i_m, i_s]/RMW) - robs_tilde**2 ), 1, 304)
            r_list = np.geomspace(1E-5, 1, 304)
            r_grid, phi_grid, theta_grid = np.meshgrid(phi_list, theta_list, r_list)

            dist_grid = np.sqrt( (r_grid)**2 + np.full(r_grid.shape, robs_tilde**2) 
                                - 2*robs_tilde*r_grid*np.cos(phi_grid)*np.sin(theta_grid) )

            integrand = r_grid**2/dist_grid
            integrand[dist_grid < (dmin[i_m, i_s]/RMW)] = 0.
            integrand[dist_grid > (dmax[i_m, i_s]/RMW)] = 0.

            int_over_r = np.trapz(integrand, r_grid, axis=2)
            #print(int_over_r.shape, integrand.shape)
            int_over_phi = np.trapz(int_over_r, phi_list, axis=1)
            int_over_vol[i_m, i_s] = np.trapz(int_over_phi, theta_list, axis=0)

    return int_over_vol
#%%

#%%
def get_dfdlogh_const_sphere(tSR, tGW, hTilde, MList, aList, hVal, dcrit, RMW=20*kpc, tMW = 13.6*1E9*Year): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    R_min, R_max = get_R_limits(tSR, tGW, hTilde, hVal, dcrit, RMW, tMW)

    fMList = dndM(MList)
    integrand = 3./2. * dcrit/RMW * (R_max**2. - R_min**2.)/(RMW**2.) * hTilde/hVal * tGW/tMW
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)
    res = np.trapz(fMList*intOvera, x=MList)

    return res
#%%

########## FUNCTION FOR CONSTANT DISTRIBUTION OVER A DISC, THIN DISC APPROX ##########

#%%
def get_dfdlogh_center_thin_disc(tSR, tGW, hTilde, MList, aList, hVal, dcrit, rd, tIn): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    dmin, dmax = get_d_limits(tSR, tGW, hTilde, hVal, dcrit, tIn)

    fMList = dndM(MList)
    integrand = dcrit/rd * (np.exp(-dmin/rd) - np.exp(-dmax/rd)) * hTilde/hVal * tGW/tIn
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)
    res = np.trapz(fMList*intOvera, x=MList)

    return res
#%%

########## FUNCTION FOR CONSTANT DISTRIBUTION OVER A DISC, WITHOUT THIN DISC APPROX ##########

#%%
def get_int_over_vol(rd, zmax, dmin, dmax):

    int_over_vol = np.zeros( dmin.shape )
    z_list = np.linspace(0, 1, 201)

    for i_m in range(dmin.shape[0]):
        for i_s in range(dmin.shape[1]):

            rho_list = np.geomspace(0.1*(dmin[i_m, i_s]/rd), 10.*(dmax[i_m, i_s]/rd), 304)
            rho_grid, z_grid = np.meshgrid(rho_list, z_list)

            dist_grid = np.sqrt( (rho_grid)**2 + zmax**2/rd**2 * (z_grid)**2 )

            integrand = rho_grid*np.exp(-rho_grid)/dist_grid
            integrand[dist_grid < (dmin[i_m, i_s]/rd)] = 0.
            integrand[dist_grid > (dmax[i_m, i_s]/rd)] = 0.

            int_over_rho = np.trapz(integrand, rho_list, axis=1)
            int_over_vol[i_m, i_s] = np.trapz(int_over_rho, z_list, axis=0)

    return int_over_vol
#%%

#%%
def get_vol_comp(rd, zmax, dmin, dmax):

    int_over_vol = np.zeros( dmin.shape )
    z_list = np.linspace(0, 1, 201)

    for i_m in range(dmin.shape[0]):
        for i_s in range(dmin.shape[1]):

            #dist_grid[dist_grid < (dmin[i_m, i_s]/rd)] = np.nan 
            #dist_grid[dist_grid > dmax[i_m, i_s]/rd] = np.nan
            #rho_grid[dist_grid < dmin[i_m, i_s]/rd] = 0.
            #rho_grid[dist_grid > dmax[i_m, i_s]/rd] = 0.
            #print('Full ', dist_grid.shape )
            #print('Below or above: ', dist_grid[np.isnan(dist_grid)].shape )

            rho_list = np.geomspace(0.1*(dmin[i_m, i_s]/rd), 10.*(dmax[i_m, i_s]/rd), 304)
            rho_grid, z_grid = np.meshgrid(rho_list, z_list)

            dist_grid = np.sqrt( (rho_grid)**2 + zmax**2/rd**2 * (z_grid)**2 )

            integrand = rho_grid*np.exp(-rho_grid)/dist_grid
            integrand[dist_grid < (dmin[i_m, i_s]/rd)] = 0.
            integrand[dist_grid > (dmax[i_m, i_s]/rd)] = 0.

            int_over_rho = np.trapz(integrand, rho_list, axis=1)
            int_over_vol[i_m, i_s] = np.trapz(int_over_rho, z_list, axis=0)

    return int_over_vol/(np.exp(-dmin/rd) - np.exp(-dmax/rd))
#%%

#%%
def get_dfdlogh_center_disc(tSR, tGW, hTilde, MList, aList, hVal, dcrit, rd, zmax, tIn): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    dmin, dmax = get_d_limits(tSR, tGW, hTilde, hVal, dcrit, tIn)
    int_over_vol = get_int_over_vol(rd, zmax, dmin, dmax)

    fMList = dndM(MList)
    integrand = dcrit/rd * (int_over_vol) * hTilde/hVal * tGW/tIn
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)
    res = np.trapz(fMList*intOvera, x=MList)

    return res
#%%

########## OLD ##########

#%%
def get_int_over_vol_OLD(r_d, rho_obs, d_crit, z_max, dmin, dmax):

    rho_list = np.geomspace(1E-3, 100, 104)
    phi_list = np.linspace(0, 2*np.pi, 99)
    rho_grid, phi_grid = np.meshgrid(rho_list, phi_list)

    rho_sq = (rho_grid)**2 + (rho_obs/r_d)**2 - 2*rho_grid*(rho_obs/r_d)*np.cos(phi_grid)

    dmin_geom = np.sqrt( rho_sq )
    dmax_geom = np.sqrt( rho_sq + (z_max/r_d)**2 )
    
    y = np.linspace(0, 1, 134)

    int_over_rho = np.zeros( dmin.shape )

    for i_m in range(dmin.shape[0]):
        for i_s in range(dmin.shape[1]):
            dmin_final = np.maximum( np.full( dmin_geom.shape, dmin[i_m, i_s]/r_d), dmin_geom)
            dmax_final = np.minimum( np.full( dmax_geom.shape, dmax[i_m, i_s]/r_d), dmax_geom)

            bad_pos = (dmin_final > dmax_final)
            dmin_final[bad_pos] = np.nan
            dmax_final[bad_pos] = np.nan

            ### Integrate over the distance 
            dist_list = (dmin_final + y[..., None, None]*(dmax_final - dmin_final))
            integrand = 1/np.sqrt( dist_list**2 - rho_sq[None, ...]/r_d**2 )
            integrand[np.isnan(integrand)] = 0; dist_list[np.isnan(dist_list)] = 0
            int_over_y = np.trapz(integrand, dist_list, axis=0)
            int_over_phi = np.trapz(int_over_y, phi_list, axis=0)
            int_over_rho[i_m, i_s] = np.trapz(int_over_phi*np.exp(-rho_list), rho_list, axis=0)

    return 0.85/(2*np.pi)*d_crit/z_max*int_over_rho
#%%


