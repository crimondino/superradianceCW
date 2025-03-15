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
#Mmin, Mmax = 5, 30
#Norm = 1/1.35*(Mmax**(1.35)-Mmin**(1.35))/( (Mmax*Mmin)**(1.35))
def dndM(m):
    # Black hole mass distribution from 2003.03359
    # return 1/Norm*1/m**(2.35)
    return 1/m**(2.35)

def EM_lum(eps, alpha, Mc, MBH):
    '''Total EM power emitted by the SR cloud'''
    return eps**2*(0.131*alpha-0.188*alpha**2)*Mc/(GN*MBH)

#%%
def get_hTilde_peak(bc, alpha_grid, MList, aList, rd):
    '''Function to compute the strain hTilde as a function of mass, spin at the peak of the SR cloud growth'''

    tSR = np.zeros((len(MList), len(aList)))
    tGW = np.zeros((len(MList), len(aList)))
    hTilde = np.zeros((len(MList), len(aList)))
    fGWdot = np.zeros((len(MList), len(aList)))
    Fpeak = np.zeros((len(MList), len(aList)))
    Mcpeak = np.zeros((len(MList), len(aList)))
    #PowerRatiopeak = np.zeros((len(MList), len(aList)))
    tEMtGWRatio = np.zeros((len(MList), len(aList)))
    
    for i_m, mbh in enumerate(MList):
        alpha = alpha_grid[i_m, 0]

        for i_a, abh0 in enumerate(aList):
            try:
                wf = bc.make_waveform(mbh, abh0, alpha, units="physical+alpha")
                if wf.azimuthal_num()==1:
                    tSR[i_m, i_a] = wf.cloud_growth_time()*Second
                    tGW[i_m, i_a] = wf.gw_time()*Second
                    hTilde[i_m, i_a] = wf.strain_char(0., dObs=(rd/Mpc) )
                    fGWdot[i_m, i_a] = wf.freqdot_gw(0) #*(Hz/Second)
                    Fpeak[i_m, i_a] = (0.13*alpha-0.19*alpha**2)*wf.mass_cloud(0)/(GN*mbh)/(4*np.pi*rd**2)/(erg/Second/CentiMeter**2)
                    Mcpeak[i_m, i_a] = wf.mass_cloud(0)
                    tEMtGWRatio[i_m, i_a] = wf.power_gw(0)*Watt/EM_lum(1, alpha, wf.mass_cloud(0), mbh)
                else:
                    tSR[i_m, i_a] = np.nan
                    tGW[i_m, i_a] = np.nan
                    hTilde[i_m, i_a] = np.nan
                    fGWdot[i_m, i_a] = np.nan
                    Fpeak[i_m, i_a] = np.nan
                    Mcpeak[i_m, i_a] = np.nan
                    tEMtGWRatio[i_m, i_a] =  np.nan
            except ValueError:
                #print("ValueError", mbh, abh0, alpha)
                tSR[i_m, i_a] = np.nan
                tGW[i_m, i_a] = np.nan
                hTilde[i_m, i_a] = np.nan
                fGWdot[i_m, i_a] = np.nan
                Fpeak[i_m, i_a] = np.nan
                Mcpeak[i_m, i_a] = np.nan
                tEMtGWRatio[i_m, i_a] =   np.nan
                pass 
            
    return tSR, tGW, hTilde, fGWdot, Fpeak, Mcpeak, tEMtGWRatio


#%%
def get_d_minusplus(tSR, tGW, hTilde, hVal, rd, tInEn):

    dUL = hTilde/hVal*rd

    deltaT = tInEn - tSR + tGW
    xx = np.sqrt( deltaT**2. - 4.*dUL*tGW )

    dminus = 1/2.*( deltaT - xx )
    dplus = 1/2.*( deltaT + xx )
    
    return dminus, dplus


#%%
def get_d_limits(tSR, dminus, dplus, hTilde, hVal, rd, tIn):

    dUL = hTilde/hVal*rd    
    dmax_temp = np.array([dplus, dUL, tIn-tSR]).T  
    return dminus, np.min(dmax_temp, axis=2).T


#%%
def get_fdot_tobs(fdot0, tobs_over_tGW):
    return fdot0/(1. + tobs_over_tGW)**2

def get_fdot_tobs_exact(fdot0, tauRatio, tobs_over_tGW):
    return fdot0*np.exp(tobs_over_tGW/tauRatio)/(tauRatio - (1.+tauRatio)*np.exp(tobs_over_tGW/tauRatio) )**2

def get_Mcfrac_tobs_Ratio(tauRatio, tobs_over_tGW):
    # using the mask to avoid overflow error
    mask = (tobs_over_tGW/tauRatio < 500.)
    res = np.full_like(tobs_over_tGW, np.nan, dtype=np.float64)
    res[mask] = (np.exp(tobs_over_tGW[mask]/tauRatio)*(1. + tauRatio) - tauRatio )/(1. + tobs_over_tGW[mask])
    return res

#%%
def get_F_tobs(Fpeak, tobs_over_tGW):
    return Fpeak/(1. + tobs_over_tGW)

#%%
def Efield_fn(epsilon, alpha, mu, Nocc):
    '''SR cloud electric field'''
    return 1./np.sqrt(np.pi)*epsilon*alpha**(3/2)*mu**2*np.sqrt(Nocc)

def Gamma_pair_fn(epsilon, mu, alpha, Nocc):
    '''Pair production rate from photon-assisted Schwinger'''
    E_field = Efield_fn(epsilon, alpha, mu, Nocc)
    exp_factor = 4*MElectron**6*mu**2/((ElectronCharge*E_field)**4)
    
    return AlphaEM/(2*np.pi)*ElectronCharge*E_field/MElectron*np.exp(-exp_factor)


########## FUNCTIONS FOR BHs IN THE DISC (THIN DISC APPROX) ##########

#%%
def get_vol_int_disc(rd, robs, dmin, dmax, hTilde, hVal, fdot0, Fpeak, tauRatio_eps, n_phi = 101, n_rho = 99):

    int_over_vol = np.zeros( hTilde.shape )
    phi_list = np.linspace(0, 2*np.pi, n_phi)
    robs_tilde = robs/rd
    robs_tilde_sq = robs_tilde**2

    rho_ll = 0.1 * (dmin / rd - robs_tilde)
    rho_ll[dmin / rd <= robs_tilde] = 1E-4
    rho_ul = 10.*(dmax/rd + robs_tilde)

    cos_phi_grid = np.cos(phi_list)[:, np.newaxis]

    non_nan_indices = np.where(~np.isnan(hTilde))
    locations = list(zip(non_nan_indices[0], non_nan_indices[1]))

    for i_pair in locations:
        rho_list = np.concatenate( [ [0], np.geomspace( rho_ll[i_pair], rho_ul[i_pair], n_rho) ])
        rho_grid = rho_list[np.newaxis, :]

        dist_grid = np.sqrt( (rho_grid)**2 + robs_tilde_sq - 2*robs_tilde*rho_grid*cos_phi_grid )
        dist_grid[dist_grid==0] = np.nan

        tobs_over_tGW = (hTilde[i_pair]/hVal / dist_grid - 1.)
        # Upper bound on SR spin-up rate fdot
        Hfdot = np.heaviside(1./get_fdot_tobs(fdot0[i_pair], tobs_over_tGW)-1., 1.)  
        #Hfdot = np.heaviside(1./get_fdot_tobs_exact(fdot0[i_pair], tauRatio_eps[i_pair], tobs_over_tGW)-1., 1.)  # using the correct time evolution
        # Lower bound on the SR radio flux luminosity
        Hflux = np.heaviside(get_F_tobs(Fpeak[i_pair], tobs_over_tGW)/(dist_grid**2)-1., 1.) 
        # Upper bound on the EM power emitted such that Mc(t) power law time evolution is correct within 10%
        Hpower = np.heaviside(0.1 - np.abs(get_Mcfrac_tobs_Ratio(tauRatio_eps[i_pair], tobs_over_tGW) - 1.), 1.)            

        integrand = Hfdot*Hflux*Hpower*rho_grid*np.exp(-rho_grid)/dist_grid
        integrand[dist_grid < (dmin[i_pair]/rd)] = 0.
        integrand[dist_grid > (dmax[i_pair]/rd)] = 0.
        integrand[np.isnan(integrand)] = 0.

        int_over_rho = np.trapz(integrand, rho_list, axis=1)
        int_over_vol[i_pair] = np.trapz(int_over_rho, phi_list, axis=0)

    return int_over_vol/(2.*np.pi)


def get_Hplasma(mu, eps, Mpeak, alpha_grid):
    ### Compute the minimum value of epsilon allowed when the cloud reaches saturation
    Nocc = Mpeak*MSolar/mu
    gamma_pair_ratio = Gamma_pair_fn(eps, mu, alpha_grid, Nocc)/(alpha_grid*mu)
    return np.heaviside(gamma_pair_ratio - 1., 1.)  


#%%
def get_dfdlogh_disc(eps, tauRatio, tSR, tGW, hTilde, MList, aList, hVal, fdot0, Fpeak, rd, robs, tIn, xdisc): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    dminus, dplus = get_d_minusplus(tSR, tGW, hTilde, hVal, rd, tIn)
    dmin, dmax = get_d_limits(tSR, dminus, dplus, hTilde, hVal, rd, tIn) #get_d_limits(tSR, tGW, hTilde, hVal, rd, tIn)
    tauRatio_eps = tauRatio/eps**2.
    int_over_vol = get_vol_int_disc(rd, robs, dmin, dmax, hTilde, hVal, fdot0, Fpeak, tauRatio_eps)

    fMList = dndM(MList)
    integrand = (int_over_vol) * hTilde/hVal * tGW/tIn #* Hplasma
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)

    return xdisc*np.trapz(fMList*intOvera, x=MList)



########## FUNCTIONS FOR BHs IN THE BULGE ##########

#%%
#def get_vol_int_bulge(Rb, robs, dminusEn, dplusEn, dminusIn, dplusIn, dmin, dmax, tSR, tEn, hTilde, hVal, fdot0, fdotMax, n_phi = 101, n_theta = 105, n_r = 98):
def get_vol_int_bulge(tSR, tGW, hTilde, hVal, fdot0, fdotMax, dcrit, Rb, robs, tIn, tEn, n_phi = 101, n_theta = 105, n_r = 98):

    dminusEn, dplusEn = get_d_minusplus(tSR, tGW, hTilde, hVal, dcrit, tEn)
    dminusIn, dplusIn = get_d_minusplus(tSR, tGW, hTilde, hVal, dcrit, tIn)
    dmin, dmax = get_d_limits(tSR, dminusIn, dplusIn, hTilde, hVal, dcrit, tIn)

    int_over_vol = np.zeros( dmin.shape )
    r_list = np.geomspace(1E-5, 1E3, n_r)
    phi_list = np.linspace(0, 2*np.pi, n_phi)
    theta_list = np.linspace(0, np.pi, n_theta)
    robs_tilde = robs/Rb
    robs_tilde_sq = robs_tilde**2

    r_grid = r_list[np.newaxis, np.newaxis, :]
    r_grid_sq = r_grid**2
    sin_theta_grid = np.sin(theta_list)[:, np.newaxis, np.newaxis]
    cos_phi_grid = np.cos(phi_list)[np.newaxis, :, np.newaxis]

    #r_list = np.geomspace(1E-5, 1, n_r)
    #r_grid, phi_grid, theta_grid = np.meshgrid(phi_list, theta_list, r_list)

    for i_m in range(dmin.shape[0]):
        for i_s in range(dmin.shape[1]):

            dist_grid = np.sqrt( r_grid_sq + robs_tilde_sq 
                                - 2*robs_tilde*r_grid*cos_phi_grid*sin_theta_grid )

            # require fdot to be small enough
            tobs_over_tGW = (hTilde[i_m, i_s]/hVal / dist_grid - 1.)
            H = np.heaviside(fdotMax/get_fdot_tobs(fdot0[i_m, i_s], tobs_over_tGW)-1., 1.) 

            integrand = H*r_grid_sq*np.exp(-r_grid)/dist_grid

            cond1 = ( (dist_grid < (tEn - tSR[i_m, i_s])/Rb ) &
                     ( (dist_grid < dminusIn[i_m, i_s]/Rb) | 
                       ( (dminusEn[i_m, i_s]/Rb < dist_grid) & (dist_grid < dplusEn[i_m, i_s]/Rb)) | 
                       (dist_grid > dplusIn[i_m, i_s]/Rb) ) )
            cond2 = ( (dist_grid > (tEn - tSR[i_m, i_s])/Rb ) & 
                     ( (dist_grid < dmin[i_m, i_s]/Rb) | (dist_grid > dmax[i_m, i_s]/Rb) ) )
            
            integrand[cond1] = 0.
            integrand[cond2] = 0.

            int_over_r = np.trapz(integrand, r_list, axis=2)
            int_over_phi = np.trapz(int_over_r, phi_list, axis=1)
            int_over_vol[i_m, i_s] = np.trapz(int_over_phi, theta_list, axis=0)

    return int_over_vol/(8.*np.pi)
#%%


#%%
def get_dfdlogh_bulge(tSR, tGW, hTilde, MList, aList, hVal, fdot0, fdotMax, dcrit, Rb, robs, tIn, tEn, xbulge): 
    '''Returns the differential pdf of h as a function of log10(h)'''
   
    int_over_vol = get_vol_int_bulge(tSR, tGW, hTilde, hVal, fdot0, fdotMax, dcrit, Rb, robs, tIn, tEn)

    fMList = dndM(MList)
    integrand = dcrit/Rb * int_over_vol * hTilde/hVal * tGW/tIn
    integrand[np.isnan(integrand)] = 0

    intOvera = np.trapz(integrand, x=aList, axis=1)

    return xbulge*np.trapz(fMList*intOvera, x=MList)
#%%