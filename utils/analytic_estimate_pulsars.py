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
def get_F_peak(bc, alpha_grid, MList, aList, rd):
    '''Function to compute the radio flux density as a function of mass, spin at the peak of the SR cloud growth'''

    tSR = np.zeros((len(MList), len(aList)))
    tGW = np.zeros((len(MList), len(aList)))
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
                    Fpeak[i_m, i_a] = (0.13*alpha-0.19*alpha**2)*wf.mass_cloud(0)/(GN*mbh)/(4*np.pi*rd**2)/(erg/Second/CentiMeter**2)
                    Mcpeak[i_m, i_a] = wf.mass_cloud(0)
                    #PowerRatiopeak[i_m, i_a] =  wf.power_gw(0)*Watt/EM_lum(1, alpha, wf.mass_cloud(0), mbh)
                    tEMtGWRatio[i_m, i_a] = wf.power_gw(0)*Watt/EM_lum(1, alpha, wf.mass_cloud(0), mbh)
                else:
                    tSR[i_m, i_a] = np.nan
                    tGW[i_m, i_a] = np.nan
                    Fpeak[i_m, i_a] = np.nan
                    Mcpeak[i_m, i_a] = np.nan
                    tEMtGWRatio[i_m, i_a] =  np.nan
            except ValueError:
                #print("ValueError", mbh, abh0, alpha)
                tSR[i_m, i_a] = np.nan
                tGW[i_m, i_a] = np.nan
                Fpeak[i_m, i_a] = np.nan
                Mcpeak[i_m, i_a] = np.nan
                tEMtGWRatio[i_m, i_a] =  np.nan
                pass 
            
    return tSR, tGW, Fpeak, Mcpeak, tEMtGWRatio

#%%
#def get_Mcfrac_tobs_Ratio(tauRatio, tobs_over_tGW):
#    return (np.exp(tobs_over_tGW/tauRatio)*(1. + tauRatio) - tauRatio )/(1. + tobs_over_tGW)
def get_Mcfrac_tobs_Ratio(tauRatio, tobs_over_tGW):
    # using the mask to avoid overflow error
    mask = (tobs_over_tGW/tauRatio < 500.)
    res = np.full_like(tobs_over_tGW, np.nan, dtype=np.float64)
    res[mask] = (np.exp(tobs_over_tGW[mask]/tauRatio)*(1. + tauRatio) - tauRatio )/(1. + tobs_over_tGW[mask])
    return res

#%%
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

def get_F_tobs(Fpeak, tobs_over_tGW):
    return Fpeak/(1. + tobs_over_tGW)

def get_vol_int_disc(rd, robs, tSR, tGW, Fpeak_fr_eps_FVal, tIn, tauRatio_eps, n_phi = 101, n_rho = 120):

    int_over_vol = np.zeros( Fpeak_fr_eps_FVal.shape )
    phi_list = np.linspace(0, 2*np.pi, n_phi)
    robs_tilde = robs/rd
    robs_tilde_sq = robs_tilde**2

    rho_list = np.concatenate( [ [0], np.geomspace( 1.E-5, 1.E3, n_rho) ]) # check that this range is enough
    rho_grid = rho_list[np.newaxis, :]

    cos_phi_grid = np.cos(phi_list)[:, np.newaxis]
    dist_grid_sq = (rho_grid)**2 + robs_tilde_sq - 2*robs_tilde*rho_grid*cos_phi_grid

    non_nan_indices = np.where(~np.isnan(tSR))
    locations = list(zip(non_nan_indices[0], non_nan_indices[1]))

    for i_pair in locations:
        tobs_over_tGW = (Fpeak_fr_eps_FVal[i_pair] / (dist_grid_sq) - 1.)

        Htobs1 = np.heaviside((tIn - tSR[i_pair] - np.sqrt(dist_grid_sq)*rd - tobs_over_tGW*tGW[i_pair]), 1.) 
        Htobs2 = np.heaviside(tobs_over_tGW, 1.) 
        Hpower = np.heaviside(0.1 - np.abs(get_Mcfrac_tobs_Ratio(tauRatio_eps[i_pair], tobs_over_tGW) - 1.), 1.)     

        integrand = Htobs1*Htobs2*Hpower*rho_grid*np.exp(-rho_grid)/dist_grid_sq 
        integrand[np.isnan(integrand)] = 0.

        int_over_rho = np.trapz(integrand, rho_list, axis=1)
        int_over_vol[i_pair] = np.trapz(int_over_rho, phi_list, axis=0)

    return int_over_vol/(2.*np.pi)


#%%
def get_dfdlogF_disc(mu, eps, Mpeak, alpha_grid, tauRatio, tSR, tGW, Fpeak_fr_eps_FVal, MList, aList, rd, robs, tIn, xdisc): 
    '''Returns the differential pdf of h as a function of log10(h)'''

    tauRatio_eps = tauRatio/eps**2.
    int_over_vol = get_vol_int_disc(rd, robs, tSR, tGW, Fpeak_fr_eps_FVal, tIn, tauRatio_eps)

    ### Compute the minimum value of epsilon allowed when the cloud reaches saturation
    Nocc = Mpeak*MSolar/mu
    gamma_pair_ratio = Gamma_pair_fn(eps, mu, alpha_grid, Nocc)/(alpha_grid*mu)
    Hplasma = np.heaviside(gamma_pair_ratio - 1., 1.)   

    fMList = dndM(MList)
    integrand = (int_over_vol) * Fpeak_fr_eps_FVal * tGW/tIn * Hplasma
    integrand[np.isnan(integrand)] = 0.

    intOvera = np.trapz(integrand, x=aList, axis=1)

    return xdisc*np.trapz(fMList*intOvera, x=MList)


#%% 
### Test to speed up the function get_vol_int_disc_old. This one is actually slower
def get_vol_int_disc_test(rd, robs, tSR, tGW, Fpeak_fr_eps_FVal, Fpeak_temp, tIn, tauRatio_eps, n_phi = 101, n_rho = 120):

    int_over_vol = np.zeros( Fpeak_fr_eps_FVal.shape )
    phi_list = np.linspace(0, 2*np.pi, n_phi)
    robs_tilde = robs/rd
    robs_tilde_sq = robs_tilde**2

    rho_list = np.concatenate( [ [0], np.geomspace( 1.E-5, 1.E3, n_rho) ]) # check that this range is enough
    rho_grid = rho_list[np.newaxis, :]

    cos_phi_grid = np.cos(phi_list)[:, np.newaxis]
    dist_grid_sq = (rho_grid)**2 + robs_tilde_sq - 2*robs_tilde*rho_grid*cos_phi_grid
    tobs_over_tGW = (Fpeak_fr_eps_FVal[:, :, np.newaxis, np.newaxis] / (dist_grid_sq[np.newaxis, np.newaxis, :, :]) - 1.)

    Htobs1 = np.heaviside((tIn - tSR[:, :, np.newaxis, np.newaxis] 
                           - np.sqrt(dist_grid_sq)[np.newaxis, np.newaxis, :, :]*rd - tobs_over_tGW*tGW[:, :, np.newaxis, np.newaxis]), 1.) 
    Htobs2 = np.heaviside(tobs_over_tGW, 1.) 

    Hpower = 1. #np.heaviside(0.1 - np.abs(get_Mcfrac_tobs_Ratio(tauRatio_eps[i_pair], tobs_over_tGW) - 1.), 1.)     # *** COMMENT OUT ***
    #Hflux = np.heaviside(get_F_tobs(Fpeak_temp[i_pair], tobs_over_tGW)/dist_grid_sq-1., 1.) # *** COMMENT OUT ***

    integrand = Htobs1*Htobs2*Hpower*rho_grid[np.newaxis, np.newaxis, :, :]*np.exp(-rho_grid[np.newaxis, np.newaxis, :, :])/dist_grid_sq[np.newaxis, np.newaxis, :, :] 
    integrand[np.isnan(integrand)] = 0.

    int_over_rho = np.trapz(integrand, rho_list, axis=3)
    int_over_vol = np.trapz(int_over_rho, phi_list, axis=2)/(2.*np.pi)        

    return int_over_vol

