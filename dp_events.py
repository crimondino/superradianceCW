#%%
import numpy as np
import importlib
import sys
from tqdm import tqdm
import time

from utils.my_units import *

#importlib.reload(sys.modules['utils.analytic_estimate_events'])
#importlib.reload(sys.modules['utils.load_pulsars'])
from utils.analytic_estimate_events import *
from utils.load_pulsars import load_pulsars_fnc
from superrad import ultralight_boson as ub
#%%

#%%
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
pulsars = pulsars[pulsars['suggested_pipeline'] == 'NB'] # select only those with narrow band search
print(len(pulsars))
# Removing frequency that are close to each over because they give similar results
test, pulsars_ind = np.unique(np.round(pulsars['F_GW'].to_numpy(), -1), return_index=True)
freq_GW = (pulsars['F_GW'].iloc[pulsars_ind]).to_numpy()
fdot_range = (pulsars['fdot range or resolution [Hz/s]'].iloc[pulsars_ind]).to_numpy()
h_UL = (pulsars['upper_limits'].iloc[pulsars_ind]).to_numpy()
print(len(freq_GW))
#%%

#%%
# Grid of values of M, a and t
#MList = np.geomspace(Mmin, Mmax, 108) 
#aList = np.linspace(0, 1, 103)
MList = np.geomspace(Mmin, Mmax, 205) #113) 
aMin, aMax = 0, 0.99
aList = np.linspace(aMin, aMax, 198) #105)

# Parameters used for Galactic BH distribution
d_crit = 5*kpc; r_d = 2.15*kpc; z_max = 75.*pc; R_b = 120*pc; rho_obs = 8.*kpc
tIn, tEndB, tEndD = 10.*1E9*Year, 8.*1E9*Year, 0.
x_disc, x_bulge = 0.85, 0.15
#%%

#%%
bc = ub.UltralightBoson(spin=1, model="relativistic") 
#%%

#%%
t_SR, t_GW, h_Tilde, fGW_dot = get_hTilde_peak(bc, freq_GW[0], MList, aList, d_crit)
#%%

#%%
hVal = 5.E-26
dminusEn, dplusEn = get_d_minusplus(t_SR, t_GW, h_Tilde, hVal, d_crit, tEndB)
dminusIn, dplusIn = get_d_minusplus(t_SR, t_GW, h_Tilde, hVal, d_crit, tIn)
#%%

#%%
norm_disc = 1/(aMax-aMin)
norm_bulge = 1/(aMax-aMin)/(1-tEndB/tIn)

hVal = 5.E-26
dfdlogh_disc = norm_disc*get_dfdlogh_disc(t_SR, t_GW, h_Tilde, MList, aList, hVal, fGW_dot, fdot_range[0], d_crit, r_d, rho_obs, tIn, x_disc)
#%%

#%%
dminusEn, dplusEn = get_d_minusplus(t_SR, t_GW, h_Tilde, hVal, d_crit, tEndB)
dminusIn, dplusIn = get_d_minusplus(t_SR, t_GW, h_Tilde, hVal, d_crit, tIn)
dmin, dmax = get_d_limits(t_SR, dminusIn, dplusIn, h_Tilde, hVal, d_crit, tIn)
#%%

#%%
n_phi = 101; n_theta = 105; n_r = 98

int_over_vol = np.zeros( dmin.shape )
r_list = np.geomspace(1E-5, 1E3, n_r)
phi_list = np.linspace(0, 2*np.pi, n_phi)
theta_list = np.linspace(0, np.pi, n_theta)
robs_tilde = rho_obs/R_b
robs_tilde_sq = robs_tilde**2

r_grid = r_list[np.newaxis, np.newaxis, :]
r_grid_sq = r_grid**2
sin_theta_grid = np.sin(theta_list)[:, np.newaxis, np.newaxis]
cos_phi_grid = np.cos(phi_list)[np.newaxis, :, np.newaxis]
dist_grid = np.sqrt( r_grid_sq + robs_tilde_sq - 2*robs_tilde*r_grid*cos_phi_grid*sin_theta_grid )
#%%

#%%
i_m, i_s = 134, 165

# require fdot to be small enough
tobs_over_tGW = (h_Tilde[i_m, i_s]/hVal / dist_grid - 1.)
H = np.heaviside(fdot_range[0]/get_fdot_tobs(fGW_dot[i_m, i_s], tobs_over_tGW)-1., 1.) 
integrand = H*r_grid_sq*np.exp(-r_grid)/dist_grid
#%%

#%%
cond1 = ( (dist_grid < (tEndB - t_SR[i_m, i_s])/R_b ) &
            ( (dist_grid < dminusIn[i_m, i_s]/R_b) | 
            ( (dminusEn[i_m, i_s]/R_b < dist_grid) & (dist_grid < dplusEn[i_m, i_s]/R_b)) | 
            (dist_grid > dplusIn[i_m, i_s]/R_b) ) )
cond2 = ( (dist_grid > (tEndB - t_SR[i_m, i_s])/R_b ) & 
            ( (dist_grid < dmin[i_m, i_s]/R_b) | (dist_grid > dmax[i_m, i_s]/R_b) ) )

integrand[cond1] = 0.
integrand[cond2] = 0.

int_over_r = np.trapz(integrand, r_list, axis=2)
int_over_phi = np.trapz(int_over_r, phi_list, axis=1)
int_over_vol[i_m, i_s] = np.trapz(int_over_phi, theta_list, axis=0)
#%%

#%%
dfdlogh_bulge = norm_bulge*get_dfdlogh_bulge(t_SR, t_GW, h_Tilde, MList, aList, hVal, fGW_dot, fdot_range[0], d_crit, R_b, rho_obs, tIn, tEndB, x_bulge)
#%%


#%%
hList = np.geomspace(9E-31, 5E-24, 70)
#dfdlogh_center_sphere = np.zeros((len(freq_GW), len(hList)))
#dfdlogh_sphere = np.zeros((len(freq_GW), len(hList)))
dfdlogh_center_disc = np.zeros((len(freq_GW), len(hList)))
dfdlogh_disc = np.zeros((len(freq_GW), len(hList)))
Nevent = np.zeros((len(freq_GW)))
Ntot = 3000
#%%

#%%
start_time = time.time()
for i_fGW in tqdm(range(4)):
    t_SR, t_GW, h_peak, fdot_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, d_crit)

    for i_h, hVal in enumerate(tqdm(hList)):  
        #dfdlogh_center_sphere[i_fGW, i_h] = get_dfdlogh_const_center_sphere(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, RMW, tIn)
        #dfdlogh_sphere[i_fGW, i_h] = get_dfdlogh_const_sphere(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, rho_obs, RMW, tIn)
        dfdlogh_disc[i_fGW, i_h] = get_dfdlogh_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, rho_obs, tIn)
        #dfdlogh_center_disc[i_fGW, i_h] = get_dfdlogh_center_thin_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, tIn)
        dfdlogh_center_disc[i_fGW, i_h] = get_dfdlogh_center_disc(t_SR, t_GW, h_peak, MList, aList, hVal, d_crit, r_d, z_max, tIn)

#    selh = (hList > h_UL[i_fGW])
#    Nevent[i_fGW] = Ntot*np.trapz(dfdlogh[i_fGW][selh]/hList[selh], x=hList[selh])
print("--- %s seconds ---" % (time.time() - start_time))
#%%

#%%
for i_fGW in tqdm(range(4)):
    print('Tot integral = ', np.trapz(dfdlogh_center_disc[i_fGW]/hList, x=hList))
#%%


############## Plots ##############
#%%
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib.lines as mlines
rcParams['mathtext.rm'] = 'Times New Roman' 
rcParams['text.usetex'] = True
rcParams['font.family'] = 'times' #'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':14})
#%%

#%%
### Plot signal strain distribution df/dlogh
ntot = 10
dfdlogh = []
dfdlogh_lowmass = []
freq_GW = np.zeros(ntot)

for i in range(ntot):
    list_temp = np.load('data/disc_events/dfdlogh_disc_NB_'+str(i)+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh.append(list_temp[1:])
    list_temp = np.load('data/disc_events/dfdlogh_disc_NB_5_20_0_1_'+str(i)+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh_lowmass.append(list_temp[1:])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("mako", ntot) 

for i_fGW in range(ntot):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(dfdlogh[i_fGW][:, 0], dfdlogh[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    line,=ax.plot(dfdlogh_lowmass[i_fGW][:, 0], dfdlogh_lowmass[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashed', linewidth=1)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\bar{h}$', fontsize=font_s); ax.set_ylabel(r'$df_{h}/d\log \bar{h}$', fontsize=font_s); 
ax.set_title('Signal strain distribution', fontsize=font_s);
legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc='lower left', fontsize=14)
ax.set_ylim(5E-8,0.4)
ax.grid()

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'$M_{\rm max} = 30\ M_{\odot}, \chi_{\rm max} = 1$')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'$M_{\rm max} = 20\ M_{\odot}, \chi_{\rm max} = 1$')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'$M_{\rm max} = 30\ M_{\odot}, \chi_{\rm max} = 0.5$')
# Add the second legend for the linestyles
legend2 = ax.legend(handles=[line_style1, line_style2, line_style3], loc='upper right', handletextpad=0.5, frameon=False, 
                   labelspacing=0.2, handlelength=1, fontsize=14)

# Add the first legend back to the plot
ax.add_artist(legend)

fig.tight_layout()
fig.savefig('figs/strain_dist_disc.pdf', bbox_inches="tight")
#%%

#%%
### Plot signal strain distribution df/dlogh
ntot = 10
cum_dist = []
cum_dist_lowmass = []
freq_GW = np.zeros(ntot)

for i in range(ntot):
    list_temp = np.load('data/disc_events/dfdlogh_disc_NB_'+str(i)+'.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist.append(cum_dist_temp)

    list_temp = np.load('data/disc_events/dfdlogh_disc_NB_5_20_0_1_'+str(i)+'.npy')
    freq_GW[i] = list_temp[0, 1]

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist_lowmass.append(cum_dist_temp)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("mako", ntot) 

for i_fGW in range(ntot):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    line,=ax.plot(cum_dist[i_fGW][:, 0], cum_dist[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, label=str(round(freq_GW[i_fGW], 1))+', '+str(round(mu_temp, 1)))
    line,=ax.plot(cum_dist_lowmass[i_fGW][:, 0], cum_dist_lowmass[i_fGW][:, 1], color = colors[i_fGW], linewidth=1, linestyle='dashed')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\bar{h}$', fontsize=font_s); ax.set_ylabel(r'$P(h>\bar{h})$', fontsize=font_s); 
ax.set_title('Cumulative signal strain distribution', fontsize=font_s);
ax.legend(title='$f_{\mathrm{GW}}\ [Hz]$', frameon=False, labelspacing=0.2, ncol=2, loc='lower left', fontsize=14)
ax.set_xlim(4E-28,5E-25) #ax.set_ylim(5E-8,0.4)
legend = ax.legend(title='$f_{\mathrm{GW}}\ [{\mathrm{Hz}}], m\ [10^{-13}\ {\mathrm{eV}}]$', handletextpad=0.5, frameon=False, 
                  labelspacing=0.2, ncol=2, columnspacing=1,handlelength=1, loc='lower left', fontsize=14)
ax.grid()

line_style1 = mlines.Line2D([], [], color='black', linestyle='solid', label=r'$M_{\rm max} = 30\ M_{\odot}, \chi_{\rm max} = 1$')
line_style2 = mlines.Line2D([], [], color='black', linestyle='dashed', label=r'$M_{\rm max} = 20\ M_{\odot}, \chi_{\rm max} = 1$')
line_style3 = mlines.Line2D([], [], color='black', linestyle='dotted', label=r'$M_{\rm max} = 30\ M_{\odot}, \chi_{\rm max} = 0.5$')
# Add the second legend for the linestyles
legend2 = ax.legend(handles=[line_style1, line_style2, line_style3], loc='upper right', handletextpad=0.5, frameon=False, 
                   labelspacing=0.2, handlelength=1, fontsize=14)

# Add the first legend back to the plot
ax.add_artist(legend)

fig.tight_layout()
#fig.savefig('figs/strain_cumulative_disc.pdf', bbox_inches="tight")
#%%


#%%
### Plot for the scalar field to compare with sims results
ntot = 6
dfdlogh = []
cum_dist = []
cum_dist_masha = []
masha_filenames = ['data/scalar/Fig15_'+str(i)+'_2003_03359.txt' for i in [2, 5, 8, 15, 20, 35]]
Ntot = 1.E8
freq_GW = np.zeros(ntot)

for i in range(ntot):
    list_temp = np.load('data/disc_events/dfdlogh_disc_scalar_'+str(i)+'.npy')
    freq_GW[i] = list_temp[0, 1]
    dfdlogh.append(list_temp[1:])

    cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )
    for i_h in range(len(list_temp[1:, 0]-1)):
        cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
        cum_dist_temp[i_h, 1] = Ntot*np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
    cum_dist.append(cum_dist_temp)

    cum_dist_masha.append(np.loadtxt(masha_filenames[i]))

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("deep", ntot) 

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(ntot):
    mu_temp = np.pi*freq_GW[i_fGW]*Hz/(1.E-13*eV)
    #line,=ax.plot(dfdlogh[i_fGW][:, 0], dfdlogh[i_fGW][:, 1], color = colors[i_fGW], label=str(int(mu_temp)))
    line,=ax.plot(cum_dist[i_fGW][:, 0], cum_dist[i_fGW][:, 1], color = colors[i_fGW], label=str(round(mu_temp, 1)))
    line,=ax.plot(cum_dist_masha[i_fGW][:, 0]*1.E-26, cum_dist_masha[i_fGW][:, 1], color = colors[i_fGW], linestyle='dashed', linewidth=2)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$h_0$', fontsize=font_s); ax.set_ylabel(r'Number of signals above $h_0$', fontsize=font_s); 
ax.set_title('Scalar field, comparison with Sylvia et al.', fontsize=font_s);
ax.legend(title='$\mu\ [10^{-13}\ {\mathrm eV}]$', frameon=False, labelspacing=0.2)
ax.set_xlim(2E-27,3E-24); ax.set_ylim(1, 5E6)
ax.grid()
ax.text(1.2E-25, 1.5E6, r'Sylvia (dashed)', fontsize=12)
ax.text(1.2E-25, 5E5, r'Mine (solid)', fontsize=12)
#ax.text(1E-30, 1200, r'$s_{\rm min} = 0,\ s_{\rm max} = 1$', fontsize=12)

fig.tight_layout()
fig.savefig('figs/scalar_comp.pdf', bbox_inches="tight")
#%%

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16
colors = sns.color_palette("deep", 4) 

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(4):
    #line,=ax.plot(hList, Ntot*dfdlogh_center_disc[i_fGW], linestyle='dashed', color = colors[i_fGW])
    line,=ax.plot(hList, Ntot*dfdlogh_disc[i_fGW], color = colors[i_fGW], label=str(round(freq_GW[i_fGW], 1))+' Hz')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'h', fontsize=font_s); ax.set_ylabel(r'$dN/d\log h$', fontsize=font_s); 
ax.set_title('Galactic disc, Earth observer, $N_{\mathrm{tot}} =$ '+str(Ntot), fontsize=font_s);
ax.legend(title='$f_{\rm GW}$')
ax.set_ylim(0.01,1E4)
ax.grid()
ax.text(1E-30, 3000, r'$M_{\rm min} = 5\ M_{\odot},\ M_{\rm max} = 30\ M_{\odot}$', fontsize=12)
ax.text(1E-30, 1200, r'$s_{\rm min} = 0,\ s_{\rm max} = 1$', fontsize=12)

fig.tight_layout()
fig.savefig('figs/dndlogh_disc.pdf', bbox_inches="tight")
#%%

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
font_s=16

#for i_fGW in range(len(freq_GW)):
for i_fGW in range(1):
    line,=ax.plot(hList, Ntot*dfdlogh_center_disc[i_fGW], linestyle='dashed', label='GC observer') #label=str(round(freq_GW[i_fGW], 1))+' Hz')
    line,=ax.plot(hList, Ntot*dfdlogh_disc[i_fGW], label='Earth observer')

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'h', fontsize=font_s); ax.set_ylabel(r'$dN/d\log h$', fontsize=font_s); 
#ax.set_title('Number of events per unit strain, $N_{\mathrm{tot}} =$ '+str(Ntot), fontsize=font_s);
ax.set_title('Galactic disc spatial distribution, $f_{\rm GW}$ = '+str(round(freq_GW[i_fGW], 1))+' Hz');
#ax.legend(title='$f_{\rm GW}$')
ax.legend()
ax.set_ylim(0.01,1E4)
ax.grid()
ax.text(2E-31, 3000, r'$M_{\rm min} = 5\ M_{\odot}, $')
ax.text(1E-29, 3000, r'$M_{\rm max} = 30\ M_{\odot}$')
ax.text(2E-31, 1000, r'$s_{\rm min} = 0, $')
ax.text(3E-30, 1000, r'$s_{\rm max} = 1$')

fig.tight_layout()
fig.savefig('figs/dndlogh_disc_comp.pdf', bbox_inches="tight")
#%%

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
colorlist =  ['darkslateblue', 'darkred', 'orangered', 'green'] #['teal', 'orange', 'indianred']
liestyle_list = ['solid', 'dashed', 'dashdot']
font_s=16

ax.plot(freq_GW, Nevent, 's', marker='o')

ax.set_title('N events above UL for different pulsars, $N_{\mathrm{tot}} =$ '+str(Ntot),fontsize=font_s);
ax.set_xlabel(r'$f_{\mathrm{GW}}$ [Hz]', fontsize=font_s); ax.set_ylabel(r'$N_{\mathrm{ev}}(h>h_{\mathrm{UL}})$', fontsize=font_s); 
ax.set_yscale('log')
#ax.grid()
#ax.set_xlim(100,300); #ax.set_ylim(0,10)

fig.tight_layout()
fig.savefig('figs/Nexpected_new.pdf', bbox_inches="tight")
#%%


######### OLD CODE: uniform spatial distribution #########

#%%
hList = np.geomspace(1E-27, 1E-24, 300)
dfdlogh = np.zeros((len(freq_GW), len(hList)))
Nevent = np.zeros((len(freq_GW)))
Ntot = 3000

for i_fGW in tqdm(range(len(freq_GW))):
    t_SR, t_GW, h_peak = get_hTilde_peak(bc, freq_GW[i_fGW], MList, aList, RMW)

    for i_h, hVal in enumerate(hList):    
        dfdlogh[i_fGW, i_h] = get_dfdlogh(t_SR, t_GW, h_peak, MList, aList, hVal)

    selh = (hList > h_UL[i_fGW])
    Nevent[i_fGW] = Ntot*np.trapz(dfdlogh[i_fGW][selh]/hList[selh], x=hList[selh])
#%%

for i_fGW in tqdm(range(len(freq_GW))):

    selh = (hList > h_UL[i_fGW])
    Nevent[i_fGW] = Ntot*np.trapz(dfdlogh[i_fGW][selh]/hList[selh], x=hList[selh])

