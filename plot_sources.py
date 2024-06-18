#%%
import numpy as np
import importlib
import sys
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import seaborn as sns

from utils.my_units import *

importlib.reload(sys.modules['utils.load_pulsars'])
from utils.load_pulsars import load_pulsars_fnc
#%%

#%%
### Loading csv file with pulsars data as a panda dataframe
pulsars = load_pulsars_fnc()
#pulsars = pulsars[~pulsars['DECJ'].isna()]
#pulsars = pulsars[~pulsars['DECJ'].isin(['#ERROR!', '#REF!', '#VALUE!'])]
pulsars = pulsars[~pulsars['DECJ'].isin(['#ERROR!'])] # I couldn't find 'J1838-0022g' on the ATNF catalog. It is the only one with DECJ missing
pulsars.replace('NB', 'Narrow Band', inplace=True) 
print(len(pulsars))
#%%


#%%
# Example RA and Dec in degrees
ra_list = pulsars['RAJ'].to_numpy()
dec_list = pulsars['DECJ'].to_numpy()
coord_list = [ra_list[i]+' '+dec_list[i] for i in range(len(ra_list))]
coord_list
#%%

#%%
pipelines = np.unique(pulsars['suggested_pipeline'].to_numpy())
#%%

#%%
sky_coord = SkyCoord(coord_list, unit=(u.hourangle, u.deg), frame='icrs', obstime='J2000')
galactic_coords = sky_coord.galactic
l = galactic_coords.l.deg
b = galactic_coords.b.deg
#%%

# Plot using Healpy
markers = ['D', 'o', '^', 's', 'd', 'p']
marker_size = 40
colors = sns.color_palette("bright", len(pipelines)) 

hp.mollview(title="Sky Map of selected sources", coord='G', flip='geo')
hp.projscatter(270, 0, lonlat=True, s=marker_size, color='k', marker='x', label='Earth')
for i_p, p_name in enumerate(pipelines): 
    l_temp = l[pulsars['suggested_pipeline'] == p_name]
    b_temp = b[pulsars['suggested_pipeline'] == p_name]
    hp.projscatter(l_temp, b_temp, lonlat=True, s=marker_size, color=colors[i_p], marker=markers[i_p], label=p_name)
hp.graticule()
plt.legend(loc='lower left')
plt.savefig('figs/sky_map_galactic.pdf', format='pdf')
#plt.show()
#%%

#%%
NSIDE = 2**5
print(
    "Approximate resolution at NSIDE {} is {:.2} deg".format(
        NSIDE, hp.nside2resol(NSIDE, arcmin=True) / 60
    )
)

NPIX = hp.nside2npix(NSIDE)
print(NPIX)
sources_p = hp.ang2pix(NSIDE, [np.pi/2, np.pi], [0, np.pi/4])
#%%

#%%
m = np.zeros(NPIX) 
for i in range(len(sources_p)):
    m[sources_p[i]] = 10
m[m==0] = np.nan
#m = np.arange(NPIX)
hp.mollview(m, title="Mollview image RING")
hp.projscatter(l, b, s=20, lonlat=True, color='red', label='Sources')
hp.graticule()
#%%