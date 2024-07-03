#%%
import numpy as np
import importlib
import sys
import healpy as hp
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, Galactocentric
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
dist_list = [pulsars.iloc[i]['DIST']*u.kpc for i in range(len(ra_list))]
#coord_list
#%%

#%%
pipelines = pulsars['suggested_pipeline'].unique() #np.unique(pulsars['suggested_pipeline'].to_numpy())
#%%

#%%
sky_coord = SkyCoord(coord_list, unit=(u.hourangle, u.deg), distance=dist_list, frame='icrs', obstime='J2000')
galactic_coords = sky_coord.galactic
l = galactic_coords.l.deg
b = galactic_coords.b.deg
#%%

#%%
# Define the Galactocentric frame with the distance from the Sun to the Galactic Center
galactocentric_frame = Galactocentric(galcen_distance=8.122 * u.kpc)

# Convert to Galactocentric coordinates
galactocentric_coords = galactic_coords.transform_to(galactocentric_frame)

# Extract the 3D positions in Galactocentric coordinates
x = galactocentric_coords.cartesian.x.to(u.kpc).value
y = galactocentric_coords.cartesian.y.to(u.kpc).value
z = galactocentric_coords.cartesian.z.to(u.kpc).value

# rotate by 90 degrees x-y plane so that the Sun location is shown better in the plot
xtemp = np.copy(x)
x = -np.copy(y) #x*np.sqrt(2)/2 - y*np.sqrt(2)/2 #
y = np.copy(xtemp) #xtemp*np.sqrt(2)/2 + y*np.sqrt(2)/2
#%%



############## Plots ##############

#%%
# Function to plot a sphere
def plot_sphere(ax, center, radius):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.1)

# 3D plot on a sphere
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5))
#fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,7), projection='3d')
#ax = fig.add_subplot(111, projection='3d')
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
xlim = 13

#u = np.linspace(0, 2 * np.pi, 500)
#v = np.linspace(0, xlim, 50)
#x_disc = v[:, np.newaxis] * np.cos(u)
#y_disc = v[:, np.newaxis] * np.sin(u)
#color_intensity = np.exp(-np.sqrt(x_disc**2 + y_disc**2) / 5)
#0 * np.ones_like(x_disc)
#ax.plot_surface(x_disc, y_disc, np.zeros_like(x_disc), 
#                facecolors=plt.cm.Greys(color_intensity), norm=True, alpha=0.1, rstride=1, cstride=1)


# Plot the 3D points
markers = ['D', 'o', '^', 's', 'd', 'p']
colors = sns.color_palette("bright", len(pipelines)) 

for i_p, p_name in enumerate(pipelines): 
    x_temp = x[pulsars['suggested_pipeline'] == p_name]
    y_temp = y[pulsars['suggested_pipeline'] == p_name]
    z_temp = z[pulsars['suggested_pipeline'] == p_name]
    dist_temp = pulsars[pulsars['suggested_pipeline'] == p_name]['DIST'].to_numpy()
    ax.scatter(x_temp, y_temp, z_temp, s=50/(dist_temp)**(1/4), color=colors[i_p], marker=markers[i_p], label=p_name, alpha=0.7)
    for xi, yi, zi in zip(x_temp, y_temp, z_temp):
        ax.plot([xi, xi], [yi, yi], [zi, 0], color=colors[i_p], linestyle='--', linewidth=1)


ax.scatter(0, 0, 0, color='k', s=50,  marker='x', label='GC')
ax.scatter(0, -8.122, 0, color='k', s=100,  marker='$\odot$', label='Sun')

#plot_sphere(ax, [0, 0, 0], 3)
plot_sphere(ax, [0, 0, 0], 8.122)
#plot_sphere(ax, [0, 0, 0], 20)

ax.set_xlim(-xlim, xlim)
ax.set_ylim(-xlim, xlim)
ax.set_zlim(-xlim, xlim);
# Remove the background planes
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.plot([-xlim, xlim], [0, 0], [0, 0], color='gray', linestyle='-', linewidth=0.5)
ax.plot([-0, 0], [-xlim, xlim], [0, 0], color='gray', linestyle='-', linewidth=0.5)
ax.plot([-0, 0], [0, 0], [-xlim, xlim], color='gray', linestyle='-', linewidth=0.5)


# Remove the axes
ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Set the x-axis line color to fully transparent
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Set the y-axis line color to fully transparent
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))  # Set the z-axis line color to fully transparent

# Remove the axis ticks and labels
ax.set_xticks([])  # Remove x-axis ticks
ax.set_yticks([])  # Remove y-axis ticks
ax.set_zticks([])  # Remove z-axis ticks
ax.set_xticklabels([])  # Remove x-axis labels
ax.set_yticklabels([])  # Remove y-axis labels
ax.set_zticklabels([])  # Remove z-axis labels

# Set equal aspect ratio
ax.set_box_aspect([1,1,1])

#n_points = 1000
#X, Y = np.meshgrid(np.linspace(-xlim, xlim, n_points), np.linspace(-xlim, xlim, n_points))
#X[np.sqrt(X**2 + Y**2) > 5] = np.nan
#Y[np.sqrt(X**2 + Y**2) > 5] = np.nan
#Z = (-0 * X - 0 * Y) * 1. 
#ax.plot_surface(X, Y, Z, color='gray', alpha=0.1)

theta = np.linspace(0, 2*np.pi, 100)
x_circle = 8.122 * np.cos(theta)
y_circle = 8.122 * np.sin(theta)
z_circle = np.zeros_like(theta)  # Z-coordinate of the circle points
ax.plot(x_circle, y_circle, z_circle, color='gray', alpha=0.8, linestyle='--', linewidth=1)

theta = np.linspace(0, 2*np.pi, 100)
x_circle = 5 * np.cos(theta)
y_circle = 5 * np.sin(theta)
z_circle = np.zeros_like(theta)  # Z-coordinate of the circle points
ax.plot(x_circle, y_circle, z_circle, color='gray', alpha=0.8, linewidth=1)

ax.legend()
# Data for three-dimensional scattered pointspoints
#ax.scatter3D(0, 0, 0, color='k', marker='x');
#ax.scatter3D(x, y, z, cmap='Greens');
#%%

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