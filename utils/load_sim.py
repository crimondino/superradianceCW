#%%
import numpy as np

from my_units import *
#%%
#%%
from os.path import dirname, abspath, join

THIS_DIR = dirname(__file__)
SIM_DIR = abspath(join(THIS_DIR, '..', 'data/galactic_BH_sim'))
SIM_DIR = '/Users/crimondino/Dropbox (PI)/myplasma/galactic_BH_sim'

print(SIM_DIR)
#%%

#%%
sim1 = np.loadtxt(SIM_DIR+'/Mmax20_spinMax0p5/1.00e-12_Mmax20_spinMax0p5.txt')
#%%

#%%
from load_pulsars import load_pulsars_fnc
pulsars = load_pulsars_fnc()
#%%