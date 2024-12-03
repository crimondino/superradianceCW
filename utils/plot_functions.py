import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcdefaults()
from matplotlib import font_manager
from matplotlib import rcParams
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, LogLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
rcParams['mathtext.rm'] = 'Times New Roman' 
rcParams['text.usetex'] = True
rcParams['font.family'] = 'times' #'sans-serif'
font_manager.findfont('serif', rebuild_if_missing=True)
rcParams.update({'font.size':14})

DATA_DIR = '/Users/crimondino/Dropbox (PI)/myplasma/data'

def load_results(freqGWi_list, BHp_list, file_name_end):

    dfdlogh = []
    cum_dist = []

    for i, ni in enumerate(freqGWi_list):
        dfdlogh_temp = []
        cum_dist_fGW_temp = []
        for BHp_name in BHp_list:
            list_temp = np.load(DATA_DIR+'/disc_events/dndlogh_'+BHp_name+str(ni)+file_name_end+'.npy')
            dfdlogh_temp.append(list_temp[1:])

            cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )   
            for i_h in range(len(list_temp[1:, 0]-1)):
                cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
                cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
            cum_dist_fGW_temp.append(cum_dist_temp)

        dfdlogh.append(dfdlogh_temp)
        cum_dist.append(cum_dist_fGW_temp)

    return dfdlogh, cum_dist

def load_results_pulsars(freqGWi_list, BHp_list, file_name_end):

    dfdlogh = []
    cum_dist = []

    for i, ni in enumerate(freqGWi_list):
        dfdlogh_temp = []
        cum_dist_fGW_temp = []
        for BHp_name in BHp_list:
            list_temp = np.load(DATA_DIR+'/pulsars_events/dndlogF_'+BHp_name+str(ni)+file_name_end+'.npy')
            dfdlogh_temp.append(list_temp[1:])

            cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )   
            for i_h in range(len(list_temp[1:, 0]-1)):
                cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
                cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
            cum_dist_fGW_temp.append(cum_dist_temp)

        dfdlogh.append(dfdlogh_temp)
        cum_dist.append(cum_dist_fGW_temp)

    return dfdlogh, cum_dist


def log_format_func(value, tick_number):
    exponent = int(np.log10(value))

    return f"{int(value):d}" if ( (exponent == 0) ) else f"$10^{{{exponent}}}$"
