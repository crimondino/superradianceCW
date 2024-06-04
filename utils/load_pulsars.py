#%%
import numpy as np
import pandas as pd

#%%

def load_pulsars_fnc():
    ### Read in data file with upper limits
    df_triplets = pd.read_csv('data/Fake_pulsars_triplets.csv', index_col=0)
    df_triplets = df_triplets[~np.isnan(df_triplets['upper_limits'])]
    df_triplets['type'] = 'triplet'

    df_doublets = pd.read_csv('data/Fake_pulsars_doublets.csv', index_col=0)
    df_doublets = df_doublets[~np.isnan(df_doublets['upper_limits'])]
    df_doublets.rename(columns={"upper limits": "upper_limits"}, inplace=True)
    df_doublets['type'] = 'doublet'

    df_fdots = pd.read_csv('data/Fake_pulsars_fdot.csv', index_col=0)
    df_fdots = df_fdots[~np.isnan(df_fdots['upper_limits'])]
    df_fdots['type'] = 'fdot'

    columns_to_use = ['NAME', 'RAJ', 'DECJ', 'F0', 'F1', 'F_GW', 'upper_limits', 'type', 
                      'f range or resolution [Hz]', 'fdot range or resolution [Hz/s]']
    df_pulsars = pd.concat([df_triplets[columns_to_use], df_doublets[columns_to_use], df_fdots[columns_to_use]]) 
    df_pulsars.sort_values(by='F_GW', inplace=True)
    df_pulsars.reset_index(inplace=True)
    
    return df_pulsars
#%%