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
    #df_doublets.rename(columns={"upper limits": "upper_limits"}, inplace=True)
    df_doublets['type'] = 'doublet'

    df_fdots = pd.read_csv('data/Fake_pulsars_fdot.csv', index_col=0)
    df_fdots = df_fdots[~np.isnan(df_fdots['upper_limits'])]
    df_fdots['type'] = 'fdot'

    columns_to_use = ['NAME', 'RAJ', 'DECJ', 'F0', 'F1', 'DIST', 'F_GW', 'upper_limits', 'type', 'suggested_pipeline',
                      'f range or resolution [Hz]', 'fdot range or resolution [Hz/s]']
    df_pulsars = pd.concat([df_triplets[columns_to_use], df_doublets[columns_to_use], df_fdots[columns_to_use]]) 
    df_pulsars.sort_values(by='F_GW', inplace=True)
    df_pulsars.reset_index(inplace=True)

    list_names = ['J0418+6635', 'B1639+36A/J1641+3627A', 'J1844+0028g', 'J1911+0101B', 
                  'J1940+26', 'J1904+0836g', 'J1953+1844g', 'J1838-0022g']
    list_decj = ['+66:35:24.726', '+36:27:14.9788', '+00:28:48', '+01:01:50.44', '+26:01', '+08:36', '+18:44', '+00:22']
    for i in range(len(list_names)): 
        df_pulsars.loc[df_pulsars['NAME']==list_names[i], 'DECJ'] = list_decj[i]
        #df_pulsars[df_pulsars['NAME']==list_names[i]]['DECJ'] = list_decj[i]

    df_pulsars.replace('NB', 'Narrow Band', inplace=True) 
    df_pulsars.replace('analysed in O3', 'targeted', inplace=True) 
    df_pulsars.drop_duplicates(subset=['NAME', 'RAJ'], inplace=True)
    
    return df_pulsars