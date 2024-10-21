import numpy as np

def load_results(freqGWi_list, BHp_list, file_name_end):

    dfdlogh = []
    cum_dist = []

    for i, ni in enumerate(freqGWi_list):
        dfdlogh_temp = []
        cum_dist_fGW_temp = []
        for BHp_name in BHp_list:
            list_temp = np.load('data/disc_events/dndlogh_'+BHp_name+str(ni)+file_name_end+'.npy')
            freqGWi_list[i] = list_temp[0, 1]
            dfdlogh_temp.append(list_temp[1:])

            cum_dist_temp = np.zeros( (len(list_temp[1:, 0]-1), 2) )   
            for i_h in range(len(list_temp[1:, 0]-1)):
                cum_dist_temp[i_h, 0] = list_temp[1+i_h, 0]
                cum_dist_temp[i_h, 1] = np.trapz(list_temp[1+i_h:, 1]/list_temp[1+i_h:, 0], x=list_temp[1+i_h:, 0])
            cum_dist_fGW_temp.append(cum_dist_temp)

        dfdlogh.append(dfdlogh_temp)
        cum_dist.append(cum_dist_fGW_temp)

    return dfdlogh, cum_dist