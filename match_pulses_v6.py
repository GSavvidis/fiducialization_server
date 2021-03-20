import sys
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/functions/')

import numpy as np


def match_pulses_v6(filename, dd_vector, npulses):

    nevents = range(npulses.shape[0])
    slice_ = np.array([])
    matched_dd_vector = np.array([])


    f = open(filename,"w")
    with f as csv_file:
        for event in nevents:
        # for event in range(10):

                # total inds of pulses of the same event
                inds_slice = np.argwhere(dd_vector[:, 0] == event)
                # slice of dd_vector with pulses of the same event
                slice_ = dd_vector[inds_slice]
                # reshape to make it n x m array
                slice_ = np.reshape(slice_,\
                                                  (slice_.shape[0]*slice_.shape[1],\
                                                   slice_.shape[2]))
                
                slice_len = range(len(inds_slice))

                sampling_period = 1./1.041670
                startbin_north = slice_[:, 2]*sampling_period
                startbin_south = slice_[:, 14]*sampling_period

                
                startbin_north = startbin_north[startbin_north > 0]
                startbin_south = startbin_south[startbin_south > 0]

                # print("startbin_north=", startbin_north)
                # print("startbin_north.shape =", startbin_north.shape)
                # print("startbin_south=", startbin_south)

                north_pulses = startbin_north.shape[0]
                south_pulses = startbin_south.shape[0]

                timeS_north = slice_[:, 11]
                timeMuS_north = slice_[:, 12]
                timeS_south = slice_[:, 23]
                timeMuS_south = slice_[:, 24]

                if(north_pulses <= south_pulses):
                    number_pulses = range(startbin_north.shape[0])
                    # iterate over the length of the slice 
                    for i in number_pulses:
                        # for the current pulse i in the north, calculate 
                        # the difference, Δstartbin, between that single pulse 
                        # and all the pulses in the south
                        di = abs((startbin_south + timeS_south + timeMuS_south) - (startbin_north[i] + timeS_north[i] + timeMuS_north[i]))

                        # for the pulse i, find the minimum difference
                        # Δstartbin between that single pulse and all the pulses
                        # in the south
                        minimum_di = np.amin(di)

                        # get the index of the pulse in the slice
                        # that corresponds to the minimum Δstartbin di
                        ind_minimum_di = np.argwhere(di == minimum_di)
                        # reshape to make it have shape (1)
                        ind_minimum_di = np.reshape(ind_minimum_di,\
                                          (ind_minimum_di.shape[0]*ind_minimum_di.shape[1]))


                        # if(ind_minimum_di.shape[0] > 1):
                        # 	ind_minimum_di = ind_minimum_di[0]

                        # get the index of the pulse in dd_vector
                        # which corresponds to the minimum Δstartbin di
                        index_north = inds_slice[i]
                        index_south = inds_slice[ind_minimum_di]
                        # reshape 
                        index_south = np.reshape(index_south,\
                                          (index_south.shape[0]*index_south.shape[1]))
                        
                        north = dd_vector[index_north, 0:13]
                        south = dd_vector[index_south, 13:]


                        for i in range(north.shape[1]):
                            csv_file.write(str(north[0, i]) + ",")

                        for i in range(south.shape[1]):
                            if(i == south.shape[1] - 1):
                                csv_file.write(str(south[0, i]) + "\n")
                            else:
                                csv_file.write(str(south[0, i])+",")

                                    
                    
                elif(south_pulses < north_pulses):
                    number_pulses = range(startbin_south.shape[0])
                    # iterate over the length of the slice 
                    for i in number_pulses:
                        # for the current pulse i in the north, calculate 
                        # the difference, Δstartbin, between that single pulse 
                        # and all the pulses in the south
                        di = abs((startbin_north + timeS_north + timeMuS_north) - (startbin_south[i] + timeS_south[i] + timeMuS_south[i]))

                        # for the pulse i, find the minimum difference
                        # Δstartbin between that single pulse and all the pulses
                        # in the south
                        minimum_di = np.amin(di)

                        # get the index of the pulse in the slice
                        # that corresponds to the minimum Δstartbin di
                        ind_minimum_di = np.argwhere(di == minimum_di)
                        ind_minimum_di = np.reshape(ind_minimum_di,\
                                          (ind_minimum_di.shape[0]*ind_minimum_di.shape[1]))

                        # if(ind_minimum_di.shape[0] > 1):
                        # 	ind_minimum_di = ind_minimum_di[0]

                        # get the index of the pulse in dd_vector
                        # which corresponds to the minimum Δstartbin di
                        index_south = inds_slice[i]
                        index_north = inds_slice[ind_minimum_di]
                        index_north = np.reshape(index_north,\
                                          (index_north.shape[0]*index_north.shape[1]))


                        north = dd_vector[index_north, 0:13]
                        south = dd_vector[index_south, 13:]

                        for i in range(north.shape[1]):
                            csv_file.write(str(north[0, i]) + ",")

                        for i in range(south.shape[1]):
                            if(i == south.shape[1] - 1):
                                csv_file.write(str(south[0, i]) + "\n")
                            else:
                                csv_file.write(str(south[0, i])+",")

                            
    
    f.close()
    

    return


