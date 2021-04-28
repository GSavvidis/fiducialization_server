
import sys
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/functions/')
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/classes/')

import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time


'my functions'
from match_pulses_v6 import match_pulses_v6

'my classes'
from AbstractBaseClassCSV2 import GetCSVInfo

'print options'
np.set_printoptions(threshold = 20, edgeitems=10)

'disable SettingWithCopyWarning warning'
pd.options.mode.chained_assignment = None  # default='warn'

processing = np.array(["q41", "q42", "q43", "q44", "q45", "q46", "q47", "q48", "q49", "q50", 
					"q51", "q52", "q53", "q54", "q55", "q56", "q57", "q58"])
process = ["q13"]
lst_vecs = [None]*len(process)
lst_procs = [None]*len(process)
lst_npulses = [None]*len(process)
sampling_period = 1./1.041670 # 1/MHz
nchannels = 2
plafon = 100

# seconds
interval = 360

# directory to save matched pulses
directory = '/home/gsavvidis/notebooks/'

"""loop over all the processed runs and perform the matching of pulses"""
for indx, proc in enumerate(process):

    # matching of pulses for the following processing
    print(f"Matching pulses for {proc}")
    
    # List of csv of DD_Vectors to read
    lst_vecs[indx] = "/home/gsavvidis/notebooks/DD_Vectors_" + proc + ".csv"
		
    # List of csv of N_Pulses to read
    lst_npulses[indx] = "/home/gsavvidis/notebooks/DD_NPulses_" + proc + ".csv"
    
    # List of processings 
    lst_procs[indx] = proc
    
    # Read csv files (dd_vectors, npulses)
    csv_vecs = GetCSVInfo(lst_vecs, lst_procs)
    csv_npulses = GetCSVInfo(lst_npulses, lst_procs)

    # Extract dictionary of dataframes
    dict_vecs = csv_vecs.load_csv()
    dict_npulses = csv_npulses.load_csv()
    
    # Extract multiple dataframes from dictionary
    # df_v = pd.concat(dict_vecs.plafons(), keys=dict_vecs.keys())
    df_v = pd.DataFrame.from_dict(dict_vecs[proc])
    df_npulses = pd.DataFrame.from_dict(dict_npulses[proc])

    # Select specific procs from the list of procs
    # df_v = df_v[lst_procs[0]:lst_procs[-1]]	

    # total events
    nevents = len(df_npulses)

    # list of variables to be converted from samples to micro seconds
    lst_cols = ["startbin_north", "startbin_south",
                "stopbin_north", "stopbin_south",
                "db_prev_south", "db_prev_north",
                "db_next_north", "db_next_south"]

    # convert samples to micro seconds
    df_v[lst_cols] = df_v[lst_cols].mul(sampling_period)

    # filter out unphysical startbin plafons
    df_v = df_v.query("startbin_north >= 0 & startbin_south >= 0")

    # 
    flag_1 = True
    iterations = 1000

        
    # starting clock
    start = time.time()
    
    # loop over total number of events
    for event in range(nevents):
        
        if (event%iterations == 0):
            
            t = time.time()
            
            # calculate elapsed time
            elapsed = t - start
            
            # convert the value passed to the function into seconds
            ty_res = time.gmtime(elapsed)
            
            # display the value passed, into hours and minutes
            elapsed = time.strftime("%H:%M:%S", ty_res)
            print(f"{event} events matched. elapsed time: {elapsed} ")

       # with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
       # 	print(df_v[lst_cols])

        # number of north pulses
        pulses_north = int(df_npulses["npulses_north"].iloc[event])

        # number of south pulses
        pulses_south = int(df_npulses["npulses_south"].iloc[event])
        
        if (pulses_north <= pulses_south):

            """ use f-string to reference local variable
            option to reference column labels
            """
            col_n = "event"
            # select pulses of same event
            df_event = df_v.query(f"{col_n} == {event}")
            
            # select vector indexes of north pulses
            # note: v_indexes_north == pulses_north
            v_indexes_north = df_v.query(f"{col_n} == {event}").index[0:pulses_north]
            
            # select vector indexes of north pulses
            # note: v_indexes_south == pulses_south
            v_indexes_south = df_v.query(f"{col_n} == {event}").index[0:pulses_south]
            
            # correct for the time offset between north/south channel
            # units in microseconds
            lst_time_startbin_north = [df_event["startbin_north"]\
                                     + df_event["timeS_north"].mul(10e6)\
                                     + df_event["timeMuS_north"]]
            
            # units in microseconds
            lst_time_startbin_south = [df_event["startbin_south"]\
                                     + df_event["timeS_south"].mul(10e6)\
                                     + df_event["timeMuS_south"]]
            
            # add as new columns in DataFrame
            df_event["time_startbin_north"] = lst_time_startbin_north[0]
            df_event["time_startbin_south"] = lst_time_startbin_south[0]
            
            # size of list should be equal to npulses_north*npulses_south
            # and correspond to all possible delta-startbin
            lst_delta = [[None]*int(pulses_south)]*int(pulses_north)
            
            # number of possible pulse comparisons
            n_comparisons = v_indexes_south
            
            # create DataFrame to store delta-startbin
            df_delta_start = pd.DataFrame(data=lst_delta, columns=n_comparisons)

            # list to store delta startbins 
            lst_min_delta = [None]*int(pulses_north)
            
            # lists to store 
            lst_min_delta_south_idx = [None]*int(pulses_north)
            lst_min_delta_north_idx = [None]*int(pulses_north)

            # loop to calculate (delta-startbin)ij and store it in the DataFrame
            for i, ind_north in enumerate(v_indexes_north):

                # vector_indexes = npulses_south = number of
                # pulses in the south
                for j, ind_south in enumerate(v_indexes_south):

                    # assign delta-startbin to the element in position i,col in the DataFrame
                    # note: north precedes units in microseconds
                    df_delta_start.iloc[i, j] = df_event["time_startbin_north"].iloc[i]\
                                              - df_event["time_startbin_south"].iloc[j]
                    
                # get the minimum delta_startbin between i-th north pulse 
                # and all south pulses
                lst_min_delta[i] = np.min(abs(df_delta_start.iloc[i]))
                
                # transform cell value to numeric ones in order to 
                # access the index with the .idxmin() method
                numeric_cells = pd.to_numeric(abs(df_delta_start.iloc[i]))
               
                # store index of minimum delta_startbin.
                # the index of the minimum delta_startbin
                # corresponds to the pulse number in the south
                # associated with the minimum
                # note: len(lst_min_delta_south_idx) = len(lst_min_delta) 
                lst_min_delta_south_idx[i] = numeric_cells.idxmin()

                # store index of the north matched pulse corresponding
                # to the df_v
                lst_min_delta_north_idx[i] = ind_north
                
            dict_best_match = {"min_delta_i": lst_min_delta, 
                               "index_south": lst_min_delta_south_idx,
                               "index_north": lst_min_delta_north_idx}

            df_best_match = pd.DataFrame(data=dict_best_match)

            # check if same pulse is matched multiple times with other
            # pulses
            n_indexes = (len(df_best_match['index_south']))
            
            # flag to jump iterations from ind to new_ind
            flag_2 = False
           
            # note: len(df_best_match['index_south']) = npulses_north
            for ind, south_v_index in enumerate(df_best_match['index_south']):
                
                df_instances = df_best_match.query("index_south == '%s' " %str(south_v_index))
                instances = len(df_instances)
               
                # flag to jump iterations from ind to new_ind
                if(flag_2 == True):
                    if(ind in (ind, new_ind)):
                        continue

                if(instances > 1):
                    flag_2 = True
                    
                    how_many = instances

                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()

                    kept_v_idx_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_north"]
                    kept_v_idx_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_south"]

                    # keep only the values of the dataframe without the indexes
                    kept_v_idx_north = int(kept_v_idx_north.to_string(index=False))
                    kept_v_idx_south = int(kept_v_idx_south.to_string(index=False))

                    # number of pulses remaining to be checked
                    remaining = len(df_best_match['index_south']) - instances
                    new_ind = ind + remaining
                    
                    df_1 = df_v.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    df_2 = df_v.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_match = pd.concat([df_1, df_2], axis=1)
                    
                    # first if is to create and write in the csv file if it has not been already created
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=False)
                            flag_1 = False
                    
                    # the elif is to write in the csv file if it has already been created
                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=None)

                elif(instances == 1):

                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()

                    kept_v_idx_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_north"]
                    kept_v_idx_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_south"]

                    # avoid indexes
                    kept_v_idx_north = int(kept_v_idx_north.to_string(index=False))
                    kept_v_idx_south = int(kept_v_idx_south.to_string(index=False))


                    df_1 = df_v.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    df_2 = df_v.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_match = pd.concat([df_1, df_2], axis=1)
                    
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=False)
                            flag_1 = False

                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=None)
                
                #if (min_val < plafon):
                         
                       # if (len(v_indexes_south > 1) and j > 0):
                       #     delta_start_prev = df_delta_start.iloc[i,j] - df_delta_start_north_south.iloc[i,j-1]

        
        elif (pulses_south < pulses_north):

            """ use f-string to reference local variable
            option to reference column labels
            """
            col_n = "event"
            # select pulses of same event
            df_event = df_v.query(f"{col_n} == {event}")
            
            # select vector indexes of north pulses
            # note: v_indexes_north == pulses_north
            v_indexes_south = df_v.query(f"{col_n} == {event}").index[0:pulses_south]
            
            # select vector indexes of north pulses
            # note: v_indexes_north == pulses_north
            v_indexes_north = df_v.query(f"{col_n} == {event}").index[0:pulses_north]
            
            # correct for the time offset between north/south channel
            # units in microseconds
            lst_time_startbin_north = [df_event["startbin_north"]\
                                     + df_event["timeS_north"].mul(10e6)\
                                     + df_event["timeMuS_north"]]
            
            # units in microseconds
            lst_time_startbin_south = [df_event["startbin_south"]\
                                     + df_event["timeS_south"].mul(10e6)\
                                     + df_event["timeMuS_south"]]
            
            # add as new columns in DataFrame
            df_event["time_startbin_north"] = lst_time_startbin_north[0]
            df_event["time_startbin_south"] = lst_time_startbin_south[0]
            
            # size of list should be equal to npulses_north*npulses_south
            # and correspond to all possible delta-startbin
            lst_delta = [[None]*int(pulses_north)]*int(pulses_south)
            
            # number of possible pulse comparisons
            n_comparisons = v_indexes_north
            
            # create DataFrame to store delta-startbin
            df_delta_start = pd.DataFrame(data=lst_delta, columns=n_comparisons)

            # list with size equal to the number of 
            lst_min_delta = [None]*int(pulses_south)

            #
            lst_min_delta_south_idx = [None]*int(pulses_south)
            lst_min_delta_north_idx = [None]*int(pulses_south)

            # loop to calculate (delta-startbin)ij and store it in the DataFrame
            for i, ind_south in enumerate(v_indexes_south):

                # vector_indexes = npulses_north = number of
                # pulses in the north
                for j, ind_north in enumerate(v_indexes_north):

                    # assign delta-startbin to the element in position i,col in the DataFrame
                    # north precedes
                    # units in microseconds
                    df_delta_start.iloc[i, j] = df_event["time_startbin_north"].iloc[i]\
                                              - df_event["time_startbin_south"].iloc[j]
                    
                # get the minimum delta_startbin between i-th north pulse 
                # and all south pulses
                lst_min_delta[i] = np.min(abs(df_delta_start.iloc[i]))
                
                # transform cell value to numeric ones in order to 
                # access the index with the .idxmin() method
                numeric_cells = pd.to_numeric(abs(df_delta_start.iloc[i]))
               
                # store index of minimum delta_startbin.
                # the index of the minimum delta_startbin
                # corresponds to the pulse number in the south
                # associated with the minimum
                # note: len(lst_min_delta_north_idx) = len(lst_min_delta) 
                lst_min_delta_north_idx[i] = numeric_cells.idxmin()

                # store index of the north matched pulse corresponding
                # to the df_v
                lst_min_delta_south_idx[i] = ind_south
                
            dict_best_match = {"min_delta_i": lst_min_delta, 
                               "index_south": lst_min_delta_south_idx,
                               "index_north": lst_min_delta_north_idx}

            df_best_match = pd.DataFrame(data=dict_best_match)

            # length of stored indexes in the dataframe = number of delta startbin indexes = number of delta startbins
            n_indexes = (len(df_best_match['index_north']))
            
            # flag to jump iterations from ind to new_ind
            flag_2 = False
           
            # check if same pulse is matched multiple times with other pulses
            # note: len(df_best_match['index_south']) = number of delta startbins =  npulses_north 
            for ind, north_v_index in enumerate(df_best_match['index_north']):

                df_instances = df_best_match.query("index_north == '%s' " %str(north_v_index))
                instances = len(df_instances)
               
                # flag to jump iterations from ind to new_ind
                if(flag_2 == True):
                    if(ind in (ind, new_ind)):
                        continue

                if(instances > 1):
                    flag_2 = True
                    
                    how_many = instances

                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()

                    kept_v_idx_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_north"]
                    kept_v_idx_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_south"]

                    # keep only the values of the dataframe without the indexes
                    kept_v_idx_north = int(kept_v_idx_north.to_string(index=False))
                    kept_v_idx_south = int(kept_v_idx_south.to_string(index=False))
                    
                    df_1 = df_v.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    df_2 = df_v.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_match = pd.concat([df_1, df_2], axis=1)
                    
                    # number of pulses remaining to be checked
                    remaining = len(df_best_match['index_north']) - instances
                    new_ind = ind + remaining
                    
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=False)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=None)

                elif(instances == 1):

                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()

                    kept_v_idx_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_north"]
                    kept_v_idx_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_south"]

                    # avoid indexes
                    kept_v_idx_north = int(kept_v_idx_north.to_string(index=False))
                    kept_v_idx_south = int(kept_v_idx_south.to_string(index=False))

                    df_1 = df_v.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    df_2 = df_v.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_match = pd.concat([df_1, df_2], axis=1)
                    
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=False)
                            flag_1 = False

                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_DD_Vectors_" + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=None)


print("Finished matching")
end = time.time()
print("Total run time: ", end-start)

