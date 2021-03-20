
import sys
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/functions/')
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/classes/')

import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import timeit


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
process = ["q24"]
lst_vecs = [None]*len(process)
lst_procs = [None]*len(process)
lst_npulses = [None]*len(process)
sampling_period = 1./1.041670 # 1/MHz
nchannels = 2
value = 100

# directory to save matched pulses
directory = '/home/gsavvidis/notebooks/'



"""loop over all the processed runs and perform the matching of pulses"""
for indx, proc in enumerate(process):

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
    # df_v = pd.concat(dict_vecs.values(), keys=dict_vecs.keys())
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

    # filter out unphysical startbin values
    df_v = df_v.query("startbin_north >= 0 & startbin_south >= 0")

    # 
    flag = True
    iterations = 20000

    # loop over total number of events
    for event in range(nevents):
        
        if (event%iterations == 0):
            print(f"{event} events matched")

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
            # note: vector_indexes_north == pulses_north
            vector_indexes_north = df_v.query(f"{col_n} == {event}").index[0:pulses_north]
            
            # select vector indexes of north pulses
            # note: vector_indexes_south == pulses_south
            vector_indexes_south = df_v.query(f"{col_n} == {event}").index[0:pulses_south]
            
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
            n_comparisons = vector_indexes_south
            
            # create DataFrame to store delta-startbin
            df_delta_start = pd.DataFrame(data=lst_delta, columns=n_comparisons)

            # loop to calculate (delta-startbin)ij and store it in the DataFrame
            for i,ind_north in enumerate(vector_indexes_north):
                
                # vector_indexes = npulses_south = number of
                # pulses in the south
                for j,ind_south in enumerate(vector_indexes_south):
                  
                    # assign delta-startbin to the element in position i,col in the DataFrame
                    # north precedes
                    # units in microseconds
                    df_delta_start.iloc[i, j] = df_event["time_startbin_north"].iloc[i]\
                                              - df_event["time_startbin_south"].iloc[j]

                    if (abs(df_delta_start.iloc[i, j]) < value):

                        df_1 = df_v.loc[ind_north:ind_north, :"timeMuS_north"]
                        df_2 = df_v.loc[ind_south:ind_south, "number_south":]
                        df_match = pd.concat([df_1, df_2], axis=1)

                        if (flag == True):
                            # export data to csv
                            with open(directory + "matched_DD_Vectors_" + proc + ".csv","w") as f:
                                # index = False otherwise first column is a comma
                                df_match.to_csv(f, index=False, header=False)
                            
                            # turn to false to prevent writing on the same file 
                            # with headers twice
                            flag = False

                        elif (flag == False):
                            # export data to csv
                            with open(directory + "matched_DD_Vectors_" + proc + ".csv","a") as f:
                                # index = False otherwise first column is a comma
                                df_match.to_csv(f, index=False, header=None)
                    
            for pulses in range(len(df_match)):
                for i, start_south in enumerate(df_match["startbin_south"]):
                    for j, start_north in enumerate(df_match["startbin_north"]):

                        print("len(df_match)=", len(df_match))
                        print("start_south=", start_south)
                        print("start_north=", start_north)
                        stop

        if (pulses_south < pulses_north):
            """use f-string to reference local variable
              option to reference column labels
            """
            col_n = "event"
            # select pulses of same event
            df_event = df_v.query(f"{col_n} == {event}")

            # select vector indexes of north pulses
            # note: vector_indexes_north == pulses_north
            vector_indexes_north = df_v.query(f"{col_n} == {event}").index[0:pulses_north]

            # select vector indexes of north pulses
            # note: vector_indexes_south == pulses_south
            vector_indexes_south = df_v.query(f"{col_n} == {event}").index[0:pulses_south]

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

            # create list to store delta startbin for pulses
            # size of list should be equal to npulses_north*npulses_south
            # and correspond to all possible delta-startbin
            lst_delta = [[None]*int(pulses_north)]*int(pulses_south)	
            
            # number of possible pulse comparisons
            n_comparisons = vector_indexes_north

            # create DataFrame to store delta-startbin
            df_delta_start = pd.DataFrame(data=lst_delta, columns=n_comparisons)

            # loop to calculate (delta-startbin)ij and store it in the DataFrame
            for i,ind_south in enumerate(vector_indexes_south):
            
                # vector_indexes = npulses_south = number of
                # pulses in the south
                for j,ind_north in enumerate(vector_indexes_north):

                    # assign delta-startbin to the element in position i,col in the DataFrame
                    # north precedes
                    # units in microseconds
                    df_delta_start.iloc[i, j] = df_event["time_startbin_north"].iloc[i]\
                                              - df_event["time_startbin_south"].iloc[j]

                    if (abs(df_delta_start.iloc[i, j]) < value):
                            df_1 = df_v.loc[ind_north:ind_north, :"timeMuS_north"]
                            df_2 = df_v.loc[ind_south:ind_south, "number_south":]
                            df_match = pd.concat([df_1, df_2], axis=1)

                            if (flag == True):
                               # export data to csv
                               with open(directory + "matched_DD_Vectors_" + proc + ".csv","w") as f:
                                   # index = False otherwise first column is a comma
                                   df_match.to_csv(f, index=False, header=False)
                               
                               # turn to false to prevent writing on the same file 
                               # with headers twice
                               flag = False

                            elif (flag == False):
                                # export data to csv
                                with open(directory + "matched_DD_Vectors_" + proc + ".csv","a") as f:
                                    # index = False otherwise first column is a comma
                                    df_match.to_csv(f, index=False, header=None)

#            for indx, dij in enumerate(df_delta_start):
#                print(df_match[["event", "startbin_north", "startbin_south"]])
#                stop




    print("Finished matching")
