
import sys
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/functions/')
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/classes/')

import os.path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from pretty_print import pretty_print


'my functions'
from match_pulses_v6 import match_pulses_v6

'my classes'
from AbstractBaseClassCSV3 import GetCSVInfoMPA

'print options'
np.set_printoptions(threshold = 20, edgeitems=10)

'disable SettingWithCopyWarning warning'
pd.options.mode.chained_assignment = None  # default='warn'

processing = np.array(["q41", "q42", "q43", "q44", "q45", "q46", "q47", "q48", "q49", "q50", 
					"q51", "q52", "q53", "q54", "q55", "q56", "q57", "q58"])
process = ["q29"]
lst_vecs = [None]*len(process)
lst_procs = [None]*len(process)
lst_npulses = [None]*len(process)
sampling_period = 1./1.041670 # 1/MHz
nchannels = 2
plafon = 100
nrows = 10000

# seconds
interval = 360

# directory to save matched pulses
directory = '/home/gsavvidis/csv_files/'

"""loop over all the processed runs and perform the matching of pulses"""
for indx, proc in enumerate(process):

    # matching of pulses for the following processing
    print(f"Matching pulses for {proc}")
    
    # List of csv of DD_Vectors to read
    lst_vecs[indx] = directory + "tj13s000_MPA_" + proc + ".csv"
    print(lst_vecs[indx])
		
    # List of processings 
    lst_procs[indx] = proc
    
    # Read csv file
    csv_vecs = GetCSVInfoMPA(lst_vecs, lst_procs)

    # Extract dictionary of dataframes
    dict_vecs = csv_vecs.load_csv(nrows)
    
    # Extract multiple dataframes from dictionary
    # df_vecs = pd.concat(dict_vecs.plafons(), keys=dict_vecs.keys())
    df_vecs = pd.DataFrame.from_dict(dict_vecs[proc])
    df_vecs = df_vecs.unstack(level="subentry") # subentry = Channel
    #df_vecs = df_vecs.unstack(level=("DD_NPulses"))

    #with pd.option_context('display.max_rows', 100, 'display.max_columns', None):  # more options can be specified also
    #    print(df_vecs.head(15))
    
    # Select specific procs from the list of procs
    # df_vecs = df_vecs[lst_procs[0]:lst_procs[-1]]	

    # total events: +1 because index start from 0
    nevents = df_vecs.index.get_level_values("entry").max() + 1
    
    # list of variables to be converted from samples to micro seconds
    lst_cols = ["startbin_north", "startbin_south",
                "stopbin_north", "stopbin_south",
                "db_prev_south", "db_prev_north",
                "db_next_north", "db_next_south"]
    
    lst_cols = ["DD_VectorStartBin", "DD_VectorStopBin"]

    # convert samples to micro seconds
    df_vecs["DD_VectorStartBin"] = df_vecs["DD_VectorStartBin"].mul(sampling_period)
    df_vecs["DD_VectorStopBin"] = df_vecs["DD_VectorStopBin"].mul(sampling_period)
    df_vecs["DD_VectorDeltaBinPrev"] = df_vecs["DD_VectorDeltaBinPrev"].mul(sampling_period)
    df_vecs["DD_VectorDeltaBinNext"] = df_vecs["DD_VectorDeltaBinNext"].mul(sampling_period)

    #pretty_print(df_event.DD_VectorStartBin[0])
    #pretty_print(df_event.DD_VectorStartBin[1])
    
    # filter out unphysical startbin plafons
    # for that purpose, the df needs to be
    # unstacked up to the level of subentries,
    # not DD_NPulses
    mask_unphysical = pd.eval("df_vecs.DD_VectorStartBin[0] < 0 | df_vecs.DD_VectorStartBin[1] < 0")
   
    # get indices where the values are true
    inds_unphysical = mask_unphysical.index[mask_unphysical]
    df_vecs = df_vecs.drop(labels=inds_unphysical, axis=0)


    # 
    flag_1 = True
    iterations = 1000
        
    # starting clock
    start = time.time()
    
    # loop over total number of events
    for event in range(35,nevents,1):
        
        if (event%iterations == 0):
            
            t = time.time()
            
            # calculate elapsed time
            elapsed = t - start
            
            # convert the value passed to the function into seconds
            ty_res = time.gmtime(elapsed)
            
            # display the value passed, into hours and minutes
            elapsed = time.strftime("%H:%M:%S", ty_res)
            print(f"{event} events matched. elapsed time: {elapsed} ")

        
        # number of north pulses
        pulses_north = df_vecs.xs(event, level="entry").DD_VectorStartBin[0].count() # count() returns number of non-NaN values
        #pulses_north = int(df_npulses["npulses_north"].iloc[event])

        # number of south pulses
        pulses_south = df_vecs.xs(event, level="entry").DD_VectorStartBin[1].count() # count() returns number of non-NaN values
        #pulses_south = int(df_npulses["npulses_south"].iloc[event])
        
        # number of laser pulses
        pulses_laser = df_vecs.xs(event, level="entry").DD_VectorStartBin[2].count() # count() returns number of non-NaN values
        
        # unstack DD_NPulses index level
        df_vecs = df_vecs.unstack(level="DD_NPulses")
        
        if (pulses_north <= pulses_south):

            """ use f-string to reference local variable
            option to reference column labels
            """
            col_n = "event"
            # select pulses of same event
            #df_event = df_vecs.query(f"{col_n} == {event}")
            df_event = df_vecs.xs(event, level="entry")
            print(df_event)
            stop
            
            # select vector indexes of north pulses
            # note: v_indexes_north == pulses_north
            v_indexes_north = df_vecs.query(f"{col_n} == {event}").index[0:pulses_north]
            
            # select vector indexes of north pulses
            # note: v_indexes_south == pulses_south
            v_indexes_south = df_vecs.query(f"{col_n} == {event}").index[0:pulses_south]
            
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
                    # note: north precedes, units in microseconds
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
                # to the df_vecs
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
                    
                    df_1 = df_vecs.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    df_2 = df_vecs.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_match = pd.concat([df_1, df_2], axis=1)
                    
                    # first if is to create and write in the csv file if it has not been already created
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=False)
                            flag_1 = False
                    
                    # the elif is to write in the csv file if it has already been created
                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
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


                    df_1 = df_vecs.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    df_2 = df_vecs.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_match = pd.concat([df_1, df_2], axis=1)
                    
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=False)
                            flag_1 = False

                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
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
            #df_event = df_vecs.query(f"{col_n} == {event}")
            df_event = df_vecs.xs(35, level="entry", drop_level=False)

            # drop columns that have all their values nan
            df_event = df_event.dropna(axis=1, how="all")

            # get column at DD_NPulses level for North(0) and South(1) channel
            col_0 = df_event.columns.get_level_values("DD_NPulses")[0]
            col_1 = df_event.columns.get_level_values("DD_NPulses")[1]
            col_2 = df_event.columns.get_level_values("DD_NPulses")[2]
            
            # correct for the time offset between north/south channel
            # units in microseconds
            df_event["DD_VectorStartBinTime", 0, col_0] =  df_event["DD_VectorStartBin", 0, col_0]+(df_event["TimeS", 0, col_0].mul(10e6))+df_event["TimeMuS", 0, col_0]
            df_event["DD_VectorStartBinTime", 1, col_1] =  df_event["DD_VectorStartBin", 1, col_1]+(df_event["TimeS", 1, col_1].mul(11e6))+df_event["TimeMuS", 1, col_1]
            df_event["DD_VectorStartBinTime", 2, col_2] =  df_event["DD_VectorStartBin", 2, col_2]+(df_event["TimeS", 2, col_2].mul(22e6))+df_event["TimeMuS", 2, col_2]

            # Find elements which are not NaN values
            mask_npulses_0 = df_event["DD_VectorStartBinTime", 0, col_0].notnull()
            mask_npulses_1 = df_event["DD_VectorStartBinTime", 1, col_1].notnull()
            mask_npulses_2 = df_event["DD_VectorStartBinTime", 2, col_2].notnull()

            # drop NaN(False) values = select rows corresponding to pulses
            ser_npulses_0 = df_event["DD_VectorStartBinTime", 0, col_0][mask_npulses_0]
            ser_npulses_1 = df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1]
            ser_npulses_2 = df_event["DD_VectorStartBinTime", 2, col_2][mask_npulses_2]

            # size of list should be equal to npulses_north*npulses_south
            # and correspond to all possible delta-startbin
            lst_delta = [[None]*int(pulses_north)]*int(pulses_south)
            lst_delta_laser = [[None]*int(pulses_laser)]*int(pulses_south)
            
            # numbers to be used as columns in the dataframe
            # below. 
            n_columns = np.arange(pulses_north)
            n_columns_laser = np.arange(pulses_laser)
            
            # create DataFrame to store delta-startbin
            df_delta_start = pd.DataFrame(data=lst_delta, columns=n_columns)
            df_delta_start_laser = pd.DataFrame(data=lst_delta_laser, columns=n_columns_laser)

            # list with size equal to the number of 
            lst_min_delta = [None]*int(pulses_south)
            lst_min_delta_laser = [None]*int(pulses_laser)

            #
            lst_min_delta_south_idx = [None]*int(pulses_south)
            lst_min_delta_north_idx = [None]*int(pulses_south)
            lst_min_delta_laser_idx = [None]*int(pulses_laser)

            # loop to calculate (delta-startbin)ij and store it in the DataFrame
            #for i, ind_south in enumerate(v_indexes_south):
            for i in range(pulses_south):

                # vector_indexes = npulses_north = number of
                # pulses in the north
                #for j, ind_north in enumerate(v_indexes_north):
                for j in range(pulses_north):

                    """ Calculate delta_startbin between the i-th south pulse and the j-th north pulse.
                        Store the result in a DataFrame. The indexes i,j in df_delta_start correspond to
                        the i-th south pulse and to the j-th north pulse with respect to df_event. 
                        That means, df_delta_start.iloc[i, j] is the delta_startbin between the i-th and
                        j-th element of df_event. The values i,j themselves correspond to the index of
                        the i-th south pulse and the j-th north pulse, with respect to the position they
                        have in df_event for the given event.
                    """
                    # north precedes
                    # units in microseconds
                    df_delta_start.iloc[i, j] = df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1].iloc[i]\
                                              - df_event["DD_VectorStartBinTime", 0, col_0][mask_npulses_0].iloc[j]
                

                # vector_indexes = npulses_north = number of
                # pulses in the north
                #for j, ind_north in enumerate(v_indexes_north):
                for k in range(pulses_laser):

                    """ Same process but for laser
                    """
                    # units in microseconds
                    #print(df_event["DD_VectorStartBinTime", 2, col_2][mask_npulses_2].iloc[0])
                    df_delta_start_laser.iloc[i, k] = df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1].iloc[i]\
                                                    - df_event["DD_VectorStartBinTime", 2, col_2][mask_npulses_2].iloc[k]
                
                """ List in which the the minimum delta_startbin of the i-th
                    south pulse with all the north pulses is stored. 
                    Note: .iloc[i] returns elements corresponding to the i-th column.
                          Also, the number of columns is equal to the number
                          of south pulses and so, by typing .iloc[i], you select 
                          delta_startbins of the i-th south pulse, with all north pulses
                """
                # List that contains the minimum delta_startbin
                lst_min_delta[i] = np.min(abs(df_delta_start.iloc[i]))
                
                
                # transform cell value to numeric ones in order to 
                # access the index with the .idxmin() method
                df_delta_start_numeric_cells = pd.to_numeric(abs(df_delta_start.iloc[i]))
                
               
                """ Store index of minimum delta_startbin between the i-th south pulse and
                    all north pulses. The index df_delta_start_numeric_cells.idxmin()
                    is equal to the index of the north pulse in df_event for the
                    given event.
                    note: len(lst_min_delta_north_idx) = len(lst_min_delta) 
                """
                # List that contains north pulses
                lst_min_delta_north_idx[i] = df_delta_start_numeric_cells.idxmin()
                
                """ Store index of south pulse which gives the minimum delta_startbin
                """
                # List that contains south pulses
                lst_min_delta_south_idx[i] = i
            
            """ Convert class 'pandas.indexes.numeric.Int64Index' to numpy
                in order to add it as column to df_best_match dataframe below
            """ 
            lst_min_delta_laser[0] = abs(df_delta_start_laser.min().values)
            lst_min_delta_laser_idx[0] = abs(df_delta_start_laser.min().index.values)
            
            #print(df_delta_start_laser)
            #print(lst_min_delta_laser_idx)
            #print(lst_min_delta_laser)
            
            #pretty_print(df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1])
            #pretty_print(df_event["DD_VectorStartBinTime", 0, col_0][mask_npulses_0])
            print(lst_min_delta_north_idx)
            print(lst_min_delta_south_idx)
            print(lst_min_delta)

            """ Create dictionary to store the delta_startbins and the associated
                pulses corresponding to the minima. This dictionary will be used
                as input to a new dataframe exactly below
            """
            dict_best_match = {"min_delta_i": lst_min_delta, 
                               "index_south": lst_min_delta_south_idx,
                               "index_north": lst_min_delta_north_idx,
                               "index_laser": lst_min_delta_laser_idx}

            print(dict_best_match)
            """ Dataframe containing the delta_startbins and the associated
                pulses corresponding to the minima. 
            """
            df_best_match = pd.DataFrame(data=dict_best_match)
            print("Best match")
            print(df_best_match)
            print("")

            # length of stored indexes in the dataframe = number of delta startbin indexes = number of delta startbins
            n_indexes = (len(df_best_match['index_north']))
            
            # flag to jump iterations from ind to new_ind
            flag_2 = False
           
            # check if same pulse is matched multiple times with other pulses
            # note: len(df_best_match['index_south']) = number of delta startbins =  npulses_north 
            for ind, north_v_index in enumerate(df_best_match['index_north']):

                """ Select instances of same pulse from df_best_match, that is, same pulse
                    matched many time with other pulses, and construct a new dataframe 
                    called df_instances
                """
                df_instances = df_best_match.query("index_north == '%s' " %str(north_v_index))
                instances = len(df_instances)
                print("Dataframe df_instances")
                print(df_instances)
                print("")
                print("Instances")
                print(instances)
                print("")
               
                # flag to jump iterations from ind to new_ind
                if(flag_2 == True):
                    if(ind in (ind, new_ind)):
                        continue

                if(instances > 1):
                    flag_2 = True
                    
                    how_many = instances


                    """ In the dataframe df_instances find the minimum delta
                        startbin value and the corresponding index 
                    """
                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
                    print(kept_min_delta)
                    print(kept_min_delta_idx)
                    print("")

                    """ Return the north and south pulse associated with the minimum delta 
                        startbin from df_instances
                    """
                    kept_v_idx_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_north"]
                    kept_v_idx_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_south"]

                    # keep only the values of the dataframe without the indexes
                    kept_v_idx_north = int(kept_v_idx_north.to_string(index=False))
                    kept_v_idx_south = int(kept_v_idx_south.to_string(index=False))
                   
                  
                    df_1 = df_event
                    # separate channels into dataframes
                    df_north = df_event.xs(0, level="subentry", drop_level=False, axis=1)
                    df_south = df_event.xs(1, level="subentry", drop_level=False, axis=1)
                    df_laser = df_event.xs(2, level="subentry", drop_level=False, axis=1)
                    
                    #df_south = df_south.dropna(axis=0, how="all")
                    #df_laser = df_laser.dropna(axis=0, how="all")

                    #df_north = df_north.unstack(level="entry")
                    #df_south = df_south.unstack(level="entry")
                    #df_laser = df_laser.unstack(level="entry")
                    #pretty_print(df_north.xs(kept_v_idx_north, level="subsubentry", axis=0, drop_level=False))
                    #pretty_print(df_south.xs(kept_v_idx_south, level="subsubentry", axis=0, drop_level=False))
                    #print(df_north.xs(kept_v_idx_north, level="subsubentry", axis=0, drop_level=False))
                    #print(df_south.xs(kept_v_idx_south, level="subsubentry", axis=0, drop_level=False))
                    
                    df_north = df_north.xs(kept_v_idx_north, level="subsubentry", axis=0, drop_level=False)
                    df_south = df_south.xs(kept_v_idx_south, level="subsubentry", axis=0, drop_level=False)

                    #df_north = df_north.dropna(axis=1, how="all")
                    #df_south = df_south.dropna(axis=1, how="all")

                    #df_north = df_north.reset_index(level="subsubentry")
                    #df_south = df_south.reset_index(level="subsubentry")
                    #pretty_print(df_north)
                    #pretty_print(df_south)
                    #stop
                    #print(df_north)
                    #print(df_south)
                   

                    df_match = pd.concat([df_north, df_south], axis=0)
                    
                    """ Apply function to drop nan values and re-arange the dataframe.
                        In this way, matching is achieved
                     
                    """
                    df_match = df_match.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

                    #stop
                    #pretty_print(df_match)
                    #print(df_match)

                    #pretty_print(df_event)
                    #pretty_print(df_event.xs([35,kept_v_idx_north], level=["entry", "subsubentry"], drop_level=False, axis=0))
                    pretty_print(df_event.xs(0, level="subentry", drop_level=False, axis=1))
                    #print(df_event.xs([35,kept_v_idx_north], level=["entry", "subsubentry"], drop_level=False).columns[3])
                    stop
                    #df_1 = df_vecs.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    #df_2 = df_vecs.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    #df_match = pd.concat([df_1, df_2], axis=1)
                    
                    # number of pulses remaining to be checked
                    remaining = len(df_best_match['index_north']) - instances
                    new_ind = ind + remaining
                    
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=False)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=None)

                elif(instances == 1):

                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
                    #print(df_instances["min_delta_i"])
                    #print(df_instances["min_delta_i"].idxmin())

                    kept_v_idx_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_north"]
                    kept_v_idx_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["index_south"]

                    # avoid indexes
                    kept_v_idx_north = int(kept_v_idx_north.to_string(index=False))
                    kept_v_idx_south = int(kept_v_idx_south.to_string(index=False))
                    #print(kept_v_idx_north)
                    #print(kept_v_idx_south)
                    #df_1 = df_vecs.loc[kept_v_idx_north:kept_v_idx_north, :"timeMuS_north"]
                    #df_2 = df_vecs.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_1 = df_vecs.loc[kept_v_idx_north:kept_v_idx_north]
                    print(df_1)
                    stop
                    df_2 = df_vecs.loc[kept_v_idx_south:kept_v_idx_south, "number_south":]
                    df_match = pd.concat([df_1, df_2], axis=1)
                    
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=True)
                            flag_1 = False
                            stop

                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match.to_csv(f, index=False, header=None)


print("Finished matching")
end = time.time()
print("Total run time: ", end-start)

