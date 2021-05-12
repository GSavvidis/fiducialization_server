
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
from pretty_print import pretty_print
from functions import read_npulses
from functions import drop_laser_events
from functions import drop_events_by_index
from functions import select_events_by_query
from functions import count_pulses

'my classes'
from AbstractBaseClassCSV3 import GetCSVInfoMPA

'#print options'
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
flag_stacked = True

"""loop over all the processed runs and perform the matching of pulses"""
for indx, proc in enumerate(process):

    # matching of pulses for the following processing
    #print(f"Matching pulses for {proc}")
    
    # List of csv of DD_Vectors to read
    lst_vecs[indx] = directory + "tj13s000_MPA_" + proc + ".csv"
    #print(lst_vecs[indx])
		
    # List of processings 
    lst_procs[indx] = proc
    
    # Read csv file
    csv_vecs = GetCSVInfoMPA(lst_vecs, lst_procs)

    # Extract dictionary of dataframes
    dict_vecs = csv_vecs.load_csv(nrows)
    
    # Extract multiple dataframes from dictionary
    df_vecs = pd.DataFrame.from_dict(dict_vecs[proc])
    #print(df_vecs)
    #print("")
    

    # Select specific procs from the list of procs
    # df_vecs = df_vecs[lst_procs[0]:lst_procs[-1]]	

    # total events: +1 because index start from 0
    nevents = df_vecs.index.get_level_values("entry").max() + 1
    #print("Total number of events")
    #print(nevents)
    #print("")
    
    """ Drop laser events
    """
    level = "subentry"
    value_to_drop = 2
    df_vecs = drop_events_by_index(df_vecs, level, value_to_drop)

    # convert samples to micro seconds
    df_vecs["DD_VectorStartBin"] = df_vecs["DD_VectorStartBin"].mul(sampling_period)
    df_vecs["DD_VectorStopBin"] = df_vecs["DD_VectorStopBin"].mul(sampling_period)
    df_vecs["DD_VectorDeltaBinPrev"] = df_vecs["DD_VectorDeltaBinPrev"].mul(sampling_period)
    df_vecs["DD_VectorDeltaBinNext"] = df_vecs["DD_VectorDeltaBinNext"].mul(sampling_period)

    # filter out unphysical startbin
    #print("Selecting events with startbin > 0")
    cond = "(subentry == 0 | subentry == 1) & DD_VectorStartBin > 0"
    df_vecs = select_events_by_query(df_vecs, cond)

    #print(df_vecs)
    #print("")
    
    flag_1 = True
    iterations = 1000
        
    # starting clock
    start = time.time()
    ev = 9

    # remaining events after cuts
    lst_nevents = df_vecs.index.get_level_values("entry")

    # remove duplicates
    lst_nevents = list(set(lst_nevents))
    ##print(lst_nevents)
    
    # loop over total number of events
    for ev_idx, event in enumerate(lst_nevents):
        #event = ev
        #print(f"event = {event}")
        #if (event == 10):
        #    stop

        if (event%iterations == 0):
            
            t = time.time()
            
            # calculate elapsed time
            elapsed = t - start
            
            # convert the value passed to the function into seconds
            ty_res = time.gmtime(elapsed)
            
            # display the value passed, into hours and minutes
            elapsed = time.strftime("%H:%M:%S", ty_res)
            print(f"{event} events matched. elapsed time: {elapsed} ")

        #df_vecs = df_vecs.unstack(level=["subentry"])

        # number of north pulses
        channel_0 = 0
        channel_1 = 1
        pulses_north = count_pulses(df_vecs, event, channel_0)
        pulses_south = count_pulses(df_vecs, event, channel_1)

        #pulses_north = int(df_vecs[(df_vecs.index.get_level_values("entry") == event) & (df_vecs.index.get_level_values("subentry") == channel_0)].index.get_level_values("DD_NPulses")[0])
        #pulses_south = int(df_vecs[(df_vecs.index.get_level_values("entry") == event) & (df_vecs.index.get_level_values("subentry") == channel_1)].index.get_level_values("DD_NPulses")[0])

        ##print("Number of North pulses")
        ##print(pulses_north)
        ##print("Number of South pulses")
        ##print(pulses_south)
        ##print("")


        ##print("Unstacking subentry and DD_NPulses")
        #df_vecs = df_vecs.unstack(level=["subentry", "DD_NPulses"])
        #flag_stacked = False
        ##print(df_vecs)
        ##print("")

        # select pulses of same event
        ##print(f"Getting the cross-section for event {ev}")
        df_event = df_vecs.xs(event, level="entry", drop_level=False)
        ##print(df_event)
        ##print("")

        col1 = df_event.query("subentry == 0 | subentry == 1")["DD_VectorStartBin"] 
        col2 = df_event.query("subentry == 0 | subentry == 1")["TimeS"].mul(10e-6) 
        col3 = df_event.query("subentry == 0 | subentry == 1")["TimeMuS"] 
        new_col = col1 + col2 + col3
        df_event["DD_VectorStartBinTime"] = new_col


        ## drop columns that have all their values nan
        ##print("Dropping NaN values from df_event")
        #df_event = df_event.dropna(axis=1, how="all")
        ##print("NaN values from df_event dropped")
        ##print(df_event)
        ##print("")

        # get column at DD_NPulses level for North(0) and South(1) channel
        #col_0 = df_event.columns.get_level_values("DD_NPulses")[0]
        #col_1 = df_event.columns.get_level_values("DD_NPulses")[1]

        ##print("col_0")
        ##print(col_0)
        ##print("")

        ##print("col_1")
        ##print(col_1)
        ##print("")

        # correct for the time offset between north/south channel
        # units in microseconds
        #df_event["DD_VectorStartBinTime", 0, col_0] =  df_event["DD_VectorStartBin", 0, col_0]\
        #                                             + df_event["TimeS", 0, col_0].mul(10e-6)\
        #                                             + df_event["TimeMuS", 0, col_0]

        #df_event["DD_VectorStartBinTime", 1, col_1] =  df_event["DD_VectorStartBin", 1, col_1]\
        #                                             + df_event["TimeS", 1, col_1].mul(10e-6)\
        #                                             + df_event["TimeMuS", 1, col_1]
        
        # Find elements which are not NaN values
        #mask_npulses_0 = df_event["DD_VectorStartBinTime", 0, col_0].notnull()
        #mask_npulses_1 = df_event["DD_VectorStartBinTime", 1, col_1].notnull()
        #
        ## drop NaN(False) values = select rows corresponding to pulses
        #ser_npulses_0 = df_event["DD_VectorStartBinTime", 0, col_0][mask_npulses_0]
        #ser_npulses_1 = df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1]

        # find number of rows for current event
        nrows = len(df_event.index) 

        # find number of olumns for current event
        ncolumns = len(df_event.columns)

        #pulses_north = df_vecs.xs(event, level="entry").DD_VectorStartBin[0].count() # count() returns number of non-NaN values
        #pulses_north = int(df_npulses["npulses_north"].iloc[event])

        # number of south pulses
        #pulses_south = df_vecs.xs(event, level="entry").DD_VectorStartBin[1].count() # count() returns number of non-NaN values
        #pulses_south = int(df_npulses["npulses_south"].iloc[event])
        
        # number of laser pulses
        #pulses_laser = df_vecs.xs(event, level="entry").DD_VectorStartBin[2].count() # count() returns number of non-NaN values

        ##print("Number of Laser pulses")
        ##print(pulses_laser)

        # unstack DD_NPulses index level
        #df_vecs = df_vecs.unstack(level="DD_NPulses")
        
        if (pulses_south < pulses_north and pulses_south != 0):

            # size of list should be equal to npulses_north*npulses_south
            # and correspond to all possible delta-startbin
            lst_delta = [[None]*int(pulses_north)]*int(pulses_south)
            #lst_delta_laser = [[None]*int(pulses_laser)]*int(pulses_south)
            
            # numbers to be used as columns in the dataframe
            # below. 
            n_columns = np.arange(pulses_north)
            #n_columns_laser = np.arange(pulses_laser)
            
            # create DataFrame to store delta-startbin
            df_delta_start = pd.DataFrame(data=lst_delta, columns=n_columns)
            #df_delta_start_laser = pd.DataFrame(data=lst_delta_laser, columns=n_columns_laser)

            # list with size equal to the number of 
            lst_min_delta = [None]*int(pulses_south)
            #lst_min_delta_laser = [None]*int(pulses_south)

            #
            lst_min_delta_pulse_number_south = [None]*int(pulses_south)
            lst_min_delta_pulse_number_north = [None]*int(pulses_south)
            #lst_min_delta_pulse_number_laser = [None]*int(pulses_south)

            
            for i in range(pulses_south):
                # vector_indexes = npulses_north = number of
                # pulses in the north
                #for j, ind_north in enumerate(v_indexes_north):
                for j in range(pulses_north):


                    val_1 = df_event.query("subentry == 1")["DD_VectorStartBinTime"].iloc[i]
                    val_0 = df_event.query("subentry == 0")["DD_VectorStartBinTime"].iloc[j]
                    df_delta_start.iloc[i, j] = val_1 - val_0

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
                    #df_delta_start.iloc[i, j] = df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1].iloc[i]\
                    #                          - df_event["DD_VectorStartBinTime", 0, col_0][mask_npulses_0].iloc[j]
                

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
                    note: len(lst_min_delta_pulse_number_north) = len(lst_min_delta) 
                """
                # List that contains north pulses
                lst_min_delta_pulse_number_north[i] = df_delta_start_numeric_cells.idxmin()
                
                """ Store index of south pulse which gives the minimum delta_startbin
                """
                # List that contains south pulses
                lst_min_delta_pulse_number_south[i] = i
                
            #""" Convert class 'pandas.indexes.numeric.Int64Index' to numpy
            #    in order to add it as column to df_first_match dataframe below
            #""" 
           
            #lst_min_delta_laser[0] = abs(df_delta_start_laser.min().values)
            #""" The index of df_delta_start_laser coincides with the pulse 
            #    number in the South
            #"""
            #lst_min_delta_pulse_number_laser[0] = abs(df_delta_start_laser.min().index.values)
            
            ##print(df_delta_start_laser)
            ##print(lst_min_delta_pulse_number_laser)
            ##print(lst_min_delta_laser)
            
            #pretty_print(df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1])
            #pretty_print(df_event["DD_VectorStartBinTime", 0, col_0][mask_npulses_0])
            #print(lst_min_delta_pulse_number_north)
            #print(lst_min_delta_pulse_number_south)
            #print(lst_min_delta)

            """ Create dictionary to store the delta_startbins and the associated
                pulses corresponding to the minima. This dictionary will be used
                as input to a new dataframe exactly below
            """
            dict_first_match = {"min_delta_i": lst_min_delta, 
                                "pulse_number_south": lst_min_delta_pulse_number_south,
                                "pulse_number_north": lst_min_delta_pulse_number_north}

            #print(dict_first_match)
            #print("")
            
            """ Dataframe containing the delta_startbins and the associated
                pulses corresponding to the minima. 
            """
            df_first_match = pd.DataFrame(data=dict_first_match)
            #print("Best match")
            #print(df_first_match)
            #print("")

            # flag to jump iterations from ind to new_ind
            flag_2 = False
           
            # check if same pulse is matched multiple times with other pulses
            # note: len(df_first_match['index_south']) = number of delta startbins =  npulses_north 
            for ind, pulse_number_north in enumerate(df_first_match['pulse_number_north']):

                #print(f"ind={ind}")
                """ Select instances of same pulse from df_first_match, that is, same pulse
                    matched many time with other pulses, and construct a new dataframe 
                    called df_instances
                """
                df_instances = df_first_match.query("pulse_number_north == '%s' " %str(pulse_number_north))
                instances = len(df_instances)
                #print("Dataframe df_instances")
                #print(df_instances)
                #print("")
                #print("Instances")
                #print(instances)
                #print("")
                #print("df_first_match['pulse_number_north']")
                #print(df_first_match['pulse_number_north'])
               
                # flag to jump iterations from ind to new_ind
                #if(flag_2 == True):
                #    if(ind in (ind, new_ind)):
                #        continue

                #if(instances > 1):
                #    flag_2 = True
                #    
                #    how_many = instances
                #    #print(how_many)

                #    """ In the dataframe df_instances find the minimum delta
                #        startbin value and the corresponding index 
                #    """
                #    kept_min_delta = np.min(df_instances["min_delta_i"])
                #    
                #    """ The index coincides with the south pulse number. That means that
                #        kept_min_delta_idx = pulse number south that gives the minimum
                #        delta_startbin with the north. 
                #    """
                #    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
                #    

                #    """ Return the north and south pulse associated with the minimum delta 
                #        startbin from df_instances
                #    """
                #    kept_pulse_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_north"]
                #    kept_pulse_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_south"]

                #    # keep only the values of the dataframe without the indexes
                #    kept_pulse_north = int(kept_pulse_north.to_string(index=False))
                #    kept_pulse_south = int(kept_pulse_south.to_string(index=False))
                #    #print("kept_pulse_north")
                #    #print(kept_pulse_north)
                #    #print("")
                #    #print("kept_pulse_south")
                #    #print(kept_pulse_south)
                #    #print("")

                #    # separate channels into dataframes
                #    df_north = df_event.xs(0, level="subentry", drop_level=False, axis=1)
                #    df_south = df_event.xs(1, level="subentry", drop_level=False, axis=1)
                #    df_laser = df_event.xs(2, level="subentry", drop_level=False, axis=1)
                #    
                #    #df_south = df_south.dropna(axis=0, how="all")
                #    #df_laser = df_laser.dropna(axis=0, how="all")

                #    #df_north = df_north.unstack(level="entry")
                #    #df_south = df_south.unstack(level="entry")
                #    #df_laser = df_laser.unstack(level="entry")
                #    #pretty_print(df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False))
                #    #pretty_print(df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False))
                #    ##print(df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False))
                #    ##print(df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False))
                #    
                #    df_north = df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False)
                #    df_south = df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False)

                #    
                #    #print("kept_min_delta")
                #    #print(kept_min_delta)
                #    #print("")
                #    #print("kept_min_delta_idx")
                #    #print(kept_min_delta_idx)
                #    #print("")
                #    #print("kept_min_delta_laser")
                #    #print(kept_min_delta_laser)
                #    #print("")
                #    #print("kept_min_delta_laser_idx")
                #    #print(kept_min_delta_laser_idx)
                #    #print("")
                #    stop

                #    #df_north = df_north.dropna(axis=1, how="all")
                #    #df_south = df_south.dropna(axis=1, how="all")

                #    #df_north = df_north.reset_index(level="subsubentry")
                #    #df_south = df_south.reset_index(level="subsubentry")
                #    #pretty_print(df_north)
                #    #pretty_print(df_south)
                #    #pretty_print(df_laser)
                #    #stop
                #    ##print(df_north)
                #    ##print(df_south)

                #    df_match = pd.concat([df_north, df_south], axis=0)
                #    df_match_laser = pd.concat([df_north, df_south], axis=0)
                #    
                #    """ Apply function to drop nan values and re-arange the dataframe.
                #        In this way, matching is achieved
                #    """
                #    df_match = df_match.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

                #    #stop
                #    #pretty_print(df_match)
                #    ##print(df_match)

                #    #pretty_print(df_event)
                #    #pretty_print(df_event.xs([35,kept_pulse_north], level=["entry", "subsubentry"], drop_level=False, axis=0))
                #    #pretty_print(df_event.xs(0, level="subentry", drop_level=False, axis=1))
                #    ##print(df_event.xs([35,kept_pulse_north], level=["entry", "subsubentry"], drop_level=False).columns[3])
                #    #df_1 = df_vecs.loc[kept_pulse_north:kept_pulse_north, :"timeMuS_north"]
                #    #df_2 = df_vecs.loc[kept_pulse_south:kept_pulse_south, "number_south":]
                #    #df_match = pd.concat([df_1, df_2], axis=1)
                #    
                #    # number of pulses remaining to be checked
                #    remaining = len(df_first_match['index_north']) - instances
                #    new_ind = ind + remaining
                #    if(i == 2): 
                #        stop
                #    
                #    if (flag_1 == True):                                                         
                #        # export data to csv
                #        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                #            # index = False otherwise first column is a comma
                #            df_match.to_csv(f, index=False, header=False)
                #            flag_1 = False
               
                #    elif (flag_1 == False):
                #        # export data to csv
                #        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
                #            # index = False otherwise first column is a comma
                #            df_match.to_csv(f, index=False, header=None)

                if(instances == 1):

                    """ In the dataframe df_instances find the minimum delta
                        startbin value and the corresponding index 
                    """
                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    
                    """ The index coincides with the south pulse number. That means that
                        kept_min_delta_idx = pulse number south that gives the minimum
                        delta_startbin with the north. 
                    """
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
                    
                    """ Return the north and south pulse associated with the minimum delta 
                        startbin from df_instances
                    """
                    kept_pulse_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_north"]
                    kept_pulse_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_south"]

                    # keep only the values of the dataframe without the indexes
                    kept_pulse_north = int(kept_pulse_north.to_string(index=False))
                    kept_pulse_south = int(kept_pulse_south.to_string(index=False))

                    # separate channels into dataframes
                    #df_north = df_event.xs(0, level="subentry", drop_level=False, axis=1)
                    #df_south = df_event.xs(1, level="subentry", drop_level=False, axis=1)

                    df_north = df_event.query(f"subentry == 0 & subsubentry == {kept_pulse_north}")
                    df_south = df_event.query(f"subentry == 1 & subsubentry == {kept_pulse_south}")

                    #df_north = df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False)
                    #df_south = df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False)

                    
                    #print("kept_min_delta")
                    #print(kept_min_delta)
                    #print("")
                    #print("kept_min_delta_idx")
                    #print(kept_min_delta_idx)
                    #print("")
                    
                    df_match_final = pd.concat([df_north, df_south], axis=0)
                    """ Apply function to drop nan values and re-arange the dataframe.
                        In this way, matching is achieved
                    """
                    #df_match_final = df_match_final.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

                    #print("df_match_final")
                    #print(df_match_final)
                    #print("")

                    #print(f"i={i}")
                    #if(ind == 2): 
                    #    stop
                    
                    if (flag_1 == True):                                                         
                        """ Create the csv file
                        """
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                            df_match_final.to_csv(f, index=True, header=True)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        """ Write on the existing csv file
                        """
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
                            df_match_final.to_csv(f, index=True, header=None)

        elif (pulses_north <= pulses_south and pulses_north != 0):

            # size of list should be equal to npulses_north*npulses_south
            # and correspond to all possible delta-startbin
            lst_delta = [[None]*int(pulses_south)]*int(pulses_north)
            
            # numbers to be used as columns in the dataframe
            # below. 
            n_columns = np.arange(pulses_south)
            
            # create DataFrame to store delta-startbin
            df_delta_start = pd.DataFrame(data=lst_delta, columns=n_columns)

            # list with size equal to the number of 
            lst_min_delta = [None]*int(pulses_north)

            #
            lst_min_delta_pulse_number_south = [None]*int(pulses_north)
            lst_min_delta_pulse_number_north = [None]*int(pulses_north)

            
            for i in range(pulses_north):
                # vector_indexes = npulses_north = number of
                # pulses in the north
                #for j, ind_north in enumerate(v_indexes_north):
                for j in range(pulses_south):


                    val_1 = df_event.query("subentry == 1")["DD_VectorStartBinTime"].iloc[j]
                    val_0 = df_event.query("subentry == 0")["DD_VectorStartBinTime"].iloc[i]

                    # north precedes
                    # units in microseconds
                    df_delta_start.iloc[i, j] = val_1 - val_0

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
                    note: len(lst_min_delta_pulse_number_north) = len(lst_min_delta) 
                """
                # List that contains north pulses
                lst_min_delta_pulse_number_south[i] = df_delta_start_numeric_cells.idxmin()
                
                """ Store index of south pulse which gives the minimum delta_startbin
                """
                # List that contains south pulses
                lst_min_delta_pulse_number_north[i] = i
                
            #print(lst_min_delta_pulse_number_north)
            #print(lst_min_delta_pulse_number_south)
            #print(lst_min_delta)

            """ Create dictionary to store the delta_startbins and the associated
                pulses corresponding to the minima. This dictionary will be used
                as input to a new dataframe exactly below
            """
            dict_first_match = {"min_delta_i": lst_min_delta, 
                                "pulse_number_south": lst_min_delta_pulse_number_south,
                                "pulse_number_north": lst_min_delta_pulse_number_north}

            #print(dict_first_match)
            #print("")
            
            """ Dataframe containing the delta_startbins and the associated
                pulses corresponding to the minima. 
            """
            df_first_match = pd.DataFrame(data=dict_first_match)
            #print("Best match")
            #print(df_first_match)
            #print("")

            # flag to jump iterations from ind to new_ind
            flag_2 = False
           
            # check if same pulse is matched multiple times with other pulses
            # note: len(df_first_match['index_south']) = number of delta startbins =  npulses_north 
            for ind, pulse_number_south in enumerate(df_first_match['pulse_number_south']):

                """ Select instances of same pulse from df_first_match, that is, same pulse
                    matched many time with other pulses, and construct a new dataframe 
                    called df_instances
                """
                df_instances = df_first_match.query("pulse_number_south == '%s' " %str(pulse_number_south))
                instances = len(df_instances)
                #print("Dataframe df_instances")
                #print(df_instances)
                #print("")
                #print("Instances")
                #print(instances)
                #print("")
                #print("df_first_match['pulse_number_south']")
                #print(df_first_match['pulse_number_south'])
               
                # flag to jump iterations from ind to new_ind
                #if(flag_2 == True):
                #    if(ind in (ind, new_ind)):
                #        continue

                #if(instances > 1):
                #    flag_2 = True
                #    
                #    how_many = instances
                #    #print(how_many)

                #    """ In the dataframe df_instances find the minimum delta
                #        startbin value and the corresponding index 
                #    """
                #    kept_min_delta = np.min(df_instances["min_delta_i"])
                #    
                #    """ The index coincides with the south pulse number. That means that
                #        kept_min_delta_idx = pulse number south that gives the minimum
                #        delta_startbin with the north. 
                #    """
                #    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
                #    

                #    """ Return the north and south pulse associated with the minimum delta 
                #        startbin from df_instances
                #    """
                #    kept_pulse_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_north"]
                #    kept_pulse_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_south"]

                #    # keep only the values of the dataframe without the indexes
                #    kept_pulse_north = int(kept_pulse_north.to_string(index=False))
                #    kept_pulse_south = int(kept_pulse_south.to_string(index=False))
                #    #print("kept_pulse_north")
                #    #print(kept_pulse_north)
                #    #print("")
                #    #print("kept_pulse_south")
                #    #print(kept_pulse_south)
                #    #print("")

                #    # separate channels into dataframes
                #    df_north = df_event.xs(0, level="subentry", drop_level=False, axis=1)
                #    df_south = df_event.xs(1, level="subentry", drop_level=False, axis=1)
                #    df_laser = df_event.xs(2, level="subentry", drop_level=False, axis=1)
                #    
                #    #df_south = df_south.dropna(axis=0, how="all")
                #    #df_laser = df_laser.dropna(axis=0, how="all")

                #    #df_north = df_north.unstack(level="entry")
                #    #df_south = df_south.unstack(level="entry")
                #    #df_laser = df_laser.unstack(level="entry")
                #    #pretty_print(df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False))
                #    #pretty_print(df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False))
                #    ##print(df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False))
                #    ##print(df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False))
                #    
                #    df_north = df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False)
                #    df_south = df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False)

                #    
                #    #print("kept_min_delta")
                #    #print(kept_min_delta)
                #    #print("")
                #    #print("kept_min_delta_idx")
                #    #print(kept_min_delta_idx)
                #    #print("")
                #    #print("kept_min_delta_laser")
                #    #print(kept_min_delta_laser)
                #    #print("")
                #    #print("kept_min_delta_laser_idx")
                #    #print(kept_min_delta_laser_idx)
                #    #print("")
                #    stop

                #    #df_north = df_north.dropna(axis=1, how="all")
                #    #df_south = df_south.dropna(axis=1, how="all")

                #    #df_north = df_north.reset_index(level="subsubentry")
                #    #df_south = df_south.reset_index(level="subsubentry")
                #    #pretty_print(df_north)
                #    #pretty_print(df_south)
                #    #pretty_print(df_laser)
                #    #stop
                #    ##print(df_north)
                #    ##print(df_south)

                #    df_match = pd.concat([df_north, df_south], axis=0)
                #    df_match_laser = pd.concat([df_north, df_south], axis=0)
                #    
                #    """ Apply function to drop nan values and re-arange the dataframe.
                #        In this way, matching is achieved
                #    """
                #    df_match = df_match.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

                #    #stop
                #    #pretty_print(df_match)
                #    ##print(df_match)

                #    #pretty_print(df_event)
                #    #pretty_print(df_event.xs([35,kept_pulse_north], level=["entry", "subsubentry"], drop_level=False, axis=0))
                #    #pretty_print(df_event.xs(0, level="subentry", drop_level=False, axis=1))
                #    ##print(df_event.xs([35,kept_pulse_north], level=["entry", "subsubentry"], drop_level=False).columns[3])
                #    #df_1 = df_vecs.loc[kept_pulse_north:kept_pulse_north, :"timeMuS_north"]
                #    #df_2 = df_vecs.loc[kept_pulse_south:kept_pulse_south, "number_south":]
                #    #df_match = pd.concat([df_1, df_2], axis=1)
                #    
                #    # number of pulses remaining to be checked
                #    remaining = len(df_first_match['index_north']) - instances
                #    new_ind = ind + remaining
                #    if(i == 2): 
                #        stop
                #    
                #    if (flag_1 == True):                                                         
                #        # export data to csv
                #        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                #            # index = False otherwise first column is a comma
                #            df_match.to_csv(f, index=False, header=False)
                #            flag_1 = False
               
                #    elif (flag_1 == False):
                #        # export data to csv
                #        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
                #            # index = False otherwise first column is a comma
                #            df_match.to_csv(f, index=False, header=None)

                if(instances == 1):

                    """ In the dataframe df_instances find the minimum delta
                        startbin value and the corresponding index 
                    """
                    kept_min_delta = np.min(df_instances["min_delta_i"])
                    
                    """ The index coincides with the south pulse number. That means that
                        kept_min_delta_idx = pulse number south that gives the minimum
                        delta_startbin with the north. 
                    """
                    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
                    
                    """ Return the north and south pulse associated with the minimum delta 
                        startbin from df_instances
                    """
                    kept_pulse_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_north"]
                    kept_pulse_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_south"]

                    # keep only the values of the dataframe without the indexes
                    kept_pulse_north = int(kept_pulse_north.to_string(index=False))
                    kept_pulse_south = int(kept_pulse_south.to_string(index=False))

                    # separate channels into dataframes
                    #df_north = df_event.xs(0, level="subentry", drop_level=False, axis=1)
                    #df_south = df_event.xs(1, level="subentry", drop_level=False, axis=1)

                    df_north = df_event.query(f"subentry == 0 & subsubentry == {kept_pulse_north}")
                    df_south = df_event.query(f"subentry == 1 & subsubentry == {kept_pulse_south}")

                    #df_north = df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False)
                    #df_south = df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False)

                    
                    #print("kept_min_delta")
                    #print(kept_min_delta)
                    #print("")
                    #print("kept_min_delta_idx")
                    #print(kept_min_delta_idx)
                    #print("")
                    
                    df_match_final = pd.concat([df_north, df_south], axis=0)
                    """ Apply function to drop nan values and re-arange the dataframe.
                        In this way, matching is achieved
                    """
                    #df_match_final = df_match_final.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

                    #print("df_match_final")
                    #print(df_match_final)
                    #print("")

                    #print(f"i={i}")
                    #if(ind == 2): 
                    #    stop
                    
                    if (flag_1 == True):                                                         
                        """ Create the csv file
                        """
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
                            df_match_final.to_csv(f, index=True, header=True)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        """ Write on the existing csv file
                        """
                        # export data to csv
                        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
                            df_match_final.to_csv(f, index=True, header=None)

        


#print("Finished matching")
end = time.time()
#print("Total run time: ", end-start)

#""" Treatment of laser pulses
#"""
#if (flag_laser == True):
#
#    # number of NPulses of laser channel
#    col_2 = df_event.columns.get_level_values("DD_NPulses")[2]
#    #print("col_2")
#    #print(col_2)
#    #print("")

#    df_event["DD_VectorStartBinTime", 2, col_2] =  df_event["DD_VectorStartBin", 2, col_2]\
#                                                 + df_event["TimeS", 2, col_2].mul(10e-6)\
#                                                 + df_event["TimeMuS", 2, col_2]

#    mask_npulses_2 = df_event["DD_VectorStartBinTime", 2, col_2].notnull()
#    ser_npulses_2 = df_event["DD_VectorStartBinTime", 2, col_2][mask_npulses_2]
#
#elif (pulses_laser == 0):
#    """ To treat events where there is no laser pulse, we
#        create a dataframe of same size as df_event with same
#        column names and we fill it with zeroes
#    """
#    #df_event["DD_VectorStartBinTime", 2, col_2] = 0.0
#    #df_laser_zeroes = df_event.stack(level="subentry").xs(1, level="subentry", drop_level=False)
#    ##print("df_laser_zeroes")
#    ##print(df_laser_zeroes)
#    #stop
#if (pulses_laser == 1):
#
#    df_laser = df_event.xs(2, level="subentry", drop_level=False, axis=1)
#    kept_min_delta_laser = np.min(df_instances["min_delta_laser_i"]) 
#    
#    """ In a similar manner, the south pulse
#        that should match with the laser pulse is retrieved
#    """
#    kept_min_delta_laser_idx = df_instances["min_delta_laser_i"].idxmin()
#    
#    """ Select the laser pulse number associated with
#        the min_delta_laser
#    """
#    kept_pulse_number_laser = df_instances.query("min_delta_laser_i == '%s'" %str(kept_min_delta_laser))["pulse_number_laser"]
#    ##print(kept_pulse_number_laser)
#
#    # drop index
#    kept_pulse_number_laser = int(kept_pulse_number_laser.to_string(index=False))
#
#    # construct laser dataframe
#    df_laser = df_laser.xs(kept_pulse_number_laser, level="subsubentry", axis=0, drop_level=False)
#
#elif (pulses_laser == 0):
#    kept_min_delta_laser = 0.0 
#    
#    """ In a similar manner, the south pulse
#        that should match with the laser pulse is retrieved
#    """
#    kept_min_delta_laser_idx = 0.0
#    kept_pulse_number_laser = df_instances.query("min_delta_laser_i == '%s'" %str(kept_min_delta_laser))["pulse_number_laser"]
#    kept_pulse_number_laser = float(kept_pulse_number_laser.to_string(index=False))
#    #print(kept_pulse_number_laser)
#
#    """ Constructing the laser dataframe. The size of the frame
#        will be identical to the size of df_south. All column values
#        will be replaced be zeroes. DD_NPulses level will be set to 0.0,
#        subentry level will be set to 2
#    """
#    df_laser = df_south.xs(0, level="subsubentry", axis=0, drop_level=False)
#    df_laser = df_laser.stack(level=["DD_NPulses", "subentry"])
#    #print("df_laser")
#    #print(df_laser.index)
#    #print("")
#    as_list = df_laser.index.tolist()
#    ##print(as_list[0])
#    #stop
#    #as_list = df_laser.index.values[0][0]
#    #print(as_list)
#    #for i, val in enumerate(as_list[0]):
#
#    idx = as_list[0].index(3.0)
#    #print(idx)
#    as_list[0][idx] = 0
#    df_laser.index = as_list
#    #print(df_laser.index)
#    stop
#    df_laser.index = df_laser.index.set_levels(df_laser.index.levels[0])
#    stop
#    as_list[idx] = 0
#    df_laser.index = as_list
#    #df_laser = df_laser.reindex(0, level=["DD_NPulses"])
#    #print("df_laser")
#    #print(df_laser)
#    #print("")
#    stop
#    ##print(df_laser.columns.levels[0])
#    #laser_subentry = df_laser.columns.get_level_values("subentry")
#    ##print(laser_subentry)
#    #laser_subentry = 2
#
#    """ Returns a list of size equal with the number of columns in df_south and with values
#        equal to DD_NPulses of South channel. 
#    """
#    npulses_south = df_south.xs(0, level="subsubentry", axis=0, drop_level=False).columns.get_level_values("DD_NPulses")[0] # [0] to keep only the first element of the list
#    npulses_laser = 0.0
#
#    """ Dict to replace subentry 1(South channel) with subentry 2(Laser)
#    """
#    dict_replace_subentry = {1:2}
#    dict_replace_npulses = {npulses_south:npulses_laser}
#    #print("npulses_south")
#    #print(npulses_south)
#    #print("")
#
#    """ List of size equal to the number of columns in df_laser and which
#        contains the new value of subentry, that is 2
#    """
#    new_vals_subentry = [dict_replace_subentry[x] for x in df_laser.columns.get_level_values("subentry")]
#    new_vals_npulses = [dict_replace_npulses[x] for x in df_laser.columns.get_level_values("DD_NPulses")]
#    col_len = len(df_event.columns.levels[0])
#    new_vals_subentry = [None]*col_len
#    new_vals_npulses = [None]*col_len
#    for i in range(col_len):
#        new_vals_subentry[i] = i+1
#        new_vals_npulses[i] = 0.0
#    
#    #print("new_vals_subentry")
#    #print(new_vals_subentry)
#    #print("")
#    """ At the column level subentry, replace all old values(1=South channel) with 
#        new values(2=Laser channel)
#    """
#    #df_laser.columns.set_levels(df_laser.columns.levels["subentry"].replace(1,2), level="subentry",verify_integrity=True, inplace=True)
#    #df_laser.columns.set_levels(new_vals_subentry, level="subentry",verify_integrity=False, inplace=True)
#    #df_laser = df_laser.columns.reindex(new_vals_subentry, level="subentry")
#    df_laser.columns = df_laser.columns.set_levels(new_vals_subentry, level="subentry",verify_integrity=True, inplace=False)
#    df_laser.columns = df_laser.columns.set_levels(new_vals_npulses, level="DD_NPulses",verify_integrity=False, inplace=False)
#    #print("df_laser")
#    #print(df_laser)
#    
#    #df_laser.columns = df_laser.columns.set_levels(new_vals_subentry, level="subentry",verify_integrity=False, inplace=False)
#    #df_laser.columns = df_laser.columns.set_levels(new_vals_npulses, level="DD_NPulses",verify_integrity=False, inplace=False)
#    #print("df_south")
#    #print(df_south)
#    #print("")
#    #print("df_laser")
#    #print(df_laser)
#    #print("")
#
#    ##print(df_laser.columns.get_level_values("subentry")[0])
#    #stop
#    ##print(df_laser.columns.values[0][1])
#    #stop
#    col_names = df_laser.columns.levels[0]
#    ##print(col_names)
#    for i, name in enumerate(col_names):
#        #df_laser.columns.values[i][1] = 2
#        #df_laser.columns.values[i][2] = 0.0
#        df_laser[f"{name}"].values[0][0] = 0.0
#
#    #print(df_laser)
##print("kept_min_delta_laser")
##print(kept_min_delta_laser)
##print("")
##print("kept_min_delta_laser_idx")
##print(kept_min_delta_laser_idx)
##print("")
#
###print("df_south")
###print(df_south)
###print("")
###print("df_laser")
###print(df_laser)
###print("")
#df_match_laser = pd.concat([df_south, df_laser], axis=0)
###print("df_match_laser")
###print(df_match_laser)
###print("")
#
#""" Apply function to drop nan values and re-arange the dataframe.
#    In this way, matching is achieved
#"""
#df_match_laser = df_match_laser.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')
###print("df_match_laser")
###print(df_match_laser)
###print("")
#
##df_match_test = pd.concat([df_north, df_south], axis=0)
##df_match_test = df_match_test.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')
