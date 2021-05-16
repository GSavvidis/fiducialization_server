
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
from functions import drop_events_mi
from functions import drop_events
from functions import select_events_by_query
from functions import get_min_delta_startbins
from functions import do_the_matching
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
nrows = None

flag_stacked = True
flag_drop_ontrigger = True

# seconds
interval = 360

# directory to save matched pulses
directory = '/home/gsavvidis/csv_files/'

if flag_drop_ontrigger == True:
    prefix = "offtrigger_matched_tj13s000_MPA_"

elif flag_drop_ontrigger == False:
    prefix = "matched_tj13s000_MPA_"

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

    """ Filter out unphysical startbin
    """
    #print("Selecting events with startbin > 0")
    #cond = "(subentry == 0 | subentry == 1) & DD_VectorStartBin > 0"
    #df_vecs = select_events_by_query(df_vecs, cond)
    #pretty_print(df_vecs)
    #print("")

    """ Drop ontrigger events
    """
    if flag_drop_ontrigger == True:

        cond_0 = "(DD_VectorStartBin > 3500 & DD_VectorStartBin < 4500)" 
        cond_1 = "(DD_VectorStartBin > 3500 & DD_VectorStartBin < 4500)" 

        print("Dropping ontrigger events")
        df_vecs = drop_events_mi(df_vecs, cond_0, cond_1, unstacked=False)
        print("Offtrigger events successfully selected")

    pretty_print(df_vecs)
    print("")
    
    flag_1 = True
    iterations = 1000
        
    # starting clock
    start = time.time()
    ev = 6

    # remaining events after cuts
    lst_nevents = df_vecs.index.get_level_values("entry")
    #print(lst_nevents)

    """ Remove duplicates and sort the final list
    """
    lst_nevents = list(set(lst_nevents))
    lst_nevents = sorted(lst_nevents) 
    #print(lst_events)
    
    # loop over total number of events
    for n, event in enumerate(lst_nevents):
        #event = lst_nevents[n]
        #print(f"n = {n}, event = {event}")
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


        # number of north pulses
        channel_0 = 0
        channel_1 = 1
        pulses_north = count_pulses(df_vecs, event, channel_0)
        pulses_south = count_pulses(df_vecs, event, channel_1)


        # select pulses of same event
        df_event = df_vecs.xs(event, level="entry", drop_level=False)
        #print(df_event)
        #print("")

        col1 = df_event.query("subentry == 0 | subentry == 1")["DD_VectorStartBin"] 
        col2 = df_event.query("subentry == 0 | subentry == 1")["TimeS"].mul(10e-6) 
        col3 = df_event.query("subentry == 0 | subentry == 1")["TimeMuS"] 
        new_col = col1 + col2 + col3
        df_event["DD_VectorStartBinTime"] = new_col

        # find number of rows for current event
        nrows = len(df_event.index) 

        # find number of olumns for current event
        ncolumns = len(df_event.columns)

        
        if (pulses_south < pulses_north and pulses_south != 0):

            lst_min_delta, lst_min_pulse_south, lst_min_pulse_north = get_min_delta_startbins(df_event, pulses_south, pulses_north)

            """ Create dictionary to store the delta_startbins and the associated
                pulses corresponding to the minima. This dictionary will be used
                as input to a new dataframe exactly below
            """
            dict_first_match = {"min_delta_i": lst_min_delta, 
                                "pulse_number_south": lst_min_pulse_south,
                                "pulse_number_north": lst_min_pulse_north} 

            """ Dataframe containing the delta_startbins and the associated
                pulses corresponding to the minima. 
            """
            df_first_match = pd.DataFrame(data=dict_first_match)

            # flag to jump iterations from ind to new_ind
            flag_2 = False
           
            # check if same pulse is matched multiple times with other pulses
            # note: len(df_first_match['index_south']) = number of delta startbins =  npulses_north 
            for ind, pulse_number_north in enumerate(df_first_match['pulse_number_north']):

                """ Select instances of same pulse from df_first_match, that is, same pulse
                    matched many time with other pulses, and construct a new dataframe 
                    called df_instances
                """
                df_instances = df_first_match.query("pulse_number_north == '%s' " %str(pulse_number_north))
                instances = len(df_instances)
               
                # flag to jump iterations from ind to new_ind
                if(flag_2 == True):
                    if(ind in (ind, new_ind)):
                        continue

                if(instances > 1):
                    flag_2 = True
                    
                    df_match_final, new_ind = do_the_matching(ind, df_event, df_instances, instances, df_first_match)
                    
                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match_final.to_csv(f, index=True, header=True)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match_final.to_csv(f, index=True, header=None)

                if(instances == 1):

                    df_match_final, new_ind = do_the_matching(ind, df_event, df_instances, instances, df_first_match)

                    """ Apply function to drop nan values and re-arange the dataframe.
                        In this way, matching is achieved
                    """
                    #df_match_final = df_match_final.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

                    if (flag_1 == True):                                                         
                        """ Create the csv file
                        """
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","w") as f:
                            df_match_final.to_csv(f, index=True, header=True)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        """ Write on the existing csv file
                        """
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","a") as f:
                            df_match_final.to_csv(f, index=True, header=None)

        elif (pulses_north <= pulses_south and pulses_north != 0):

            lst_min_delta, lst_min_pulse_south, lst_min_pulse_north = get_min_delta_startbins(df_event, pulses_south, pulses_north)
                
            """ Create dictionary to store the delta_startbins and the associated
                pulses corresponding to the minima. This dictionary will be used
                as input to a new dataframe exactly below
            """
            dict_first_match = {"min_delta_i": lst_min_delta, 
                                "pulse_number_south": lst_min_pulse_south,
                                "pulse_number_north": lst_min_pulse_north} 

            
            """ Dataframe containing the delta_startbins and the associated
                pulses corresponding to the minima. 
            """
            df_first_match = pd.DataFrame(data=dict_first_match)

            # flag to jump iterations from ind to new_ind
            flag_2 = False
           
            # check if same pulse is matched multiple times with other pulses
            # note: len(df_first_match['index_south']) = number of delta startbins =  npulses_north 
            for ind, pulse_number_north in enumerate(df_first_match['pulse_number_north']):

                """ Select instances of same pulse from df_first_match, that is, same pulse
                    matched many time with other pulses, and construct a new dataframe 
                    called df_instances
                """
                df_instances = df_first_match.query("pulse_number_north == '%s' " %str(pulse_number_north))
                instances = len(df_instances)
               
                # flag to jump iterations from ind to new_ind
                if(flag_2 == True):
                    if(ind in (ind, new_ind)):
                        continue

                if(instances > 1):
                    flag_2 = True
                    
                    df_match_final, new_ind = do_the_matching(ind, df_event, df_instances, instances, df_first_match)

                    if (flag_1 == True):                                                         
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","w") as f:
                            # index = False otherwise first column is a comma
                            df_match_final.to_csv(f, index=True, header=True)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","a") as f:
                            # index = False otherwise first column is a comma
                            df_match_final.to_csv(f, index=True, header=None)

                if(instances == 1):
                    df_match_final, new_ind = do_the_matching(ind, df_event, df_instances, instances, df_first_match)
                    
                    """ Apply function to drop nan values and re-arange the dataframe.
                        In this way, matching is achieved
                    """
                    #df_match_final = df_match_final.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

                    if (flag_1 == True):                                                         
                        """ Create the csv file
                        """
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","w") as f:
                            df_match_final.to_csv(f, index=True, header=True)
                            flag_1 = False
               
                    elif (flag_1 == False):
                        """ Write on the existing csv file
                        """
                        # export data to csv
                        with open(directory + prefix + proc + ".csv","a") as f:
                            df_match_final.to_csv(f, index=True, header=None)


#print("Finished matching")
end = time.time()
#print("Total run time: ", end-start)

