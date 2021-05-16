import pandas as pd
import numpy as np



def read_npulses(directory, proc):
    
    lst_names = ["north", "south", "laser"]
    df_npulses = pd.read_csv(directory + "DD_NPulses_" + proc + ".csv", names = lst_names)

    return df_npulses




def drop_laser_events(df):
    # select laser 
    df_laser = df[df.index.isin([2], level="subentry")]
    
    # select entry(event) indexes corresponding to laser events
    idx_laser = df_laser.index.get_level_values("entry")
    df = df.drop(labels=idx_laser, axis=0)

    return df



def drop_events_by_index(df, level_name, value_to_drop):
    
    # select dataframe to drop
    df_to_drop = df[df.index.isin([value_to_drop], level=f"{level_name}")]
    
    # select entry(event) indexes to drop
    idx_to_drop = df_to_drop.index.get_level_values("entry")
    df = df.drop(labels=idx_to_drop, axis=0)
    
    return df



def drop_events(df, cond, unstacked):

    if unstacked == True:

        """ Returns masked series with True 
            values for elements satisfying the
            given condition
        """
        result = pd.eval(cond)

        """ Get indices where the values are True
        """
        inds = result.index[result]

        """ Drop indexes where values are True
        """
        df = df.drop(labels=inds, axis=0)

    elif unstacked == False:
        print("Dataframe is not unstacked. Cannot drop events")

    return df


def get_min_delta_startbins(df_event, npulses_south, npulses_north):

    # This condiction must always be satisfaided:npulses_i < npulses_j
    if npulses_south < npulses_north:
        npulses_i = npulses_south
        npulses_j = npulses_north
        ch_i = "1"
        ch_j = "0"
    
    elif npulses_north <= npulses_south:
        npulses_i = npulses_north
        npulses_j = npulses_south
        ch_i = "0"
        ch_j = "1"

    # list should be equal to npulses_j*npulses_i delta_startbins
    lst_delta = [[None]*int(npulses_j)]*int(npulses_i)
    
    # number of columns = number of j pulses
    n_columns = np.arange(npulses_j)
    
    # create DataFrame to store delta-startbin
    df_delta_start = pd.DataFrame(data=lst_delta, columns=n_columns)

    # list with size equal to the number of 
    lst_min_delta = [None]*int(npulses_i)

    #
    lst_min_pulse_i = [None]*int(npulses_i)
    lst_min_pulse_j = [None]*int(npulses_i)

    for i in range(npulses_i):
        for j in range(npulses_j):

            # units in microseconds
            val_i = df_event.query(f"subentry == {ch_i}")["DD_VectorStartBinTime"].iloc[i]
            val_j = df_event.query(f"subentry == {ch_j}")["DD_VectorStartBinTime"].iloc[j]
            df_delta_start.iloc[i, j] = val_i - val_j

        # List that contains the minimum delta_startbin
        s_delta_startbin = df_delta_start.iloc[i]
        lst_min_delta[i] = s_delta_startbin.min()
       
        # Pulse numbers of j-th pulses
        s_delta_startbin = pd.to_numeric(s_delta_startbin)
        lst_min_pulse_j[i] = s_delta_startbin.idxmin()
        
        # Pulse numbers of i-th pulses
        lst_min_pulse_i[i] = i
    
    if npulses_south < npulses_north:
        lst_min_pulse_south = lst_min_pulse_i
        lst_min_pulse_north = lst_min_pulse_j
    
    elif npulses_north <= npulses_south:
        lst_min_pulse_south = lst_min_pulse_j
        lst_min_pulse_north = lst_min_pulse_i

    return lst_min_delta, lst_min_pulse_south, lst_min_pulse_north




def do_the_matching(ind, df_event, df_instances, instances, df_first_match):

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
    kept_pulse_north = int(kept_pulse_north.values)
    kept_pulse_south = int(kept_pulse_south.values)
    #kept_pulse_north = int(kept_pulse_north.to_string(index=False))
    #kept_pulse_south = int(kept_pulse_south.to_string(index=False))

    # separate channels into dataframes
    df_north = df_event.xs(0, level="subentry", drop_level=False, axis=0)
    df_south = df_event.xs(1, level="subentry", drop_level=False, axis=0)
    #print("df_south")
    #print(df_south)
    #print("df_north")
    #print(df_north)
    
    df_north = df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False)
    df_south = df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False)

    df_match = pd.concat([df_north, df_south], axis=0)
    
    """ Apply function to drop nan values and re-arange the dataframe.
        In this way, matching is achieved
    """
    #df_match = df_match.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

    # number of pulses remaining to be checked
    if instances == 1:
        new_ind = 0

    elif instances > 1:
        remaining = len(df_first_match['pulse_number_south']) - instances
        new_ind = ind + remaining

    return df_match, new_ind


def select_events_by_query(df, condition):
    df = df.query(condition)

    return df



def select_by_index_level(df, ch_i, level):
    df = df[df.index.isin([ch_i], level=f"{level}")]
    
    return df

def select_laser(df):
    # select laser channel
    df_laser = df[df.index.isin([2], level="subentry")]


def count_pulses(df, event, channel):
    df_pulses = df.query(f"entry == {event} & subentry == {channel}")
    lst_pulses = df_pulses.index.get_level_values("DD_NPulses")

    if (len(lst_pulses) == 0):
        pulses = 0

    elif (len(lst_pulses) != 0):
        #pulses = int(df_pulses.index.get_level_values("DD_NPulses")[0])
        pulses = len(lst_pulses)

    return pulses


def add_new_column(df, col_name, new_col):
    df[f"{col_name}"] = new_col

    return df





#def match_pulses(df, ch_i, ch_j, pulses_ch_i, pulses_ch_j):
        #elif (pulses_south < pulses_north and pulses_south != 0):

        #    # size of list should be equal to npulses_north*npulses_south
        #    # and correspond to all possible delta-startbin
        #    lst_delta = []
        #    lst_delta_laser = [[None]*int(pulses_laser)]*int(pulses_south)
        #    
        #    # numbers to be used as columns in the dataframe
        #    # below. 
        #    n_columns_laser = np.arange(pulses_laser)
        #    
        #    # create DataFrame to store delta-startbin
        #    df_delta_start_laser = pd.DataFrame(data=lst_delta_laser, columns=n_columns_laser)

        #    # list with size equal to the number of 
        #    lst_min_delta_laser = [None]*int(pulses_south)

        #    #
        #    lst_min_delta_pulse_number_south = [None]*int(pulses_south)
        #    lst_min_delta_pulse_number_laser = [None]*int(pulses_south)

        #    """ use f-string to reference local variable
        #    option to reference column labels
        #    """
        #    col_n = "event"

        #    # select pulses of same event
        #    #df_event = df_vecs.query(f"{col_n} == {event}")
        #    print(f"Getting the cross-section for event {ev}")
        #    df_event = df_vecs.xs(ev, level="entry", drop_level=False)

        #    # drop columns that have all their values nan
        #    print("Dropping NaN values from df_event")
        #    df_event = df_event.dropna(axis=1, how="all")
        #    print("NaN values from df_event dropped")
        #    print(df_event)
        #    print("")

        #    # get column at DD_NPulses level for North(0) and South(1) channel
        #    col_1 = df_event.columns.get_level_values("DD_NPulses")[1]
        #    print("col_1")
        #    print(col_1)
        #    print("")

        #    # number of NPulses of laser channel
        #    col_2 = df_event.columns.get_level_values("DD_NPulses")[2]
        #    print("col_2")
        #    print(col_2)
        #    print("")

        #    # correct for the time offset between north/south channel
        #    # units in microseconds

        #    df_event["DD_VectorStartBinTime", 1, col_1] =  df_event["DD_VectorStartBin", 1, col_1]\
        #                                                 + df_event["TimeS", 1, col_1].mul(10e-6)\
        #                                                 + df_event["TimeMuS", 1, col_1]
        #    
        #    df_event["DD_VectorStartBinTime", 2, col_2] =  df_event["DD_VectorStartBin", 2, col_2]\
        #                                                 + df_event["TimeS", 2, col_2].mul(10e-6)\
        #                                                 + df_event["TimeMuS", 2, col_2]
        #    # Find elements which are not NaN values
        #    mask_npulses_1 = df_event["DD_VectorStartBinTime", 1, col_1].notnull()
        #    mask_npulses_2 = df_event["DD_VectorStartBinTime", 2, col_2].notnull()
        #    
        #    # drop NaN(False) values = select rows corresponding to pulses
        #    ser_npulses_1 = df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1]
        #    ser_npulses_2 = df_event["DD_VectorStartBinTime", 2, col_2][mask_npulses_2]

        #    # find number of rows for current event
        #    nrows = len(df_event.index) 
        #    print(nrows)

        #    # find number of olumns for current event
        #    ncolumns = len(df_event.columns)
        #    print(ncolumns)
        #    
        #    for i in range(pulses_south):
        #        # vector_indexes = npulses_north = number of
        #        # pulses in the north
        #        #for j, ind_north in enumerate(v_indexes_north):
        #        
        #        for k in range(pulses_laser):

        #            # units in microseconds
        #            df_delta_start_laser.iloc[i, k] = df_event["DD_VectorStartBinTime", 1, col_1][mask_npulses_1].iloc[i]\
        #                                            - df_event["DD_VectorStartBinTime", 2, col_2][mask_npulses_2].iloc[k]
        #        

        #        
        #        """ Convert class 'pandas.indexes.numeric.Int64Index' to numpy
        #            in order to add it as column to df_first_match dataframe below
        #        """ 
        #        lst_min_delta_laser[i] = abs(df_delta_start_laser.min().values.item())
        #        df_delta_start_laser_numeric_cells = pd.to_numeric(abs(df_delta_start_laser.iloc[i]))

        #        """ The index of df_delta_start_laser coincides with the pulse 
        #            number in the South
        #        """
        #        lst_min_delta_pulse_number_laser[i] = abs(df_delta_start_laser_numeric_cells.idxmin())

        #        """ Store index of south pulse which gives the minimum delta_startbin
        #        """
        #        # List that contains south pulses
        #        lst_min_delta_pulse_number_south[i] = i
        #        
        #    print(lst_min_delta_pulse_number_laser)
        #    print(lst_min_delta_pulse_number_south)
        #    print(lst_min_delta)

        #    """ Create dictionary to store the delta_startbins and the associated
        #        pulses corresponding to the minima. This dictionary will be used
        #        as input to a new dataframe exactly below
        #    """
        #    if (flag_laser == True):
        #        dict_first_match = {"min_delta_i": lst_min_delta, 
        #                            "min_delta_laser_i": lst_min_delta_laser,
        #                            "pulse_number_south": lst_min_delta_pulse_number_south,
        #                            "pulse_number_north": lst_min_delta_pulse_number_north,
        #                            "pulse_number_laser": lst_min_delta_pulse_number_laser}
        #    
        #    elif (flag_laser == False):
        #        dict_first_match = {"min_delta_i": lst_min_delta, 
        #                            "min_delta_laser_i": lst_min_delta_laser,
        #                            "pulse_number_south": lst_min_delta_pulse_number_south,
        #                            "pulse_number_north": lst_min_delta_pulse_number_north}

        #    print(dict_first_match)
        #    print("")
        #    stop
        #    
        #    """ Dataframe containing the delta_startbins and the associated
        #        pulses corresponding to the minima. 
        #    """
        #    df_first_match = pd.DataFrame(data=dict_first_match)
        #    print("Best match")
        #    print(df_first_match)
        #    print("")

        #    # flag to jump iterations from ind to new_ind
        #    flag_2 = False
        #   
        #    # check if same pulse is matched multiple times with other pulses
        #    # note: len(df_first_match['index_south']) = number of delta startbins =  npulses_north 
        #    for ind, pulse_number_north in enumerate(df_first_match['pulse_number_north']):

        #        print(f"ind={ind}")
        #        """ Select instances of same pulse from df_first_match, that is, same pulse
        #            matched many time with other pulses, and construct a new dataframe 
        #            called df_instances
        #        """
        #        df_instances = df_first_match.query("pulse_number_north == '%s' " %str(pulse_number_north))
        #        instances = len(df_instances)
        #        print("Dataframe df_instances")
        #        print(df_instances)
        #        print("")
        #        print("Instances")
        #        print(instances)
        #        print("")
        #        print("df_first_match['pulse_number_north']")
        #        print(df_first_match['pulse_number_north'])
        #       
        #        # flag to jump iterations from ind to new_ind
        #        #if(flag_2 == True):
        #        #    if(ind in (ind, new_ind)):
        #        #        continue

        #        #if(instances > 1):
        #        #    flag_2 = True
        #        #    
        #        #    how_many = instances
        #        #    print(how_many)

        #        #    """ In the dataframe df_instances find the minimum delta
        #        #        startbin value and the corresponding index 
        #        #    """
        #        #    kept_min_delta = np.min(df_instances["min_delta_i"])
        #        #    
        #        #    """ The index coincides with the south pulse number. That means that
        #        #        kept_min_delta_idx = pulse number south that gives the minimum
        #        #        delta_startbin with the north. 
        #        #    """
        #        #    kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
        #        #    

        #        #    """ Return the north and south pulse associated with the minimum delta 
        #        #        startbin from df_instances
        #        #    """
        #        #    kept_pulse_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_north"]
        #        #    kept_pulse_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_south"]

        #        #    # keep only the values of the dataframe without the indexes
        #        #    kept_pulse_north = int(kept_pulse_north.to_string(index=False))
        #        #    kept_pulse_south = int(kept_pulse_south.to_string(index=False))
        #        #    print("kept_pulse_north")
        #        #    print(kept_pulse_north)
        #        #    print("")
        #        #    print("kept_pulse_south")
        #        #    print(kept_pulse_south)
        #        #    print("")

        #        #    # separate channels into dataframes
        #        #    df_north = df_event.xs(0, level="subentry", drop_level=False, axis=1)
        #        #    df_south = df_event.xs(1, level="subentry", drop_level=False, axis=1)
        #        #    df_laser = df_event.xs(2, level="subentry", drop_level=False, axis=1)
        #        #    
        #        #    #df_south = df_south.dropna(axis=0, how="all")
        #        #    #df_laser = df_laser.dropna(axis=0, how="all")

        #        #    #df_north = df_north.unstack(level="entry")
        #        #    #df_south = df_south.unstack(level="entry")
        #        #    #df_laser = df_laser.unstack(level="entry")
        #        #    #pretty_print(df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False))
        #        #    #pretty_print(df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False))
        #        #    #print(df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False))
        #        #    #print(df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False))
        #        #    
        #        #    df_north = df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False)
        #        #    df_south = df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False)

        #        #    if (pulses_laser == 1):
        #        #        kept_min_delta_laser = np.min(df_instances["min_delta_laser_i"]) 
        #        #        
        #        #        """ In a similar manner, the south pulse
        #        #            that should match with the laser pulse is retrieved
        #        #        """
        #        #        kept_min_delta_laser_idx = df_instances["min_delta_laser_i"].idxmin()
        #        #        
        #        #        """ Select the laser pulse number associated with
        #        #            the min_delta_laser
        #        #        """
        #        #        kept_pulse_number_laser = df_instances.query("min_delta_laser_i == '%s'" %str(kept_min_delta_laser))["pulse_number_laser"]

        #        #        # drop index
        #        #        kept_pulse_number_laser = int(kept_pulse_number_laser.to_string(index=False))
        #        #        # construct laser dataframe
        #        #        df_laser = df_laser.xs(kept_pulse_number_laser, level="subsubentry", axis=0, drop_level=False)
        #        #    
        #        #    elif (pulses_laser == 0):
        #        #        kept_min_delta_laser = 0.0 
        #        #        
        #        #        """ In a similar manner, the south pulse
        #        #            that should match with the laser pulse is retrieved
        #        #        """
        #        #        kept_min_delta_laser_idx = 0.0
        #        #        kept_pulse_number_laser = df_instances.query("min_delta_laser_i == '%s'" %str(kept_min_delta))["pulse_number_laser"]
        #        #        kept_pulse_number_laser = int(kept_pulse_number_laser.to_string(index=False))
        #        #        df_laser = df_laser.xs(kept_pulse_number_laser, level="subsubentry", axis=0, drop_level=False)
        #        #    
        #        #    print("kept_min_delta")
        #        #    print(kept_min_delta)
        #        #    print("")
        #        #    print("kept_min_delta_idx")
        #        #    print(kept_min_delta_idx)
        #        #    print("")
        #        #    print("kept_min_delta_laser")
        #        #    print(kept_min_delta_laser)
        #        #    print("")
        #        #    print("kept_min_delta_laser_idx")
        #        #    print(kept_min_delta_laser_idx)
        #        #    print("")
        #        #    stop

        #        #    #df_north = df_north.dropna(axis=1, how="all")
        #        #    #df_south = df_south.dropna(axis=1, how="all")

        #        #    #df_north = df_north.reset_index(level="subsubentry")
        #        #    #df_south = df_south.reset_index(level="subsubentry")
        #        #    #pretty_print(df_north)
        #        #    #pretty_print(df_south)
        #        #    #pretty_print(df_laser)
        #        #    #stop
        #        #    #print(df_north)
        #        #    #print(df_south)

        #        #    df_match = pd.concat([df_north, df_south], axis=0)
        #        #    df_match_laser = pd.concat([df_north, df_south], axis=0)
        #        #    
        #        #    """ Apply function to drop nan values and re-arange the dataframe.
        #        #        In this way, matching is achieved
        #        #    """
        #        #    df_match = df_match.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

        #        #    #stop
        #        #    #pretty_print(df_match)
        #        #    #print(df_match)

        #        #    #pretty_print(df_event)
        #        #    #pretty_print(df_event.xs([35,kept_pulse_north], level=["entry", "subsubentry"], drop_level=False, axis=0))
        #        #    #pretty_print(df_event.xs(0, level="subentry", drop_level=False, axis=1))
        #        #    #print(df_event.xs([35,kept_pulse_north], level=["entry", "subsubentry"], drop_level=False).columns[3])
        #        #    #df_1 = df_vecs.loc[kept_pulse_north:kept_pulse_north, :"timeMuS_north"]
        #        #    #df_2 = df_vecs.loc[kept_pulse_south:kept_pulse_south, "number_south":]
        #        #    #df_match = pd.concat([df_1, df_2], axis=1)
        #        #    
        #        #    # number of pulses remaining to be checked
        #        #    remaining = len(df_first_match['index_north']) - instances
        #        #    new_ind = ind + remaining
        #        #    if(i == 2): 
        #        #        stop
        #        #    
        #        #    if (flag_1 == True):                                                         
        #        #        # export data to csv
        #        #        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
        #        #            # index = False otherwise first column is a comma
        #        #            df_match.to_csv(f, index=False, header=False)
        #        #            flag_1 = False
        #       
        #        #    elif (flag_1 == False):
        #        #        # export data to csv
        #        #        with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
        #        #            # index = False otherwise first column is a comma
        #        #            df_match.to_csv(f, index=False, header=None)

        #        if(instances == 1):

        #            """ In the dataframe df_instances find the minimum delta
        #                startbin value and the corresponding index 
        #            """
        #            kept_min_delta = np.min(df_instances["min_delta_i"])
        #            
        #            """ The index coincides with the south pulse number. That means that
        #                kept_min_delta_idx = pulse number south that gives the minimum
        #                delta_startbin with the north. 
        #            """
        #            kept_min_delta_idx = df_instances["min_delta_i"].idxmin()
        #            
        #            """ Return the north and south pulse associated with the minimum delta 
        #                startbin from df_instances
        #            """
        #            kept_pulse_north = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_north"]
        #            kept_pulse_south = df_instances.query("min_delta_i == '%s'" %str(kept_min_delta))["pulse_number_south"]

        #            # keep only the values of the dataframe without the indexes
        #            kept_pulse_north = int(kept_pulse_north.to_string(index=False))
        #            kept_pulse_south = int(kept_pulse_south.to_string(index=False))

        #            # separate channels into dataframes
        #            df_north = df_event.xs(0, level="subentry", drop_level=False, axis=1)
        #            df_south = df_event.xs(1, level="subentry", drop_level=False, axis=1)

        #            df_north = df_north.xs(kept_pulse_north, level="subsubentry", axis=0, drop_level=False)
        #            df_south = df_south.xs(kept_pulse_south, level="subsubentry", axis=0, drop_level=False)

        #            if (pulses_laser == 1):

        #                df_laser = df_event.xs(2, level="subentry", drop_level=False, axis=1)
        #                kept_min_delta_laser = np.min(df_instances["min_delta_laser_i"]) 
        #                
        #                """ In a similar manner, the south pulse
        #                    that should match with the laser pulse is retrieved
        #                """
        #                kept_min_delta_laser_idx = df_instances["min_delta_laser_i"].idxmin()
        #                
        #                """ Select the laser pulse number associated with
        #                    the min_delta_laser
        #                """
        #                kept_pulse_number_laser = df_instances.query("min_delta_laser_i == '%s'" %str(kept_min_delta_laser))["pulse_number_laser"]
        #                #print(kept_pulse_number_laser)

        #                # drop index
        #                kept_pulse_number_laser = int(kept_pulse_number_laser.to_string(index=False))

        #                # construct laser dataframe
        #                df_laser = df_laser.xs(kept_pulse_number_laser, level="subsubentry", axis=0, drop_level=False)
        #            
        #            elif (pulses_laser == 0):
        #                kept_min_delta_laser = 0.0 
        #                
        #                """ In a similar manner, the south pulse
        #                    that should match with the laser pulse is retrieved
        #                """
        #                kept_min_delta_laser_idx = 0.0
        #                kept_pulse_number_laser = df_instances.query("min_delta_laser_i == '%s'" %str(kept_min_delta_laser))["pulse_number_laser"]
        #                kept_pulse_number_laser = float(kept_pulse_number_laser.to_string(index=False))
        #                print(kept_pulse_number_laser)

        #                """ Constructing the laser dataframe. The size of the frame
        #                    will be identical to the size of df_south. All column values
        #                    will be replaced be zeroes. DD_NPulses level will be set to 0.0,
        #                    subentry level will be set to 2
        #                """
        #                df_laser = df_south.xs(0, level="subsubentry", axis=0, drop_level=False)
        #                df_laser = df_laser.stack(level=["DD_NPulses", "subentry"])
        #                print("df_laser")
        #                print(df_laser.index)
        #                print("")
        #                as_list = df_laser.index.tolist()
        #                #print(as_list[0])
        #                #stop
        #                #as_list = df_laser.index.values[0][0]
        #                print(as_list)
        #                #for i, val in enumerate(as_list[0]):

        #                idx = as_list[0].index(3.0)
        #                print(idx)
        #                as_list[0][idx] = 0
        #                df_laser.index = as_list
        #                print(df_laser.index)
        #                stop
        #                df_laser.index = df_laser.index.set_levels(df_laser.index.levels[0])
        #                stop
        #                as_list[idx] = 0
        #                df_laser.index = as_list
        #                #df_laser = df_laser.reindex(0, level=["DD_NPulses"])
        #                print("df_laser")
        #                print(df_laser)
        #                print("")
        #                stop
        #                #print(df_laser.columns.levels[0])
        #                #laser_subentry = df_laser.columns.get_level_values("subentry")
        #                #print(laser_subentry)
        #                #laser_subentry = 2

        #                """ Returns a list of size equal with the number of columns in df_south and with values
        #                    equal to DD_NPulses of South channel. 
        #                """
        #                npulses_south = df_south.xs(0, level="subsubentry", axis=0, drop_level=False).columns.get_level_values("DD_NPulses")[0] # [0] to keep only the first element of the list
        #                npulses_laser = 0.0

        #                """ Dict to replace subentry 1(South channel) with subentry 2(Laser)
        #                """
        #                dict_replace_subentry = {1:2}
        #                dict_replace_npulses = {npulses_south:npulses_laser}
        #                print("npulses_south")
        #                print(npulses_south)
        #                print("")

        #                """ List of size equal to the number of columns in df_laser and which
        #                    contains the new value of subentry, that is 2
        #                """
        #                new_vals_subentry = [dict_replace_subentry[x] for x in df_laser.columns.get_level_values("subentry")]
        #                new_vals_npulses = [dict_replace_npulses[x] for x in df_laser.columns.get_level_values("DD_NPulses")]
        #                col_len = len(df_event.columns.levels[0])
        #                new_vals_subentry = [None]*col_len
        #                new_vals_npulses = [None]*col_len
        #                for i in range(col_len):
        #                    new_vals_subentry[i] = i+1
        #                    new_vals_npulses[i] = 0.0
        #                
        #                print("new_vals_subentry")
        #                print(new_vals_subentry)
        #                print("")
        #                """ At the column level subentry, replace all old values(1=South channel) with 
        #                    new values(2=Laser channel)
        #                """
        #                #df_laser.columns.set_levels(df_laser.columns.levels["subentry"].replace(1,2), level="subentry",verify_integrity=True, inplace=True)
        #                #df_laser.columns.set_levels(new_vals_subentry, level="subentry",verify_integrity=False, inplace=True)
        #                #df_laser = df_laser.columns.reindex(new_vals_subentry, level="subentry")
        #                df_laser.columns = df_laser.columns.set_levels(new_vals_subentry, level="subentry",verify_integrity=True, inplace=False)
        #                df_laser.columns = df_laser.columns.set_levels(new_vals_npulses, level="DD_NPulses",verify_integrity=False, inplace=False)
        #                print("df_laser")
        #                print(df_laser)
        #                
        #                #df_laser.columns = df_laser.columns.set_levels(new_vals_subentry, level="subentry",verify_integrity=False, inplace=False)
        #                #df_laser.columns = df_laser.columns.set_levels(new_vals_npulses, level="DD_NPulses",verify_integrity=False, inplace=False)
        #                print("df_south")
        #                print(df_south)
        #                print("")
        #                print("df_laser")
        #                print(df_laser)
        #                print("")

        #                #print(df_laser.columns.get_level_values("subentry")[0])
        #                #stop
        #                #print(df_laser.columns.values[0][1])
        #                #stop
        #                col_names = df_laser.columns.levels[0]
        #                #print(col_names)
        #                for i, name in enumerate(col_names):
        #                    #df_laser.columns.values[i][1] = 2
        #                    #df_laser.columns.values[i][2] = 0.0
        #                    df_laser[f"{name}"].values[0][0] = 0.0

        #                print(df_laser)

        #            
        #            print("kept_min_delta")
        #            print(kept_min_delta)
        #            print("")
        #            print("kept_min_delta_idx")
        #            print(kept_min_delta_idx)
        #            print("")
        #            print("kept_min_delta_laser")
        #            print(kept_min_delta_laser)
        #            print("")
        #            print("kept_min_delta_laser_idx")
        #            print(kept_min_delta_laser_idx)
        #            print("")

        #            #print("df_south")
        #            #print(df_south)
        #            #print("")
        #            #print("df_laser")
        #            #print(df_laser)
        #            #print("")
        #            df_match_laser = pd.concat([df_south, df_laser], axis=0)
        #            #print("df_match_laser")
        #            #print(df_match_laser)
        #            #print("")

        #            """ Apply function to drop nan values and re-arange the dataframe.
        #                In this way, matching is achieved
        #            """
        #            df_match_laser = df_match_laser.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')
        #            #print("df_match_laser")
        #            #print(df_match_laser)
        #            #print("")
        #            
        #            #df_match_test = pd.concat([df_north, df_south], axis=0)
        #            #df_match_test = df_match_test.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')
        #            
        #            df_match_final = pd.concat([df_match_laser, df_north], axis=0)
        #            """ Apply function to drop nan values and re-arange the dataframe.
        #                In this way, matching is achieved
        #            """
        #            df_match_final = df_match_final.apply(lambda x: pd.Series(x.dropna().values)).dropna(axis='columns')

        #            #print("df_match_final")
        #            #print(df_match_final)
        #            #print("")
        #            #df_match_final = df_match_final.stack(level=["DD_NPulses", "subentry"])
        #            #print("Stacked df_match_final")
        #            #print(df_match_final)
        #            #print("")

        #            print(f"i={i}")
        #            if(ind == 2): 
        #                stop
        #            
        #            if (flag_1 == True):                                                         
        #                """ Create the csv file
        #                """
        #                # export data to csv
        #                with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","w") as f:
        #                    # index = False otherwise first column is a comma
        #                    df_match_final.to_csv(f, index=False, header=True)
        #                    flag_1 = False
        #       
        #            elif (flag_1 == False):
        #                """ Write on the existing csv file
        #                """
        #                # export data to csv
        #                with open(directory + "matched_tj13s000_MPA_" + proc + ".csv","a") as f:
        #                    # index = False otherwise first column is a comma
        #                    df_match_final.to_csv(f, index=False, header=None)
