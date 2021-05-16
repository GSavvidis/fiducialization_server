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


def common_elements(a, b):
    a.sort_values()
    b.sort_values()
    i, j = 0, 0
    common = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            common.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return common


def drop_events_mi(df, cond_0, cond_1, unstacked=False):

    if unstacked == True:
        print("Ekanes malakia")

    elif unstacked == False:
        ch_0 = 0
        ch_1 = 1

        df_0 = df.xs(ch_0, level="subentry", drop_level=False)
        df_1 = df.xs(ch_1, level="subentry", drop_level=False)
        
        events_0 = df_0.query(f"{cond_0}").index.get_level_values(level="entry")
        events_1 = df_1.query(f"{cond_1}").index.get_level_values(level="entry")
        
        common = common_elements(events_0, events_1)
        
        df = df.drop(labels=common, axis=0)

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
            s_val_i = df_event.query(f"subentry == {ch_i}")["DD_VectorStartBinTime"].iloc[i]
            s_val_j = df_event.query(f"subentry == {ch_j}")["DD_VectorStartBinTime"].iloc[j]
            df_delta_start.iloc[i, j] = s_val_i - s_val_j

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




