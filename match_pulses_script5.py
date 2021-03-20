
import sys
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/functions/')
#sys.path.append('/Users/georgesavvidis/Documents/PhD/data_analysis/lsm/classes/')

import os.path
import pandas 
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from matplotlib import path

'my functions'
from match_pulses_v6 import match_pulses_v6


'my classes'
from AbstractBaseClassCSV import CSVGetInfo

'print options'
np.set_printoptions(threshold = 20, edgeitems=10)

plot_path = '/Users/georgesavvidis/Documents/PhD/presentations/working_slides_pictures/populations/'

'''
Run: tj13s000
Gas: 135 mbar CH4
'''

'''Processing Information'''

'''
names correspond to the following channels = [ 'Channel 0','Channel 1','Channel 2']
Channel 0 = North
Channel 1 = South
Channel 2 = Laser
'''


directory = "/home/gsavvidis/notebooks/"
processing = np.array(["q41", "q42", "q43", "q44", "q45", "q46", "q47", "q48", "q49", "q50", 
					"q51", "q52", "q53", "q54", "q55", "q56", "q57", "q58"])
process = np.array(["q29"])

"""loop over all the processed runs and perform the matching of pulses"""
for idx, proc in enumerate(process):
	
    '''Read csv files'''
    data_dd_vectors = CSVGetInfo(directory, "DD_Vectors_"+ proc + ".csv")
    data_npulses = CSVGetInfo(directory, "DD_NPulses_" + proc + ".csv" )
    
    """transform it to numpy array"""
    dd_vectors = data_dd_vectors.make_array(data_dd_vectors)
    npulses = data_npulses.make_array(data_npulses)
    
    print('dd_vectors original shape ={}\n'.format(dd_vectors.shape))
    
    """directory to save matched pulses """
    dir_match = '/home/gsavvidis/notebooks/'

    """name of csv file with the matched pulses"""
    matched_pulses_filename = "matched_DD_Vectors_"+ proc + ".csv"
    complete_name = os.path.join(dir_match, matched_pulses_filename)

    """if the directory does not exist, make it"""
    if not os.path.isdir(directory):
        os.mkdir(directory)

    print("Starting matching of " + proc)
    match_pulses_v6(complete_name, dd_vectors, npulses)
    # match_pulses_v7(complete_name, dd_vectors, npulses)
    
    print("Finished matching")
