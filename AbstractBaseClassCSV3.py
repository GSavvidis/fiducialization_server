from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import csv

"""Abstract class to read data from csv files"""

class AbstractBaseClassCSV3(ABC):

    def __init__(self, lst_files, lst_procnames):
        self._lst_files = lst_files
        self._lst_runnames = lst_procnames

        # self._filepath = filepath
        # self._filename = filename


    @property
    @abstractmethod
    def lst_files(self):
        pass
    
    @lst_files.setter
    @abstractmethod
    def lst_files(self, value):
        pass

    @property
    @abstractmethod
    def lst_procnames(self):
        pass

    @lst_procnames.setter
    @abstractmethod
    def lst_files(self, value):
        pass
    

    @abstractmethod
    def load_csv(self):
        pass
        
# Class to be used for MPA analysis
class GetCSVInfo(AbstractBaseClassCSV3):
    
    """ """
    @property
    def lst_files(self):
        return self._lst_files

    @lst_files.setter
    def lst_files(self, files):
        self._lst_files = files

    @property
    def lst_procnames(self):
        return self._lst_runnames

    @lst_procnames.setter
    def lst_procnames(self, list_runnames):
        self._lst_runnames = list_runnames

    def load_csv(self, nrows=None):		

        # df = pd.DataFrame()
        # DF_list = list(self._lst_runnames)
        DF_dict = {}
        
        for indx, file in enumerate(self._lst_files):		
            """find number of rows and columns in the ntuple file"""
            with open(file, "r") as current_file:
                if "DD_Vectors" in file:
                    runname = self._lst_runnames[indx]
                    print(f"{runname}")
                    DF_dict[runname] = pd.read_csv(current_file,
                                                   header = 0)

                elif "NPulses" in file:
                    runname = self._lst_runnames[indx]
                    DF_dict[runname] = pd.read_csv(current_file,
                                                   header = 0)

    
        return DF_dict
    
    def display_shape(self):
        print("The shape of the data array is {}".format(len(self._data)))


# Class to be used for DD2 analysis
class GetCSVInfoDD2(AbstractBaseClassCSV3):
    
    """ """
    @property
    def lst_files(self):
        return self._lst_files

    @lst_files.setter
    def lst_files(self, files):
        self._lst_files = files

    @property
    def lst_procnames(self):
        return self._lst_runnames

    @lst_procnames.setter
    def lst_procnames(self, list_runnames):
        self._lst_runnames = list_runnames

    def load_csv(self, nrows):		

        DF_dict = {}
        
        for indx, file in enumerate(self._lst_files):			
            # read csv file
            with open(file, "r") as current_file:
                runname = self._lst_runnames[indx]
                print(f"{runname}")
                DF_dict[runname] = pd.read_csv(current_file,
                                               header = 0, 
                                               nrows=nrows,
                                               index_col=[0,1]) # throws warning

                #DF_dict[runname] = DF_dict[runname].set_index([0,1], inplace=False)

    
        return DF_dict
    
    def display_shape(self):
        print("The shape of the data array is {}".format(len(self._data)))

    


# Class to be used for MPA analysis
class GetCSVInfoMPA(AbstractBaseClassCSV3):
    
    """ """
    @property
    def lst_files(self):
        return self._lst_files

    @lst_files.setter
    def lst_files(self, files):
        self._lst_files = files

    @property
    def lst_procnames(self):
        return self._lst_runnames

    @lst_procnames.setter
    def lst_procnames(self, list_runnames):
        self._lst_runnames = list_runnames

    def load_csv(self, nrows):		

        DF_dict = {}
        
        for indx, file in enumerate(self._lst_files):			
            # read csv file
            with open(file, "r") as current_file:
                runname = self._lst_runnames[indx]
                print(f"{runname}")
                DF_dict[runname] = pd.read_csv(current_file,
                                               header = 0, 
                                               nrows=nrows,
                                               index_col=[0,1,2,13]) # throws warning

                #DF_dict[runname] = DF_dict[runname].set_index([0,1], inplace=False)

    
        return DF_dict
    
    def display_shape(self):
        print("The shape of the data array is {}".format(len(self._data)))
