from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import csv

"""Abstract class to read data from csv files"""

class AbstractBaseClassCSV2(ABC):

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
class GetCSVInfo(AbstractBaseClassCSV2):
	
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

	def load_csv(self):		

		# df = pd.DataFrame()
		# DF_list = list(self._lst_runnames)
		DF_dict = {}
		
		for indx, file in enumerate(self._lst_files):		
			"""find number of rows and columns in the ntuple file"""
			with open(file, "r") as current_file:

				if "DD_Vectors" in file:
					names = [
							"event",	             # 0
							"number_north",          # 1
							"startbin_north",        # 2
							"stopbin_north",         # 3
							"rawampl_north",         # 4
							"rawrise_north",         # 5
							"rawwidth_north",        # 6
							"amplADU_north",         # 7
							"risetime_north",            # 8
							"DD_Rise25pct_North",    # 9
							"DD_Rise75pct_North",    # 10
							"timeS_north",           # 11
							"timeMuS_north",         # 12
							"number_south",          # 13
							"startbin_south",        # 14
							"stopbin_south",         # 15
							"rawampl_south",         # 16
							"rawrise_south",         # 17
							"rawwidth_south",        # 18
							"amplADU_south",         # 19
							"risetime_south",            # 20
							"DD_Rise25pct_South",    # 21
							"DD_Rise75pct_South",    # 22
							"timeS_south",           # 23
							"timeMuS_south",         # 24
							"number_laser",          # 25
							"startbin_laser",        # 26
							"stopbin_laser",         # 27
							"rawampl_laser",         # 28
							"rawrise_laser",         # 29
							"rawwidth_laser",        # 30
							"amplADU_laser",         # 31
							"risetime_laser",            # 32
							"DD_RawRise50pct_Laser", # 33
							"TimeS_Laser",           # 34
							"TimeMuS_Laser",         # 35
							"db_prev_north",         # 36
							"db_next_north",         # 37
							"db_prev_south",         # 38
							"db_next_south",         # 39
							"db_prev_laser",         # 40
							"db_next_laser"	         # 41
							]

					cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12,
							13, 14, 15,16, 17, 18, 19, 20, 23,
							24, 28, 36, 37, 38, 39]
					runname = self._lst_runnames[indx]
					print(f"{runname}")
					DF_dict[runname] = pd.read_csv(current_file,
												   names = names,
												   header = None)

				elif "NPulses" in file:
					names = ["npulses_north",
							 "npulses_south", 
							 "npulses_laser"]
					runname = self._lst_runnames[indx]
					DF_dict[runname] = pd.read_csv(current_file,
												   names = names,
												   header = None,
												   skiprows=None)

	
		return DF_dict
	
	def display_shape(self):
		print("The shape of the data array is {}".format(len(self._data)))


# Class to be used for DD2 analysis
class GetCSVInfoDD2(AbstractBaseClassCSV2):
	
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

	def load_csv(self):		

		# df = pd.DataFrame()
		# DF_list = list(self._lst_runnames)
		DF_dict = {}
		
		for indx, file in enumerate(self._lst_files):		
			"""find number of rows and columns in the csv file"""
			with open(file, "r") as current_file:
				names = ["event",                  # 0
						 "rawampl_north",          # 1
						 "rawampl_south",          # 2
						 "rawampl_laser",          # 3
						 "rawrise_north",          # 4
						 "rawrise_south",          # 5
						 "rawrise_laser",          # 6
						 "rawwidth_north",         # 7
						 "rawwidth_south",         # 8
						 "rawwidth_laser",         # 9
						 "ampl_north",             # 10
						 "ampl_south",             # 11
						 "ampl_laser",             # 12
						 "amplADU_north",          # 13
						 "amplADU_south",          # 14
						 "amplADU_laser",          # 15
						 "risetime_north",         # 16
						 "risetime_south",         # 17
						 "risetime_laser",         # 18
						 "positiveintegral_north", # 19
						 "positiveintegral_south", # 20
						 "positiveintegral_laser", # 21
						 "negativeintegral_north", # 22
						 "negativeintegral_south", # 23
						 "negativeintegral_laser", # 24
						 "time_north",             # 25
						 "time_south",             # 26
						 "time_laser",             # 27
						 "timemus_north",          # 28
						 "timemus_south",          # 29
						 "timemus_laser",          # 30
						 "deltatprevs_north",      # 31
						 "deltatprevs_south",      # 32
						 "deltatprevs_laser",      # 33
						 "rawmaxsample_north",     # 34
						 "rawmaxsample_south",     # 35
						 "rawmaxsample_laser",     # 36
						 "decmaximum_north",       # 37
						 "decmaximum_south",       # 38
						 "decmaximum_laser",       # 39
						 "maximum_north",          # 40
						 "maximum_south",          # 41
						 "maximum_laser",          # 42
						 "DD_Rise25pct_north",     # 43
						 "DD_Rise25pct_south",     # 44
						 "DD_Rise25pct_laser",     # 45
						 "DD_Rise75pct_north",     # 46
						 "DD_Rise75pct_south",     # 47
						 "DD_Rise75pct_laser",     # 48
						 "DD_RawRise50pct_north",  # 49
						 "DD_RawRise50pct_south",  # 50
						 "DD_RawRise50pct_laser",  # 51
						 "TimeS_north",            # 52
						 "TimeS_south",            # 53
						 "TimeS_laser",            # 54
						 "TimeMuS_north",          # 55
						 "TimeMuS_south",          # 56
						 "TimeMuS_laser"]          # 57

				cols = [0 ,1 ,2, 3, 4, 5, 6, 7, 8, 9, 13, 14, 
						15, 16, 17, 18, 19, 20, 22, 23, 25, 26, 
						27, 28, 29, 30, 31, 32, 33, 52, 53, 54, 
						55, 56, 57]
				runname = self._lst_runnames[indx]
				print(f"{runname}")
				DF_dict[runname] = pd.read_csv(current_file,
												   names = names,
												   header = None,
												   usecols = cols)
	
		return DF_dict
	
	def display_shape(self):
		print("The shape of the data array is {}".format(len(self._data)))

	


