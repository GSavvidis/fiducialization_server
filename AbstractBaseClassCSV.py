from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

"""Abstract class to read data from csv files"""

class AbstractBaseClassCSV(ABC):

	def __init__(self, path, filename):
		self._path = path
		self._filename = filename

	@property
	@abstractmethod
	def path(self):
		pass

	@path.setter
	@abstractmethod
	def path(self, value):
		pass

	@property
	@abstractmethod
	def filename(self):
		pass
	
	@filename.setter
	@abstractmethod
	def filename(self, value):
		pass

	@abstractmethod
	def make_array(self, value):
		pass

	@abstractmethod
	def display_shape(self):
		pass



class CSVGetInfo(AbstractBaseClassCSV):
	""" """

	@property
	def path(self):
		# The docstring for the path property
		print('Getting value of path')
		return self._path

	@path.setter
	def path(self, value):
		if '/' in value:
			self._path = value
			print('Setting value of path to {}'.format(value))
		else:
			print("Error: {} is not a valid path string".format(value))

	@property
	def filename(self):
		# The docstring for the filename property
		print("Getting value of filename")
		return self._filename

	@filename.setter
	def filename(self, value):
		if '.' in value:
			self._filename = value
			print("Setting value of filename to {}".format(value))
		else:
			print("Error: {} is not a valid filename".format(value))

	def make_array(self, value):
		vector = np.array(pd.read_csv(self._path + self._filename, header=None))
		return vector


	def display_shape(self):
		data = np.array(pd.read_csv(self._path + self._filename, header=None))
		print(self._filename)
		print(data.shape)


if __name__ == '__main__':

	data_dd_vector = CSVGetInfo("/Users/georgesavvidis/Documents/PhD/data_analysis/north_south_analysis/cross_talk/csv_files/MPA/tj13s000/MPA_tj13s000_q21/vectors_q21_version4/", "DD_Vectors_q21.csv")

	data_npulses = CSVGetInfo("/Users/georgesavvidis/Documents/PhD/data_analysis/north_south_analysis/cross_talk/csv_files/MPA/tj13s000/MPA_tj13s000_q21/", "DD_NPulses_q21.csv" )

	data_dd_vector.display_shape()
	data_npulses.display_shape()

	dd_vector = data_dd_vector.vectorize(data_dd_vector)

	print(type(data_dd_vector))
	print(type(dd_vector))

	# print(data.__dict__)
	# print(data.__class__)
	# print(data.__dir__)
		
# Data[:,0] = event
# Data[:,1] = number_north
# Data[:,2] = start_north
# Data[:,3] = stop_north
# Data[:,4] = rawampl_north
# Data[:,5] = rawrise_north
# Data[:,6] = rawwidth_north
# Data[:,7] = amplADU_north
# Data[:,8] = rise_north
# Data[:,9] = DD_Rise25pct_North
# Data[:,10] = DD_Rise75pct_North
# Data[:,11] = TimeS_North
# Data[:,12] = TimeMuS_North
# Data[:,13] = number_south
# Data[:,14] = start_south
# Data[:,15] = stop_south
# Data[:,16] = rawampl_south
# Data[:,17] = rawrise_south
# Data[:,18] = rawwidth_south
# Data[:,19] = amplADU_south
# Data[:,20] = rise_south
# Data[:,21] = DD_Rise25pct_South
# Data[:,22] = DD_Rise75pct_South
# Data[:,23] = TimeS_South
# Data[:,24] = TimeMuS_South
# Data[:,25] = number_laser
# Data[:,26] = start_laser
# Data[:,27] = stop_laser
# Data[:,28] = rawampl_laser
# Data[:,29] = rawrise_laser
# Data[:,30] = rawwidth_laser
# Data[:,31] = amplADU_laser
# Data[:,32] = rise_laser
# Data[:,33] = DD_RawRise50pct_Laser
# Data[:,34] = TimeS_Laser
# Data[:,35] = TimeMuS_Laser
# Data[:,36] = DeltaBinPrev_North
# Data[:,37] = DeltaBinNext_North
# Data[:,38] = DeltaBinPrev_South
# Data[:,39] = DeltaBinNext_South
# Data[:,40] = DeltaBinPrev_Laser
# Data[:,41] = DeltaBinNext_Laser

		
	