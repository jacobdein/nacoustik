"""
Wave class, utilities

Jacob Dein 2016
wavescape
Author: Jacob Dein
License: MIT
"""


from os import path
from sox import file_info
import numpy as np
from scipy.io import wavfile


class Wave:
	"""Create wave object"""
	
	
	def __init__(self, filepath):
		"""
		
		filepath: string
			file path to wave file
			
		"""
		
		self.filepath = filepath
		self.basename = path.basename(filepath)
		
		# sox properties
		# bit depth
		self.bit_depth = file_info.bitrate(filepath)
		# number of samples
		self.n_samples = file_info.num_samples(filepath)
		# number of channels
		self.n_channels = file_info.channels(filepath)
		# channels
		self.channels = np.arange(self.n_channels)
		# duration
		self.duration = file_info.duration(filepath)
		
		def __str__(self):
			return self.basename


	def read(self, normalize = False):
		"""Read wave file
		
		normalize: string
			normalize file based on maximum value
			default: False
		"""
		self.rate, self.samples = wavfile.read(self.filepath)
		if normalize:
			self.samples = self.samples / (2.**(self.bit_depth - 1))


def sum_decibels(x):
	"""Sum an array of decibels
	
	x: numpy array
		array of decibel values to sum
		
		"""
	return 10 * np.log10(np.sum(10**(x/10)))