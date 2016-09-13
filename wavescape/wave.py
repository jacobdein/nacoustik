"""
Wave class

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
		
		Parameters
		----------
		filepath: string
			file path to wave file
		"""
		
		self.filepath = filepath
		self.basename = path.basename(filepath)
		
		# properties
		self.bit_depth = file_info.bitrate(filepath)		# bit depth
		self.n_samples = file_info.num_samples(filepath)	# number of samples
		self.n_channels = file_info.channels(filepath)		# number of channels
		self.channels = np.arange(self.n_channels)			# channels
		self.duration = file_info.duration(filepath)		# duration
		
		def __str__(self):
			return self.basename


	def read(self, normalize = False):
		"""
		Read wave file
		
		Parameters
		----------
		normalize: string, default = False
			normalize the wave based on the potential maximum value
			that is determined by the bit depth of each sample
		"""
		
		self.rate, self.samples = wavfile.read(self.filepath)
		if normalize:
			self.samples = self.samples / (2.**(self.bit_depth - 1))