"""
Wave class

Jacob Dein 2016
wavescape
Author: Jacob Dein
License: MIT
"""


from sys import stderr
from os import path
from sox import file_info
import numpy as np
from scipy.io import wavfile


class Wave:
	"""Create wave object"""
	
	
	def __init__(self, wave):
		"""
		
		Parameters
		----------
		wave: file path to WAV file or numpy array of a WAV signal samples
			array must be in the shape (n_samples, n_channels)

		"""
		
		if type(wave) is str:
			self.filepath = wave
			self.basename = path.basename(wave)
			
			# properties
			self.bit_depth = file_info.bitrate(wave)		# bit depth
			self.n_samples = file_info.num_samples(wave)	# number of samples
			self.n_channels = file_info.channels(wave)		# number of channels
			self.duration = file_info.duration(wave)		# duration
		else:
			self.samples = wave
			self.n_samples = len(wave)						# number of samples
			self.n_channels = wave.shape[1]					# number of channels
		self.channels = np.arange(self.n_channels)			# channels
		self.normalized = False								# normalized

		# def __str__():


	def normalize(self, value = None):
		"""
		Normalize wave file
		
		Parameters
		----------
		value: float, default = None
			normalize the wave signal
			If 'None', the wave will be normalized
			based on the potential maximum value
			that is determined by the bit depth of each sample
		"""
		
		try:
			self.samples = self.samples / (2.**(self.bit_depth - 1))
			self.normalized = True
		except AttributeError as error:
			print(error, file = stderr)


	def read(self):
		"""
		Read wave file

		"""
		
		try:
			self.rate, self.samples = wavfile.read(self.filepath)
		except AttributeError as error:
			print(error, file = stderr)