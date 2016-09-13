"""
utilities

Jacob Dein 2016
wavescape
Author: Jacob Dein
License: MIT
"""


import numpy as np


def sum_decibels(x):
	"""Sum an array of decibels
	
	Parameters
	----------
	x: numpy array
		array of decibel values to sum
	"""
	
	return 10 * np.log10(np.sum(10**(x / 10)))