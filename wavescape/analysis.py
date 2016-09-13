"""
Tools for spectral analysis

Jacob Dein 2016
wavescape
Author: Jacob Dein
License: MIT
"""


import numpy as np
from scipy.fftpack import fft
from scipy.signal import spectrogram, get_window


def psd(wave, units = 'decibels', scaling = 'density', 
		window_length = 1000, window_overlap = 50, window_shape = 'hann', 
		pressure_reference = 20.):
	"""
	Estimate the power spectral density (psd) of a wave
	
	Parameters
	----------
	wave: Wave object
		reference to a wavescape Wave object
	
	units: string, default = 'decibels'
		result units in 'decibels' or 'watts'
		
	scaling: string, default = 'density'
		result scaling, 'spectrum' or 'density'
	
	window_length: integer, default = 1000
		length of analysis window in number of samples
	
	window_overlap: integer, default = 50
		amount of analysis window overlap in percent
	
	window_shape: string, default = 'hann'
		shape of analysis window,
		refer to scipy.signal for window types
	
	pressure_reference: float, default = 20.
		reference pressure for measurements in air in micropascals
	"""
	
	# check parameters
	if units not in ['decibels', 'watts']:
		raise ValueError("'{0}' are not acceptable units".format(units))
	
	# convert window_overlap percent value to decimal value
	window_overlap = window_overlap / 100.
	
	# compute the number of analysis windows (used to allocate the result array)
	n_windows = int(np.ceil(wave.n_samples - (window_overlap * window_length)) / ((1 - window_overlap) * window_length))
	
	# compute the psd spectrogram for each channel
	psd = np.array( [ np.empty(shape = (int((window_length / 2) + 1), n_windows)) for channel in wave.channels ] )
	for channel in wave.channels:
		f, t, psd[channel] = spectrogram(wave.samples[:, channel], 
										 fs = wave.rate, 
										 window = window_shape,
										 nperseg = window_length, 
										 noverlap = window_length * window_overlap, 
										 return_onesided = True, 
										 scaling = scaling)
		
	# convert to decibels
	if units == 'decibels':
		return f, t, 10 * np.log10(psd / (pressure_reference**2))
	else:
		return f, t, psd


def sel(wave, units = 'decibels', bin_width = 1000, 
		window_length = 1000, window_overlap = 50, window_shape = 'hann', 
		pressure_reference = 20.):
	"""
	Estimate the sound exposure level (sel) per minute from a wave
	
	Parameters
	----------
	wave : Wave object
		reference to a wavescape Wave object
	
	units : string, default = 'decibels'
		result units in 'decibels' or 'watts'
	
	bin_width : int, default = 1000
		width of frequency bins in herz
		
	scaling: string, default = 'density'
		result scaling, 'spectrum' or 'density'
	
	window_length: integer, default = 1000
		length of analysis window in number of samples
	
	window_overlap: integer, default = 50
		amount of analysis window overlap in percent
	
	window_shape: string, default = 'hann'
		shape of analysis window
		refer to scipy.signal for window types
	
	pressure_reference: float, default = 20.
		reference pressure for measurements in air in micropascals
	"""
	
	# check parameters
	if units not in ['decibels', 'watts']:
		raise ValueError("'{0}' are not acceptable units".format(units))
	
	# convert window_overlap percent value to decimal value
	window_overlap = window_overlap / 100.
		
	# compute power spectrum
	f, t, pss = psd(wave, units = 'watts', scaling = 'spectrum', 
					window_length = window_length, window_overlap = 50, window_shape = 'hann')
	
	# compute and apply 'b' term
	w = get_window(window_shape, window_length, fftbins = True)
	alpha = 0.5		# need to investigate if this changes for different window types
	b = (1 / window_length) * np.sum((w / alpha)**2)
	pss = (1 / b) * pss

	# determine frequency bins
	bins = np.arange(0, (wave.rate / 2), bin_width)
	bin_bound_indicies = np.searchsorted(f, bins)
	# include last index
	bin_bound_indicies = np.append(bin_bound_indicies, pss.shape[1])
	
	# compute sel
	sel = np.empty(len(bins))
	for i in (bins / bin_width).astype(np.int):
	    low_bound = bin_bound_indicies[i]
	    high_bound = bin_bound_indicies[i + 1]
	    sel[i] = (pss[:, low_bound:high_bound, :].sum())
	
	# divide by wave duration in minutes
	sel = sel / (wave.n_samples / wave.rate / 60)
	
	# calculate anthrophony and biophony
	anthrophony = sel[0:2].sum()
	biophony = sel[2:10].sum()
	
	# convert to decibels
	if units == 'decibels':
		sel = 10 * np.log10(sel / (pressure_reference**2))
		anthrophony = 10 * np.log10(anthrophony / (pressure_reference**2))
		biophony = 10 * np.log10(biophony / (pressure_reference**2))
		return sel, anthrophony, biophony
	else:
		return sel, anthrophony, biophony
