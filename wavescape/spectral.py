"""
Tools for spectral analysis

Jacob Dein 2016
wavescape
Author: Jacob Dein
License: MIT
"""


import numpy as np
from scipy.fftpack import fft
from scipy.signal import hann, spectrogram


def psd(wave, units = 'decibels', scaling = 'spectrum'):
	"""
	Estimate power spectral density ...
	
	Parameters
	----------
	wave : wavescape.wave object
		reference to wave object
	
	units : string, optional
		result units, 'decibels' or 'watts'
		default: 'decibels'
		
	scaling : string, option
		result scaling, 'spectrum' or 'density'
		default: 'spectrum'
	"""
	
	window_length = wave.rate / 10
	window_overlap = 0.5
	window_shape = 'hann'
	pressure_reference = 20.
	n_windows = int(np.ceil(wave.n_samples - (window_overlap * window_length)) / ((1 - window_overlap) * window_length))
	
	w = hann(window_length, sym=False)
	alpha = 0.5
	b = (1 / window_length) * np.sum((w / alpha)**2)
	
	# compute spectrogram for each channel
	psd = np.array( [ np.empty(shape = (int((window_length / 2) + 1), n_windows)) for channel in wave.channels ] )
	for channel in wave.channels:
		f, t, psd[channel] = spectrogram(wave.samples[:,channel], fs=wave.rate, window=window_shape, 
		                              nperseg=window_length, noverlap=window_length*window_overlap, 
		                              return_onesided=True, scaling=scaling)
		# b term
		psd = (1 / b) * psd
		
	# convert to decibels
	if units == 'decibels':
		return f, t, 10 * np.log10(psd / (pressure_reference**2))
	else:
		return f, t, psd


def sel(wave, units = 'decibels'):
	"""
	Estimate sound exposure level ...
	
	Parameters
	----------
	wave : wavescape.wave object
		reference to wave object
	
	units : string, optional
		result units, 'decibels' or 'watts'
		default: 'decibels'
	
	"""
	# compute psd
	f, t, a = psd(wave, units = 'watts', scaling = 'spectrum')
	
	# frequency bins
	bins = np.arange(0, (wave.rate / 2), 1000)
	bin_bound_indicies = np.searchsorted(f, bins)
	# include last index
	bin_bound_indicies = np.append(bin_bound_indicies, a.shape[1])
	
	# compute sel
	sel = np.empty(len(bins))
	for i in (bins / 1000).astype(np.int):
	    low_bound = bin_bound_indicies[i]
	    high_bound = bin_bound_indicies[i + 1]
	    sel[i] = (a[:, low_bound:high_bound, :].sum())
	
	# divide by wave duration
	sel = sel / (wave.n_samples / wave.rate / 60)
	
	# calculate anthrophony and biophony
	anthrophony = sel[0:2].sum()
	biophony = sel[2:10].sum()
	
	# convert to decibels
	if units == 'decibels':
		pressure_reference = 20.
		sel = 10 * np.log10(sel / (pressure_reference**2))
		anthrophony = 10 * np.log10(anthrophony / (pressure_reference**2))
		biophony = 10 * np.log10(biophony / (pressure_reference**2))
		return sel, anthrophony, biophony
	else:
		return sel, anthrophony, biophony
