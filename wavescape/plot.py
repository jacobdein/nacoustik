"""
plots

Jacob Dein 2016
wavescape
Author: Jacob Dein
License: MIT
"""


import numpy as np
from wavescape.analysis import psd
from wavescape.wave import Wave
#from wavescape.colormaps import spectro_white
import matplotlib.pyplot as plt
from matplotlib import rcParams


def plot_spectrogram(wave, rate = None, units = 'decibels', scaling = 'density', 
		window_length = 1000, window_overlap = 50, window_shape = 'hann', 
		pressure_reference = 20.):
	"""
	Plot the power spectral density (psd) of a wave
	
	Parameters
	----------
	wave: Wave object
		reference to a wavescape Wave object
		
	rate: sample rate of signal, default = None
		required when 'wave' is a numpy array
		if 'None', the rate will be determined by the 'wave' object
	
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
	# check wave
	if type(wave) is not Wave:
		wave = Wave(wave)
	if not hasattr(wave, 'samples'):
		wave.read()
	# check rate
	if rate is None:
		rate = wave.rate
	# check units
	if units not in ['decibels', 'watts']:
		raise ValueError("'{0}' are not acceptable units".format(units))
	
	# parameters to add
	cmap = 'gray_r'
	dpi = 192
	# fig_width
	# fig_height
	
	# compute psd
	f, t, a, a_mean = psd(wave, rate, units = units, scaling = 'density', kind = 'both', 
						  window_length = window_length, window_overlap = window_overlap, window_shape = window_shape, 							  pressure_reference = 20.)
	
	# configure figure
	fig = plt.figure(figsize=((920 / dpi) * 3, (460 / dpi) * 3), dpi=dpi)
	plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
	
	# register colormap
	#plt.register_cmap(cmap=spectro_white())
	fig.set_frameon(False)
	
	# specify frequency bins (width of 1 kiloherz)
	bins = np.arange(0, (rate / 2), 1000)
	
	# psd spectrogram left
	ax_spec_l = plt.subplot2grid((2, 10), (0, 0), rowspan=1, colspan=9)
	spec_l = ax_spec_l.pcolormesh(t, f, a[0], cmap=cmap, vmin=-150, vmax=-50)
	ax_spec_l.set(ylim=([0, rate / 2]),
	              xticks = np.arange(30, (wave.n_samples / rate), 30).astype(np.int),
	              yticks = bins.astype(np.int) + 1000)
	ax_spec_l.tick_params(length=12,
	                      bottom=False, labelbottom=False,
	                      top=False, labeltop=False,
	                      labelleft=False,
	                      labelright=False)
	ax_spec_l.set_frame_on(False)
	
	# psd mean left
	ax_mean_l = plt.subplot2grid((2, 10), (0, 9), rowspan=1, colspan=1)
	ax_mean_l.plot(a_mean[0], f, color='black')
	ax_mean_l.set(frame_on=False,
	              xlim=(-150, -100),
	              ylim=(0, rate / 2),
	              xticks = [-150, -100])
	ax_mean_l.tick_params(length=12,
	                      bottom=False, labelbottom=False,
	                      top=False, labeltop=False,
	                      left=False, labelleft=False,
	                      right=False, labelright=False)
	
	# psd spectrogram right
	ax_spec_r = plt.subplot2grid((2, 10), (1, 0), rowspan=1, colspan=9)
	spec_r = ax_spec_r.pcolormesh(t, f, a[1], cmap=cmap, vmin=-150, vmax=-50)
	ax_spec_r.set(ylim=([0, rate / 2]),
	              xticks = np.arange(30, (wave.n_samples / rate), 30).astype(np.int),
	              yticks = bins.astype(np.int) + 1000)
	ax_spec_r.tick_params(length=12,
	                      bottom=False, labelbottom=False,
	                      top=False, labeltop=False,
	                      labelleft=False,
	                      labelright=False)
	ax_spec_r.set_frame_on(False)
	
	# psd mean right
	ax_mean_r = plt.subplot2grid((2, 10), (1, 9), rowspan=1, colspan=1)
	ax_mean_r.plot(a_mean[1], f, color='black')
	ax_mean_r.set(frame_on=False,
	              xlim=(-150, -100),
	              ylim=(0, rate / 2),
	              xticks = [-150, -100])
	ax_mean_r.tick_params(length=12,
	                      bottom=False, labelbottom=False,
	                      top=False, labeltop=False,
	                      left=False, labelleft=False,
	                      right=False, labelright=False)
	
	# text
	# set font
	rcParams['font.family'] = 'sans-serif'
	rcParams['font.sans-serif'] = ['Input Sans', 'sans-serif']
	# add background to 'max frequency' text
	bbox_properties = dict(boxstyle="square, pad=0", ec='white', fc='white')
	ax_spec_l.text(x=0, y=(rate / 2), s="{0:.0f} herz".format(rate / 2),
	               va='top', size=12, bbox=bbox_properties)
	ax_mean_l.text(x=-100, y=(rate / 2), s="left", ha='right', va='top', size=12)
	ax_spec_r.text(x=0, y=(rate / 2), s="{0:.0f} herz".format(rate / 2), 
	               va='top', size=12, bbox=bbox_properties)
	ax_mean_r.text(x=-100, y=(rate / 2), s="right", ha='right', va='top', size=12)
	
	#plt.savefig(plot_filepath, dpi=(dpi / 3))
	#fig.clear()
	#plt.close()
	plt.show()