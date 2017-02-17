"""
Tools for spectral analysis

Jacob Dein 2016
nacoustik
Author: Jacob Dein
License: MIT
"""


import numpy as np
from scipy.signal import spectrogram, get_window
from nacoustik import Wave


def psd(wave, rate = None, units = 'decibels', scaling = 'density', kind = 'spectrogram',
        window_length = 1024, window_overlap = 50, window_shape = 'hann', 
        pressure_reference = 20.):
    """
    Estimate the power spectral density (psd) of a wave
    
    Parameters
    ----------
    wave: Wave object, file path to a WAV file, or numpy array of WAV signal samples
    
    rate: sample rate of signal, default = None
        required when 'wave' is a numpy array
        if 'None', the rate will be determined by the 'wave' object
    
    units: string, default = 'decibels'
        result units in 'decibels' or 'watts'
        
    scaling: string, default = 'density'
        result scaling, 'spectrum' or 'density'
    
    kind: string, default = 'spectrogram'
        result type, 'spectrogram', 'mean', or 'both'
    
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
    if kind not in ['spectrogram', 'mean', 'both']:
        raise ValueError("'{0}' is not an acceptable kind".format(kind))
    
    # convert window_overlap percent value to decimal value
    window_overlap = window_overlap / 100.
    
    # compute the number of analysis windows (used to allocate the result array)
    n_windows = int(np.ceil(wave.n_samples - (window_overlap * window_length)) / ((1 - window_overlap) * window_length))
    
    # compute the psd spectrogram for each channel
    psd = np.array( [ np.empty(shape = (int((window_length / 2) + 1), n_windows)) for channel in wave.channels ] )
    for channel in wave.channels:
        f, t, psd[channel] = spectrogram(wave.samples[:, channel], 
                                         fs = rate, 
                                         window = window_shape,
                                         nperseg = window_length, 
                                         noverlap = window_length * window_overlap, 
                                         return_onesided = True, 
                                         scaling = scaling)
    
    # compute psd mean (RMS mean)
    if kind in ['mean', 'both']:
        psd_mean = ( psd.sum(axis = 2) / psd.shape[2] )
        
    # convert to decibels
    if units == 'decibels':
        if kind == 'mean':
            f, t, 10 * np.log10(psd_mean / (pressure_reference**2))
        elif kind == 'both':
            return f, t, 10 * np.log10(psd / (pressure_reference**2)), 10 * np.log10(psd_mean / (pressure_reference**2))
        else:
            return f, t, 10 * np.log10(psd / (pressure_reference**2))
    # return watts
    else:
        if kind == 'mean':
            return f, t, psd_mean
        elif kind == 'both':
            return f, t, psd, psd_mean
        else:
            return f, t, psd


def sel(a, rate, duration, b=None, limit=2000, bin_width = 1000, return_bins=False):
    """
    Estimate the sound exposure level (sel) per minute from a wave
    
    Parameters
    ----------
    a: numpy float64 array, required
        a 3d array (channels, frequency bands, time steps)
        representing the psd (power spectral density)
        spectrogram of a wave signal in decibels
        
    rate: sample rate of signal, required
        
    duration: duration of signal in minutes, required
    
    b: numpy float64 array, default = None
        a 3d array (channels, frequency bands, time steps)
        representing the spectrogram of a wave signal (with ale applied)
        
    limit: numpy float64 array, default = None
        frequency separating anthrophony and biophony
    
    bin_width: int, default = 1000
        width of frequency bins in herz
        
    bins: boolean, default = False
        return values for each frequency bin of specified bin_width

    """    
    
    # convert to watts
    a = 10**(a / 10)
    
    # multiply by frequency delta
    f_delta = (rate / 2) / (a.shape[1] - 1)
    a = a * f_delta
    
    # compute sel for separate frequency bins if b is None
    if b is None:
        # compute frequency bins of spectrogram
        f = np.arange(0, (rate / 2) + f_delta, f_delta)
        # determine frequency bins for sel
        bins = np.arange(0, (rate / 2), bin_width)
        bin_bound_indicies = np.searchsorted(f, bins)
        # include last index
        bin_bound_indicies = np.append(bin_bound_indicies, a.shape[1])
        
        # compute sel
        sel = np.empty(len(bins))
        for i in (bins / bin_width).astype(np.int):
            low_bound = bin_bound_indicies[i]
            high_bound = bin_bound_indicies[i + 1]
            sel[i] = (a[:, low_bound:high_bound, :].sum())
        
        # divide by wave duration in minutes
        sel = sel / duration
        
        # calculate anthrophony and biophony
        anthrophony = sel[0:2].sum()
        biophony = sel[2:10].sum()
        
    else:
        # do not return bins if b is not None
        return_bins = False
        # convert to watts
        b = 10**(b / 10)
        # multiply by frequency delta
        b = b * f_delta
        
        # compute anthrophony and biophony
        anthrophony = (a - (b.data * np.invert(b.mask))).sum() / duration
        biophony = b.sum() / duration
    
    # convert back to decibels
    anthrophony = 10 * np.log10(anthrophony)
    biophony = 10 * np.log10(biophony)
    if return_bins is True:
        sel = 10 * np.log10(sel)
        return sel, anthrophony, biophony
    else:
        return anthrophony, biophony