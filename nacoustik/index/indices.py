"""
Index computation

Jacob Dein 2016
nacoustik
Author: Jacob Dein
License: MIT
"""


import numpy as np
from numba import guvectorize, float64


# implemented as a universal function via numba.guvectorize
@guvectorize([(float64[:,:,:], float64[:], float64[:], float64[:,:])], 
             '(c,f,t),(),()->(c,f)', nopython=True)
def _calculate_aci(a, time_delta, block_duration, aci):
    block_delta = int(np.around(block_duration[0] / time_delta[0]))
    n_blocks = int(np.floor(a.shape[2] / block_delta))
    remainder = int(a.shape[2] - (block_delta * n_blocks))
    for channel in range(a.shape[0]):
        for f_band in range(a.shape[1]):
            aci_f_band = np.empty(shape=(n_blocks + 1))
            for block in range(n_blocks):
                d = np.empty(shape=(block_delta - 1))
                for t_step in range(block_delta - 1):
                    d[t_step] = np.abs(a[channel, \
                                         f_band, \
                                         (t_step * (block + 1))] - \
                                       a[channel, \
                                         f_band, \
                                         ((t_step * (block + 1)) + 1)])
                D = d.sum()
                aci_f_band[block] = D / a[channel, f_band, \
                                          (block_delta * block): \
                                          (block_delta * (block + 1))].sum()
            if remainder > 1:
                d = np.empty(shape=(remainder - 1))
                for t_step in range(remainder - 1):
                    d[t_step] = np.abs(a[channel, \
                                         f_band, \
                                         ((t_step * n_blocks) + t_step)] - \
                                       a[channel, \
                                         f_band, \
                                         ((t_step * n_blocks) + t_step + 1)])
                D = d.sum()
                aci_f_band[-1] = D / a[channel, \
                                       f_band, \
                                       -(remainder + 1):-1].sum()
            # average aci value over blocks
            aci[channel, f_band] = aci_f_band.sum() / \
                                    (n_blocks + (remainder / block_delta))

            
def calculate_aci(a, time_delta, block_duration=1.):
    """
    Calculates the acoustic complexity index 
    as defined in Pieretti, et al. 2011.
    
    Pieretti, N., A. Farina, and D. Morri. 2011. A new methodology to 
    infer the singing activity of an avian community: 
    The Acoustic Complexity Index (ACI). Ecological Indicators 11: 868-873. 
    doi: 10.1016/j.ecolind.2010.11.005
    
    Parameters
    ----------
    a: numpy float64 array
        a 3d array (channels, frequency bands, time steps)
        representing the spectrogram of a wave signal
        values should be in watts
    
    time_delta: float
        amount of time in seconds represented by each window of the spectrogram
    
    block_duration: float, default = 1.
        duration in seconds of each calculation interval 
        used in the aci algorithm
    
    Returns
    ----------
    aci: numpy float64 array
        a 2d array (channels, frequency bands)
        containing the calculated aci for each frequency band
    """
    
    # determine number of histogram bins
    return _calculate_aci(a, 
                          time_delta, 
                          block_duration,
                          np.zeros(shape=(a.shape[0], a.shape[1])))