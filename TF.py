# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 22:03:10 2021

@author: Nikita
"""

import os
import time

import numpy as np
#import h5py
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt
#import sys
#import pandas as pd
import xarray as xr
import pickle as pk

from . import data_file_group_2 as dfg
#import firing_rate as fr
from . import roi_utils as roi
#import spiketrain_manager as spk
from . import useful as usf


def _calc_dfg_tf_inner(X_in, win_len=0.5, win_overlap=0.45, fmax=100,
                       fs=None, var_name='LFP'):
    # Sampling rate
    if fs is None:
        fs = X_in.attrs['proc_steps.(1).0.params.fs.value']
    
    # Select the variable to apply TF transform
    X_in = X_in[var_name]

    # Window and overlap in samples
    win_len_samp = round(win_len * fs)
    win_overlap_samp = round(win_overlap * fs)

    xz = np.zeros(len(X_in.time))
    ff, tt, _ = sig.spectrogram(
        xz, fs=fs, nperseg=win_len_samp, noverlap=win_overlap_samp)
    # Shift positions of W_ time bins to the closest X_in time bins
    idx = np.round(tt * fs).astype(int)
    tt = X_in.time.values[idx]

    def f(X, fs, nperseg, noverlap):
        _, _, S = sig.spectrogram(
            X, fs=fs, nperseg=nperseg, noverlap=noverlap, mode='complex', axis=-1)
        return S
        
    W = xr.apply_ufunc(
        f, X_in,
        kwargs={'fs': fs, 'nperseg': win_len_samp, 'noverlap': win_overlap_samp},
        input_core_dims=[['sample']],
        output_core_dims=[['freq', 'time1']],
        output_sizes={'freq': len(ff), 'time1': len(tt)},
        vectorize=False, dask='parallelized',
        output_dtypes=[np.complex128]
    )
    W = W.rename({'time1': 'time'})
    
    W = W.assign_coords(
         {'freq': ('freq', ff), 'time': ('time', tt)})
    
# =============================================================================
#     # (..., time, ...) -> (..., freq, ..., time)
#     dims = list(X_in.dims) + ['time']
#     dims[time_dim_axis_in] = 'freq'
#     
#     # Remove old coords associated with the time dimension
#     # and add new 'time' and 'freq' coords
#     coords = usf.get_xarray_coords_dict(X_in)
#     coords = {name: coord for name, coord in coords.items()
#               if coord[0] != time_dim_name}
#     coords['time'] = ('time', tt)
#     coords['freq'] = ('freq', ff)
#     
#     # Associate a DataArray object with W_
#     W = xr.DataArray(W_, coords=coords, dims=dims)
# =============================================================================

    # Leave only the frequencies of interest
    W = W.sel(freq=slice(0, fmax))
    
    # Collect the output dataset
    data_vars = {'TF': W}
    X_out = xr.Dataset(data_vars)
    return X_out


def calc_dfg_tf(dfg_in, win_len=0.5, win_overlap=0.45, fmax=100,
                var_name='LFP', need_recalc=True):
    """ Time-frequency transform of epoched LFP data."""

    proc_step_desc = ('TF', 'Time-frequency transform')
    params = {
        'win_len': {
            'val': win_len, 'short': 'wlen',
            'desc': 'Length of the temporal window in TF transform, s'
            },
        'win_overlap': {
            'val': win_overlap, 'short': 'wover',
            'desc': 'Overlap of adjacent time windows, s'
            },
        'fmax': {
            'val': fmax, 'short': 'fmax',
                'desc': 'Upper frequency in TF transform, Hz'
                }
        }
    vars_new_desc = {'TF': 'Time-frequency complex amplitude'}
    coords_new_descs = {
            'freq': 'Frequency, Hz',
            'time': 'Time, s'
            # other coordinates are taken from the input
            }
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _calc_dfg_tf_inner, proc_step_desc, params,
            need_recalc, vars_new_desc, coords_new_descs
            )    
    return dfg_out
