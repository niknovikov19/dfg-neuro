# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:40:07 2024

@author: Nikita
"""

import numpy as np
import scipy as sc
import xarray as xr

from . import data_file_group_2 as dfg
from . import useful as usf


def _calc_troi_fft_inner(X_in, t_rois, fs=None, var_name='LFP'):
    
    # Sampling rate
    if fs is None:
        fs = X_in.attrs['proc_steps.(1).0.params.fs.value']
    
    # Select the variable to apply FFT
    X_in = X_in[var_name]
    
    X_in = X_in.swap_dims({'sample': 'time'})
    t_axis = X_in.dims.index('time')
    if X_in.dims[-1] != 'time':
        raise ValueError('Time should be the last dimension')
    
    # New coordinates
    t_roi_lens = np.array([t_roi[1] - t_roi[0] for t_roi in t_rois])
    if not np.all(np.isclose(t_roi_lens, t_roi_lens[0], atol=1e-8)):
        raise ValueError('All time ROIs should have the same length')    
    tt = X_in.time.sel(time=slice(*t_rois[0]))
    ff = sc.fft.fftfreq(len(tt), 1 / fs)
    f_mask = (ff >= 0)
    ff = ff[f_mask]
    t_rois_str = [str(t_roi) for t_roi in t_rois]
    coords_new = {'freq': ('freq', ff),
                  'time_roi': ('time_roi', t_rois_str)}
    
    # Allocate output array
    X_out = usf.create_compatible_xarray(
        X_in, dims_rep=[('time', 'freq')], dims_new=['time_roi'],
        coords_new=coords_new, dtype=np.complex128)
    
    # Window function
    sz = [1] * len(X_in.dims)
    sz[t_axis] = len(tt)
    w = np.hanning(len(tt)).reshape(sz)

    def chunk_proc(X, axis, taper, f_mask):
        Y = sc.fft.fft(X * taper, axis=axis)
        Y = usf.slice_ndarray(Y, f_mask, axis)
        return Y
    
    for t_roi in t_rois:
        Xroi_in = X_in.sel(time=slice(*t_roi))        
        #Xroi_in = Xroi_in.compute()
        Xroi_out = xr.apply_ufunc(
            chunk_proc, Xroi_in,
            kwargs={'axis': t_axis, 'taper': w, 'f_mask': f_mask},
            input_core_dims=[['time']],
            output_core_dims=[['freq']],
            output_sizes={'freq': len(ff)},
            vectorize=False,
            #dask='forbidden',
            dask='parallelized',
            output_dtypes=[np.complex128]
        )
        X_out.loc[{'time_roi': str(t_roi)}] = Xroi_out
    
    # Collect the output dataset
    X_out = xr.Dataset({'TF': X_out})
    return X_out


def calc_dfg_troi_fft(dfg_in, t_rois, var_name='LFP', need_recalc=True):
    """ Windowed FFT in several time ROIs. """

    proc_step_desc = ('FFT', 'Windowed FFT in several time ROIs')
    params = {
        't_rois': {
            'val': t_rois, 'short': 't',
            'desc': 'Time windows to calculate FFT in'
            },
        'var_name': {
            'val': var_name, 'short': 'var',
            'desc': 'Variable to process'
            }
        }
    vars_new_desc = {'TF': 'Complex amplitude in time window'}
    coords_new_descs = {
            'freq': 'Frequency, Hz',
            'time_roi': 'Time ROI'
            # other coordinates are taken from the input
            }
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _calc_troi_fft_inner, proc_step_desc, params,
            need_recalc, vars_new_desc, coords_new_descs
            )    
    return dfg_out
