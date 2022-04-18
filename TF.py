# -*- coding: utf-8 -*-

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

import data_file_group_2 as dfg
#import firing_rate as fr
import roi_utils as roi
#import spiketrain_manager as spk
import useful as usf


def _calc_dfg_TF_inner(X_in, win_len=0.5, win_overlap=0.45, fmax=100):

    # Sampling rate
    fs = X_in.attrs['proc_steps.(1).0.params.fs.value']
    
    X_in = X_in.LFP

    # Window and overlap in samples
    win_len_samp = round(win_len * fs)
    win_overlap_samp = round(win_overlap * fs)
    
    Ntrials = X_in.trial_num.size
    W = None

    for m in range(Ntrials):

        x = X_in.isel(trial_num=m)
        
        # TF of the current channel + trial
        (ff, tt, Wcur) = sig.spectrogram(
            x, fs, mode='complex', window=sig.windows.hamming(win_len_samp),
            noverlap=win_overlap_samp)
        
        idx = np.round(tt * fs).astype(int)
        tt = X_in.time.data[idx]
        
        # Frequencies of interest
        idx = (ff < fmax)
        ff = ff[idx]
        Wcur = Wcur[idx,:]
        
        # Allocate output
        if W is None:            
            coords = {
                'freq': ff,
                'time': tt,
                'trial_num': X_in.trial_num,
                'trial_id': ('trial_num', X_in.trial_id)
                }
            Wnan = np.nan * np.ones((len(ff), len(tt), Ntrials),
                                    dtype='complex128')
            W = xr.DataArray(Wnan, coords=coords,
                             dims=['freq', 'time', 'trial_num'])
            
        # Store TF of the current trial
        W[:,:,m] = Wcur
        
    # Collect the output dataset
    data_vars = {'TF': W}
    X_out = xr.Dataset(data_vars)
    return X_out


def calc_dfg_TF(dfg_in, win_len=0.5, win_overlap=0.45, fmax=100,
                need_recalc=True):
    """ Time-frequency transform of epoched LFP data.

    """
    
    print('calc_dfg_TF')
    
    # Name of the processing step
    proc_step_name = 'Time-frequency transform'
    
    # Dictionary of parameters
    param_names = ['win_len', 'win_overlap', 'fmax']
    local_vars = locals()
    params = {par_name: local_vars[par_name] for par_name in param_names}
    
    # Name of the dfg's outer table column for the paths to Dataset files
    fpath_data_column = 'fpath_tf'

    # Function that converts the parameters dict to the form suitable
    # for storing into a processing step description
    def gen_proc_step_params(par):
        par_out = {
            'time_window_len': {
                'desc': 'Length of the temporal window in TF transform, s',
                'value': par['win_len']},
            'time_window_overlap': {
                'desc': 'Overlap of adjacent time windows, s',
                'value': par['win_overlap']},
            'fmax': {
                'desc': 'Upper frequency in TF transform, Hz',
                'value': par['fmax']}
        }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath(fpath_in, params):
        fpath_data_postfix = (
            f'(wlen={params["win_len"]}_'
            f'wover={params["win_overlap"]}_'
            f'fmax={params["fmax"]})')
        fpath_noext, ext  = os.path.splitext(fpath_in)
        return fpath_noext + '_' + fpath_data_postfix + ext
    
    # Description of the new variables
    vars_new_descs = {'TF': 'Time-frequency complex amplitude'}
    
    # Description of the new coordinates
    coords_new_descs = {
            'freq': 'Frequency, Hz',
            'time': 'Time, s',
            'trial_num': 'Trial number (sequential)',
            'trial_id': 'Trial number in the experiment'
    }
    
    # Call calc_dataset_ROIs() for each inner dataset of the DataFileGroup
    dfg_out = dfg.apply_dfg_inner_proc_mt(
            dfg_in, _calc_dfg_TF_inner, params, proc_step_name,
            gen_proc_step_params, fpath_data_column, gen_fpath,
            vars_new_descs, coords_new_descs, need_recalc)
    
    return dfg_out
