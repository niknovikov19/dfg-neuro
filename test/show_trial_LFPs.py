# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import copy
#import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import pickle as pk
import scipy as sc
import scipy.signal as sig
import scipy.stats as stat
import tqdm
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
import useful as usf
#import trial_manager as trl
#import spiketrain_manager as spk
#import firing_rate as fr
#import lfp
#import spike_corr as spcor
#import vis

#import roi_utils as roi
#import spike_TF_PLV as spPLV
#import useful as usf


dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

# Load epoched LFP data
fname_lfp = 'dfg_LFP_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_lfp = os.path.join(dirpath_proc, fname_lfp)
dfg_lfp = dfg.DataFileGroup(fpath_lfp)

#chan_name = 'Pancake_20130923_1_ch9'

for chan_name in dfg_lfp.outer_table.chan_name:

    # Load epoched LFP
    entry = dfg_lfp.get_table_entries_by_coords({'chan_name': chan_name})[0]
    X_lfp = dfg_lfp.load_inner_data(entry).LFP
    
    # Remove ERP
    X_lfp = X_lfp - X_lfp.mean(dim='trial_num')
    
    # Create filter
    fs = 1000
    filt_order = 5
    freq_band = (15, 35)
    #freq_band = (45, 100)
    sos = sig.butter(filt_order, freq_band, 'bandpass', output='sos', fs=fs)
    
    # Filter the trial data
    X_filt_np = sig.sosfiltfilt(sos, X_lfp.values, axis=1)
    X_filt = xr.zeros_like(X_lfp)
    X_filt.data = X_filt_np
    
    #tROI = (-0.5, 2)
    tROI = (-1, 3)
    
    mask_ROI = (X_lfp.time >= tROI[0]) & (X_lfp.time < tROI[1])
    tvec_ROI = X_lfp.time[mask_ROI].values
    X_tROI = usf.xarray_select_xr(X_filt, {'time': tvec_ROI})
    
    s = np.std(X_tROI.values.ravel())
    
    dh = 4 * s
    Nrows = 100
    Ncols = 2
    
    plt.figure(200)
    plt.clf()
    
    for m in range(Ncols):
        
        plt.subplot(1, Ncols, m + 1)
        
        for n in range(Nrows):
            
            x = X_tROI.loc[{'trial_num': m * Nrows + n}]
            
            w = np.abs(sig.hilbert(x))
            y = x.copy()
            mask = (w < 1.7 * s)
            y[mask] = np.nan
            
            plt.plot(x.time, x + n * dh, 'k')
            plt.plot(y.time, y + n * dh, 'r')
            plt.title(chan_name)

# =============================================================================
#         plt.plot([0, 0], [0, Nrows * dh], 'b')
#         plt.plot([0.5, 0.5], [0, Nrows * dh], 'b')
#         #plt.plot([0.85, 0.85], [0, Nrows * dh], 'r')
#         plt.plot([1.2, 1.2], [0, Nrows * dh], 'b')
# =============================================================================

    plt.draw()
    if plt.waitforbuttonpress():
        break


