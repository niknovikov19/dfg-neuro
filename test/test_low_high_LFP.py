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

# Data on low-high-power trial pairs
fname_trial_pairs = ('dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_'
                     'trial_pairs2_(perc_sameROI=0.5)')
dfg_trial_pairs = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_trial_pairs))

# Load epoched LFP data
fname_lfp = 'dfg_LFP_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_lfp = os.path.join(dirpath_proc, fname_lfp)
dfg_lfp = dfg.DataFileGroup(fpath_lfp)

chan_name = 'Pancake_20130923_1_ch57'

# Load trial pairs 
entry = dfg_trial_pairs.get_table_entries_by_coords({'chan_name': chan_name})[0]
X_trial_pairs = dfg_trial_pairs.load_inner_data(entry)

# Load epoched LFP
entry = dfg_lfp.get_table_entries_by_coords({'chan_name': chan_name})[0]
X_lfp = dfg_lfp.load_inner_data(entry).LFP

# Remove ERP
X_lfp = X_lfp - X_lfp.mean(dim='trial_num')

# Create filter
fs = 1000
filt_order = 5
freq_band = (15, 40)
sos = sig.butter(filt_order, freq_band, 'bandpass', output='sos', fs=fs)

# Filter the trial data
X_filt_np = sig.sosfiltfilt(sos, X_lfp.values, axis=1)
X_filt = xr.zeros_like(X_lfp)
X_filt.data = X_filt_np

#X_filt = X_lfp

idx_sort = np.argsort(X_trial_pairs.diff_difROI.values)
trial_idx_hival = X_trial_pairs.trial_id_hival.values[idx_sort]
trial_idx_loval = X_trial_pairs.trial_id_loval.values[idx_sort]
diff_difROI = X_trial_pairs.diff_difROI.values[idx_sort]
diff_sameROI = X_trial_pairs.diff_sameROI.values[idx_sort]

#tROI = (0.5, 1.2)
tROI = (0, 1.7)

mask_ROI = (X_lfp.time >= tROI[0]) & (X_lfp.time < tROI[1])
tvec_ROI = X_lfp.time[mask_ROI].values
#X_tROI = usf.xarray_select_xr(X_lfp, {'time': tvec_ROI})
X_tROI = usf.xarray_select_xr(X_filt, {'time': tvec_ROI})

X_tROI_hival = usf.xarray_select_xr(X_tROI, {'trial_num': trial_idx_hival})
X_tROI_loval = usf.xarray_select_xr(X_tROI, {'trial_num': trial_idx_loval})

N = int(len(X_tROI.time) / 2)

d1_lst = []
d2_lst = []

dh = 0.1
Nrows = 25

plt.figure()

#for n in range(len(trial_idx_hival)):
for n in range(Nrows):

    ind = -n
    trial_num_hival = trial_idx_hival[ind]
    trial_num_loval = trial_idx_loval[ind]
    x_hival = X_tROI.loc[{'trial_num': trial_num_hival}]
    x_loval = X_tROI.loc[{'trial_num': trial_num_loval}]
    
    v1_hi = np.var(x_hival.values[:N])
    v1_lo = np.var(x_loval.values[:N])
    v2_hi = np.var(x_hival.values[-N:])
    v2_lo = np.var(x_loval.values[-N:])
    d1 = v1_hi - v1_lo
    d2 = v2_hi - v2_lo
    
    d1_lst.append(d1)
    d2_lst.append(d2)
    
    plt.subplot(1, 2, 1)
    plt.plot(x_hival.time, x_hival + n * dh, 'k')
    plt.subplot(1, 2, 2)
    plt.plot(x_loval.time, x_loval + n * dh, 'k')
    
# =============================================================================
#     plt.figure(101)
#     plt.clf()
#     plt.plot(x_hival.time, x_hival + dh)
#     plt.plot(x_loval.time, x_loval)
#     #plt.title(f'{v1_hi:.4e} - {v1_lo:.4e}  {v2_hi:.4e} - {v2_lo:.4e}')
#     plt.title(f'{d1:.4e}    {d2:.4e}')
#     plt.draw()
#     if plt.waitforbuttonpress():
#         break
# =============================================================================

for m in [1, 2]:
    plt.subplot(1, 2, m)
    plt.plot([0.5, 0.5], [0, Nrows * dh], 'b')
    plt.plot([0.85, 0.85], [0, Nrows * dh], 'b')
    plt.plot([1.2, 1.2], [0, Nrows * dh], 'b')

# =============================================================================
# plt.figure()
# plt.plot(d1_lst, '.')
# plt.plot(d2_lst, '.')
# =============================================================================


