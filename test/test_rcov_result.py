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

fname_rcov = ('dfg_rcov_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)_'
              '(bins=5_iter=50_lags=31)')
fname_trial_pairs = ('dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_'
                     'trial_pairs2_(perc_sameROI=0.5)')

# Data on rate covariance and low-high-power trial pairs
dfg_rcov = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_rcov))
dfg_trial_pairs = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_trial_pairs))

# Load epoched spiketrains
fname_cell_epoched_info = 'cell_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_cell_epoched_info = os.path.join(dirpath_proc, fname_cell_epoched_info)
with open(fpath_cell_epoched_info, 'rb') as fid:
    cell_epoched_info = pk.load(fid)


def calc_lag_ROI_symm(X, lag_range):
    lags_used = X.lags.values
    mask = (lags_used >= lag_range[0]) & (lags_used <= lag_range[1])
    lags_used = lags_used[mask]
    lagROI = {'lags': lags_used}
    X_lagROI = usf.xarray_select_xr(X, lagROI)
    X_lagROI = X_lagROI.mean(dim='sample_num', skipna=True)
    return X_lagROI

    
#sess_idx_uni = np.unique(cell_epoched_info.sess_id)
#sess_id = sess_idx_uni[0]

rcov = dfg_rcov.load_inner_data(0).rcov


# Average rate covariances over trials and central lags
R = calc_lag_ROI_symm(rcov, lag_range=(-0.01, 0.01))
R = R.mean(dim='trial_num', skipna=True)

# Nullify diagonal and below
R.values = np.triu(R.values, 1)

# Unroll rcov matrix
Rvals = np.sort(R.values.ravel())
Rvals = Rvals[Rvals != np.nan]

# =============================================================================
# plt.figure()
# plt.plot(Rvals, '.')
# plt.title('rcov values')
# =============================================================================

#print(np.mean(Rvals > 0.5))

# Choose a pair of cells to analyze 
Rvals_sel = Rvals[Rvals > 0.2]
r0 = Rvals_sel[15]
#r0 = Rvals[21]
mask = (R == r0) & (R != np.nan)
R0 = R.where(mask, drop=True)
cell1_name = R0.cell1_name.values[0]
cell2_name = R0.cell2_name.values[0]
print(f'{cell1_name}\n{cell2_name}')

# Load epoched spike trains
cell_names = [cell1_name, cell2_name]
cell_data = {}
for cell_name in cell_names:
    mask = cell_epoched_info.cell_name == cell_name
    fpath_epoched_spikes = cell_epoched_info[mask].fpath_epoched.values[0]
    with open(fpath_epoched_spikes, 'rb') as fid:
        S = pk.load(fid)
    cell_data[cell_name] = {'spikes': S}
    
# Select spikes from a given time ROI
tROI = (0.85, 1.2)
for cell_name in cell_names:
    cell_data[cell_name]['spikes_ROI'] = []
    for spikes in cell_data[cell_name]['spikes']:
        spikes_ = spikes.values.item()
        mask = (spikes_ >= tROI[0]) & (spikes_ <= tROI[1])
        spikes_ROI = spikes_[mask]
        cell_data[cell_name]['spikes_ROI'].append(spikes_ROI)
        
# Stack together spike trains from all trials
Ntrials = len(cell_data[cell1_name]['spikes_ROI'])
Ntrials_used = 100
N = 0
T = tROI[1] - tROI[0]
cell_data[cell1_name]['spikes_stacked'] = []
cell_data[cell2_name]['spikes_stacked'] = []
for n in range(Ntrials):
    spikes1 = cell_data[cell1_name]['spikes_ROI'][n].copy()
    spikes2 = cell_data[cell2_name]['spikes_ROI'][n].copy()
    if (len(spikes1) == 0) and (len(spikes2) == 0):
        continue
    spikes1 += N * T
    spikes2 += N * T
    cell_data[cell1_name]['spikes_stacked'] += spikes1.tolist()
    cell_data[cell2_name]['spikes_stacked'] += spikes2.tolist()
    if N == Ntrials_used:
        break
    N = N + 1

Tmax = max(cell_data[cell1_name]['spikes_stacked'])

# =============================================================================
# plt.figure()
# cols = ['b', 'r']
# for m, cell_name in enumerate(cell_names):
#     spikes = cell_data[cell_name]['spikes_stacked'] 
#     for n in range(len(spikes)):
#         plt.plot([spikes[n], spikes[n]], [0, 1], cols[m])
# plt.plot([0, Tmax], [0, 0], 'k')
# =============================================================================

Trow = 7 * T
Nrows = int(np.ceil(Tmax / Trow))
spike_h = 0.75
row_h = 1
trial_mark_h = 1
cols = ['b', 'r']

tvec_trials = np.arange(Ntrials_used) * T

plt.figure()
for n in range(Nrows):
    t1 = n * Trow
    t2 = (n + 1) * Trow
    h = n * row_h
    # Plot spike train for each cell
    for m, cell_name in enumerate(cell_names):
        spikes = np.array(cell_data[cell_name]['spikes_stacked'])
        mask = (spikes >= t1) & (spikes < t2)
        spikes_row = spikes[mask] - t1
        for spike in spikes_row:
            plt.plot([spike, spike], [h, h + spike_h], cols[m])
    # Plot trial edges
    mask = (tvec_trials >= t1) & (tvec_trials < t2)
    tvec_trials_row = tvec_trials[mask] - t1
    for t in tvec_trials_row:
        plt.plot([t, t], [h, h + trial_mark_h], 'k')
    # Plot horizontal axis
    plt.plot([0, Trow], [h, h], 'k')





