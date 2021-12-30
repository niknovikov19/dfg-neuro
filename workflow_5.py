# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import xarray as xr
import pickle as pk

import useful as usf
import trial_manager as trl
import spiketrain_manager as spk
import firing_rate as fr
import lfp
import spike_corr as spcor
import vis

import data_file_group as dfg
import roi_utils as roi
import spike_TF_PLV as spPLV
import useful as usf


# Root paths for the data and the processing results
dirpath_root = r'H:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')


# Run arbitrary function and save the result,
# or load the previously calculated result
def run_or_load(f, fname_cache, recalc=False):    
    fpath_cache = os.path.join(dirpath_proc, fname_cache)    
    if recalc or not os.path.exists(fpath_cache):
        dfg_res = f()
        dfg_res.save(fpath_cache)
    else:
        dfg_res = dfg.DataFileGroup(fpath_cache)
    return dfg_res


# Load Time-Frequency data
fname_dfg_tf = 'dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0)'
fpath_dfg_tf = os.path.join(dirpath_proc, fname_dfg_tf)
dfg_tf = dfg.DataFileGroup(fpath_dfg_tf)

# Load epoched spiketrains
fname_cell_epoched_info = 'cell_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_cell_epoched_info = os.path.join(dirpath_proc, fname_cell_epoched_info)
with open(fpath_cell_epoched_info, 'rb') as fid:
    cell_epoched_info = pk.load(fid)


# Time ROIs to calculate spike-TF PLV
tROI_descs = [
        {'name': 'del1', 'limits': {'time': [0.5, 1.2]}},
        {'name': 'del11', 'limits': {'time': [0.5, 0.85]}},
        {'name': 'del12', 'limits': {'time': [0.85, 1.2]}},
        {'name': 'stim', 'limits': {'time': [0.05, 0.3]}},
        {'name': 'bl', 'limits': {'time': [-0.7, -0.2]}},
]
tROIset_name = 'tROIset1'

# Use a subset of channels
Nchan_used = 25
dfg_tf.outer_table = dfg_tf.outer_table[:Nchan_used]

# Calculate spike-TF PLV
fname_cache = f'dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_tROI_(nchan={Nchan_used})'
f = lambda: spPLV.calc_dfg_spike_TF_PLV(
        dfg_tf, cell_epoched_info, tROI_descs, tROIset_name)
dfg_spPLV_tROI = run_or_load(f, fname_cache, recalc=False)


# Frequency ROIs
fROI_descs = [
        {'name': 'beta', 'limits': {'freq': [15, 40]}},
        {'name': 'gamma', 'limits': {'freq': [60, 100]}}
]

def fun_multiply_PLV_by_Nspikes(X):
    X = X.copy()
    X['PLV'] = X['PLV'] * X['Nspikes']
    return X

def fun_id(X, dims):
    return X

def fun_mean(X, dims):
    return X.mean(dim=dims)

def fun_sum(X, dims):
    return X.sum(dim=dims)

def fun_nROI_mult(X, dims):
    return X * len(fROI_descs)

# Calculate freq ROIs (merge with time ROIs)
# =============================================================================
# ROI_coords = ['freq']
# coords_new_descs = {'ftROI_num': 'Time-frequency ROI'}
# dfg_spPLV_tfROI = roi.calc_data_file_group_ROIs(
#         dfg_spPLV, ROI_coords, fROI_descs,
#         reduce_fun={'PLV': fun_mean, 'Nspikes': fun_sum},
#         ROIset_dim_to_combine=['t'], ROIset_dim_name='fROI',
#         fpath_data_column='fpath_spPLV_tfROI', fpath_data_postfix='tfROI',
#         coords_new_descs=coords_new_descs,
#         preproc_fun=fun_multiply_PLV_by_Ntrials)
# =============================================================================

# Calculate freq ROIs (don't merge with time ROIs)
ROI_coords = ['freq']
coords_new_descs = {'fROI_num': 'Frequency ROI'}
fname_cache = f'dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_tROI_fROI_(nchan={Nchan_used})'
f = lambda: roi.calc_data_file_group_ROIs(
        dfg_spPLV_tROI, ROI_coords, fROI_descs,
        reduce_fun={'PLV': fun_mean, 'Nspikes': fun_id},
        ROIset_dim_to_combine=None, ROIset_dim_name='fROI',
        fpath_data_column='fpath_spPLV_tROI_fROI',
        fpath_data_postfix='tROI_fROI', coords_new_descs=coords_new_descs,
        add_ROIsz_vars=True)
dfg_spPLV_tROI_fROI = run_or_load(f, fname_cache, recalc=False)


# PLV statistics over trials
fname_cache = f'dfg_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_tROI_fROI_pval_(nchan={Nchan_used})'
f = lambda: spPLV.calc_dfg_spPLV_trial_stat(dfg_spPLV_tROI_fROI)
dfg_spPLV_tROI_fROI_pval = run_or_load(f, fname_cache, recalc=False)








