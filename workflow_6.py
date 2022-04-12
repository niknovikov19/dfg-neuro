# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import copy
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
#import spike_corr as spcor
import vis

import data_file_group_2 as dfg
from find_trial_pairs_by_samedif_TF_tfROI import dfg_find_trial_pairs_by_samedif_tfpow
import roi_utils as roi
from select_cell_pairs_by_spPLV import select_cell_pairs_by_spPLV
import spike_TF_PLV as spPLV
import rvec_cov as rcov

# Root paths for the data and the processing results
dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

root_change = ['H:', 'D:']

# Run arbitrary function and save the result,
# or load the previously calculated result
def run_or_load(f, fname_cache, recalc=False, data_type='dfg'):    
    fpath_cache = os.path.join(dirpath_proc, fname_cache)
    print('--------------------------')
    print(fname_cache)
    if recalc or not os.path.exists(fpath_cache):
        data_res = f()
        data_res.save(fpath_cache)
    else:
        if data_type == 'dfg':
            data_res = dfg.DataFileGroup(fpath_cache)
        elif data_type == 'tbl':
            data_res = dfg.DataTable(fpath_cache)
        else:
            raise ValueError('Unexpected data type')
    if data_type == 'dfg':
        data_res.change_root(root_change[0], root_change[1])
    return data_res


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
dfg_tf_chan_subset = copy.deepcopy(dfg_tf)
dfg_tf_chan_subset.outer_table = dfg_tf_chan_subset.outer_table[:Nchan_used]

# Calculate spike-TF PLV
fname_cache = f'dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_tROI_(nchan={Nchan_used})'
f = lambda: spPLV.calc_dfg_spike_TF_PLV(
        dfg_tf_chan_subset, cell_epoched_info, tROI_descs, tROIset_name)
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

# PLV statistics over trials -> table
fname_cache = f'tbl_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_tROI_fROI_pval_(nchan={Nchan_used})'
f = lambda: dfg.dfg_to_table(dfg_spPLV_tROI_fROI_pval)
tbl_spPLV_tROI_fROI_pval = run_or_load(f, fname_cache, recalc=False,
                                       data_type='tbl')

# Select cell pairs with significant PLV
fname_cache = f'tbl_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_cell_pairs_(nchan={Nchan_used})'
f = lambda: select_cell_pairs_by_spPLV(tbl_spPLV_tROI_fROI_pval,
                                       tROI_name='del1', fROI_name='beta',
                                       pthresh=0.05, rmax=50)
tbl_signif_spPLV_cell_pairs = run_or_load(f, fname_cache, recalc=False,
                                          data_type='tbl')


# TF complex amplitude -> TF power (non-phase-locked)
fname_cache = 'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)'
f = lambda: lfp.calc_dfg_TFpow(dfg_tf, subtract_mean=True)
dfg_tfpow = run_or_load(f, fname_cache, recalc=False)

# TFpow time ROIs
fname_cache = 'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_tROI'
f = lambda: roi.calc_data_file_group_ROIs(
        dfg_tfpow, ROI_coords=['time'], ROI_descs=tROI_descs,
        reduce_fun=fun_mean, ROIset_dim_name='tROI',
        fpath_data_column='fpath_TFpow_tROI', fpath_data_postfix='tROI',
        coords_new_descs={'tROI_num': 'Time ROI'}, add_ROIsz_vars=True)
dfg_tfpow_tROI = run_or_load(f, fname_cache, recalc=False)

# TFpow time + frequency ROIs
fname_cache = 'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_tROI_fROI'
f = lambda: roi.calc_data_file_group_ROIs(
        dfg_tfpow_tROI, ROI_coords=['freq'], ROI_descs=fROI_descs,
        reduce_fun=fun_mean, ROIset_dim_name='fROI',
        fpath_data_column='fpath_TFpow_tROI_fROI', fpath_data_postfix='fROI',
        coords_new_descs={'fROI_num': 'Frequency ROI'}, add_ROIsz_vars=True)
dfg_tfpow_tROI_fROI = run_or_load(f, fname_cache, recalc=False)


# Firing rates
fname_cache = 'dfg_rvec_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)'
def f(): raise NotImplementedError('Rate calculation (dfg) not implemented')
dfg_rvec = run_or_load(f, fname_cache, recalc=False)

# Firing rate covariance
nbins_jit = 5
niter_jit = 50
lag_range = (-15, 15)
time_range = (0.85, 1.2)
Nlags = lag_range[1] - lag_range[0] + 1
fname_cache = ('dfg_rcov_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)_' +
               f'(bins={nbins_jit}_iter={niter_jit}_nlags={Nlags}_' +
               f't={time_range[0]}-{time_range[1]})')
f = lambda: rcov.calc_dfg_rvec_cov_nojit(
        dfg_rvec, nbins_jit, niter_jit, time_range, lag_range)
dfg_rcov = run_or_load(f, fname_cache, recalc=True)


# Same-dif trial pairs
ROIset_same = {'fROI_name': 'beta', 'tROI_name': 'del12'}
ROIset_dif = {'fROI_name': 'beta', 'tROI_name': 'del11'}
fname_cache = 'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_trial_pairs'
f = lambda: dfg_find_trial_pairs_by_samedif_tfpow(
        dfg_tfpow_tROI_fROI, ROIset_same, ROIset_dif)
dfg_tfpow_trial_pairs = run_or_load(f, fname_cache, recalc=False)


