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
import scipy
import xarray as xr
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
# =============================================================================
# tROI_descs = [
#         {'name': 'del1', 'limits': {'time': [0.5, 1.2]}},
#         {'name': 'del11', 'limits': {'time': [0.5, 0.85]}},
#         {'name': 'del12', 'limits': {'time': [0.85, 1.2]}},
#         {'name': 'stim', 'limits': {'time': [0.05, 0.3]}},
#         {'name': 'bl', 'limits': {'time': [-0.7, -0.2]}},
# ]
# tROIset_name = 'tROIset1'
# =============================================================================
tROI_descs = [
        {'name': 'del1', 'limits': {'time': [0.5, 1.2]}}
]
tROIset_name = 'tROI=del1'

# Use a subset of channels
#Nchan_used = 25
Nchan_used = 'all'
dfg_tf_chan_subset = copy.deepcopy(dfg_tf)
if Nchan_used != 'all':
    dfg_tf_chan_subset.outer_table = dfg_tf_chan_subset.outer_table[:Nchan_used]

# Calculate spike-TF PLV
fname_cache_spPLV = (f'dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_'
                     f'{tROIset_name}_(nchan={Nchan_used}_npl)')
f = lambda: spPLV.calc_dfg_spike_TF_PLV(
        dfg_tf_chan_subset, cell_epoched_info, tROI_descs, tROIset_name,
        non_phase_locked=True)
dfg_spPLV_tROI = run_or_load(f, fname_cache_spPLV, recalc=False)


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

def fun_median(X, dims):
    return X.median(dim=dims)

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
fname_cache = (f'dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_'
               f'({tROIset_name})_fROI_(nchan={Nchan_used}_npl)')
f = lambda: roi.calc_data_file_group_ROIs(
        dfg_spPLV_tROI, ROI_coords, fROI_descs,
        reduce_fun={'PLV': fun_mean, 'Nspikes': fun_id},
        ROIset_dim_to_combine=None, ROIset_dim_name='fROI',
        fpath_data_column='fpath_spPLV_tROI_fROI',
        fpath_data_postfix='tROI_fROI', coords_new_descs=coords_new_descs,
        add_ROIsz_vars=True)
dfg_spPLV_tROI_fROI = run_or_load(f, fname_cache, recalc=False)


# PLV statistics over trials
fname_cache = (f'dfg_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_({tROIset_name})_'
               f'fROI_pval_(nchan={Nchan_used}_npl)')
f = lambda: spPLV.calc_dfg_spPLV_trial_stat(dfg_spPLV_tROI_fROI)
dfg_spPLV_tROI_fROI_pval = run_or_load(f, fname_cache, recalc=False)

# PLV statistics over trials -> table
fname_cache = (f'tbl_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_({tROIset_name})_'
               f'fROI_pval_(nchan={Nchan_used}_npl)')
f = lambda: dfg.dfg_to_table(dfg_spPLV_tROI_fROI_pval)
tbl_spPLV_tROI_fROI_pval = run_or_load(f, fname_cache, recalc=False,
                                       data_type='tbl')

# =============================================================================
# # Select cell pairs with significant PLV
# fname_cache = f'tbl_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_cell_pairs_(nchan={Nchan_used}_npl)'
# f = lambda: select_cell_pairs_by_spPLV(tbl_spPLV_tROI_fROI_pval,
#                                        tROI_name='del1', fROI_name='beta',
#                                        pthresh=0.05, rmax=50)
# tbl_signif_spPLV_cell_pairs = run_or_load(f, fname_cache, recalc=False,
#                                           data_type='tbl')
# =============================================================================

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


# Same-dif trial pairs
ROIset_same = {'fROI_name': 'beta', 'tROI_name': 'del12'}
ROIset_dif = {'fROI_name': 'beta', 'tROI_name': 'del11'}
sel_perc_sameROI = 0.5
fname_cache = ('dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_'
               f'trial_pairs2_(perc_sameROI={sel_perc_sameROI}r)')
f = lambda: dfg_find_trial_pairs_by_samedif_tfpow(
        dfg_tfpow_tROI_fROI, ROIset_same, ROIset_dif,
        sel_perc_sameROI=sel_perc_sameROI)
dfg_tfpow_trial_pairs = run_or_load(f, fname_cache, recalc=True)


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


# Average complex-valued spike-LFP PLV over trials
fname_cache_spPLV_avgtrials = fname_cache_spPLV + '_avgtrials'
dim_name = 'trial_num'
coord_names = ['trial_num', 'trial_id']
proc_info = [
    {'var_name_old': 'PLV', 'proc': None, 'var_name_new': 'PLV',
     'var_desc_new': None},
    {'var_name_old': 'Nspikes', 'proc': None, 'var_name_new': 'Nspikes',
     'var_desc_new': 'Total number of spikes'},
    {'var_name_old': 'Nspikes', 'proc': None, 'var_name_new': 'Nspikes_avg',
     'var_desc_new': 'Average number of spikes over trials'}
    ]
postfix = 'avgtrials'
def calc_PLV_avgtrials(X, trial_dim):
    Nspikes = X['Nspikes'].sum(dim=trial_dim)
    Nspikes_avg = X['Nspikes'].mean(dim=trial_dim)
    PLV = (X['PLV'] * X['Nspikes']).sum(dim=trial_dim) / Nspikes
    X_out = {'PLV': PLV, 'Nspikes': Nspikes, 'Nspikes_avg': Nspikes_avg}
    return xr.Dataset(X_out)
f = lambda: dfg.dfg_collapse_dim(dfg_spPLV_tROI, dim_name, coord_names,
                                 proc_info, postfix, calc_PLV_avgtrials)
dfg_spPLV_tROI_avgtrials = run_or_load(f, fname_cache_spPLV_avgtrials,
                                       recalc=False)

def fun_abs(X):
    return np.abs(X)

# Take absolute value of PLV
fname_cache_spPLV_avgtrials_abs = fname_cache_spPLV_avgtrials + '_abs'
proc_info = [
    {'var_name_old': 'PLV', 'proc': fun_abs, 'var_name_new': 'PLVabs',
     'var_desc_new': 'Spike-field coherence (absolute value)'},
    {'var_name_old': 'Nspikes', 'proc': None, 'var_name_new': 'Nspikes',
     'var_desc_new': None},
    {'var_name_old': 'Nspikes_avg', 'proc': None, 'var_name_new': 'Nspikes_avg',
     'var_desc_new': None}
    ]
postfix = 'abs'
f = lambda: dfg.dfg_elementwise_proc(dfg_spPLV_tROI_avgtrials, proc_info,
                                     postfix)
dfg_spPLV_tROI_avgtrials_abs = run_or_load(f, fname_cache_spPLV_avgtrials_abs,
                                           recalc=False)

# Significance of absolute PLV values
fname_cache_spPLV_avgtrials_abs_pval = fname_cache_spPLV_avgtrials_abs + '_pval'
proc_info = [
    {'var_name_old': 'PLVabs', 'proc': None, 'var_name_new': 'PLVabs',
     'var_desc_new': None},
    {'var_name_old': 'Nspikes', 'proc': None, 'var_name_new': 'Nspikes',
     'var_desc_new': None},
    {'var_name_old': 'Nspikes_avg', 'proc': None, 'var_name_new': 'Nspikes_avg',
     'var_desc_new': None},
    {'var_name_old': 'PLVabs', 'proc': None, 'var_name_new': 'pval',
     'var_desc_new': 'P-value of PLVabs'},
    ]
postfix = 'pval' 
def spPLV_stat(X):
   sigma0 = 1 / np.sqrt(2 * X['Nspikes'])
   sigma = sigma0 * np.sqrt(2 - np.pi / 2)
   mu = sigma0 * np.sqrt(np.pi / 2)
   PLV_pval = 1 - scipy.stats.norm.cdf(X['PLVabs']-mu, scale=sigma)
   PLV_pval = xr.DataArray(PLV_pval, dims=X['PLVabs'].dims,
                           coords=X['PLVabs'].coords)
   X_out = {'PLVabs': X['PLVabs'],
            'Nspikes': X['Nspikes'],
            'Nspikes_avg': X['Nspikes_avg'],
            'pval': PLV_pval}
   return xr.Dataset(X_out)
f = lambda: dfg.dfg_elementwise_proc(dfg_spPLV_tROI_avgtrials_abs,
                                     proc_info, postfix, spPLV_stat)
dfg_spPLV_tROI_avgtrials_abs_pval = run_or_load(
    f, fname_cache_spPLV_avgtrials_abs_pval, recalc=False)

# Calculate freq ROIs
fROI_descs_beta = [{'name': 'beta', 'limits': {'freq': [15, 35]}}]
ROI_coords = ['freq']
coords_new_descs = {'fROI_num': 'Frequency ROI'}
reduce_fun={'PLVabs': fun_mean, 'Nspikes': fun_id, 'Nspikes_avg': fun_id,
            'pval': fun_median}
data_desc = dfg_spPLV_tROI_avgtrials_abs_pval.get_data_desc()
fpath_data_column = data_desc['fpath_data_column'] + '_beta'
fname_cache_spPLV_avgtrials_abs_pval_beta = (
    fname_cache_spPLV_avgtrials_abs_pval + '_beta')
f = lambda: roi.calc_data_file_group_ROIs(
        dfg_spPLV_tROI_avgtrials_abs_pval, ROI_coords, fROI_descs_beta,
        reduce_fun=reduce_fun,
        ROIset_dim_to_combine=None, ROIset_dim_name='beta',
        fpath_data_column=fpath_data_column,
        fpath_data_postfix='beta', coords_new_descs=coords_new_descs,
        add_ROIsz_vars=False)
dfg_spPLV_tROI_avgtrials_abs_pval_beta = run_or_load(
    f, fname_cache_spPLV_avgtrials_abs_pval_beta, recalc=False)

# PLVabs_avgtrials_beta to table
fname_cache_spPLV_avgtrials_abs_pval_beta_tbl = (
    fname_cache_spPLV_avgtrials_abs_pval_beta.replace('dfg', 'tbl'))
f = lambda: dfg.dfg_to_table(dfg_spPLV_tROI_avgtrials_abs_pval_beta)
tbl_spPLV_tROI_avgtrials_abs_pval_beta = run_or_load(
    f, fname_cache_spPLV_avgtrials_abs_pval_beta_tbl, recalc=False,
    data_type='tbl')

# Firing rates
fname_cache_rvec = 'dfg_rvec_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)'
def f(): raise NotImplementedError('Rate calculation (dfg) not implemented')
dfg_rvec = run_or_load(f, fname_cache_rvec, recalc=False)

# Trial info
fpath_trial_info = os.path.join(dirpath_proc, 'trial_info')
with open(fpath_trial_info, 'rb') as fid:
    trial_info = pk.load(fid)

import firing_rate_2 as fr2

# Table of mean firing rates by stimulus types
sel_time_win = (0.5, 1.2)
sel_use_log = False
sel_outlier_thresh = 2.5
fname_cache_rates_by_stims = fname_cache_rvec.replace('dfg', 'tbl')
fname_cache_rates_by_stims += (
    f'_bystims_(t={sel_time_win[0]}-{sel_time_win[1]}_log={sel_use_log}_'
    f'thresh={sel_outlier_thresh})')
f = lambda: fr2.calc_dfg_firing_rates_by_stim_types(
                dfg_rvec, trial_info, time_win=sel_time_win,
                use_log=sel_use_log, outlier_thresh=sel_outlier_thresh)
tbl_rates_by_stims = run_or_load(
    f, fname_cache_rates_by_stims, recalc=False, data_type='tbl')


tbl = tbl_spPLV_tROI_avgtrials_abs_pval_beta.outer_table
tbl = tbl[tbl.pval < 0.05]
#idx = np.argsort(tbl.pval).values
#tbl = tbl.iloc[idx]

tbl2 = dfg_spPLV_tROI_avgtrials_abs_pval.outer_table

dirpath_out = r'D:\WORK\Camilo\TEST\spPLV_signif_IMG2'

for n in range(len(tbl)):
    row = tbl.iloc[n]
    chan_name = row.chan_name
    cell_name = row.cell_name
    entry = tbl2[tbl2.chan_name == chan_name].index[0]
    X = dfg_spPLV_tROI_avgtrials_abs_pval.load_inner_data(entry)
    cell_num = np.where(X.cell_name == cell_name)[0][0]
    PLVabs_vec = X.PLVabs[:, 0, cell_num]
    pval_vec = X.pval[:, 0, cell_num]
    PLVabs_vec_signif_1 = PLVabs_vec[pval_vec < 0.05]
    PLVabs_vec_signif_2 = PLVabs_vec[pval_vec < 0.01]
    plt.figure(100)
    plt.clf()
    plt.plot(PLVabs_vec.freq, PLVabs_vec)
    plt.plot(PLVabs_vec_signif_1.freq, PLVabs_vec_signif_1, 'k.')
    plt.plot(PLVabs_vec_signif_2.freq, PLVabs_vec_signif_2, 'r.')
    plt.xlabel('Frequency')
    plt.ylabel('PLVabs')
    plt.title(f'{chan_name}  -  {cell_name}')
    plt.draw()
    fname_out = f'pair_{n}.png'
    fpath_out = os.path.join(dirpath_out, fname_out)
    plt.savefig(fpath_out)
        
    

