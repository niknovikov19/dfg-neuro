# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import copy
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sc
import xarray as xr
import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import useful as usf
import trial_manager as trl
import spiketrain_manager as spk
import firing_rate as fr
import lfp
#import spike_corr as spcor
import vis

import data_file_group_2 as dfg
import roi_utils as roi
import spike_TF_PLV as spPLV
import useful as usf

 
dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

fname_rcov = r'dfg_rcov_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)_(bins=5_iter=50_lags=31)'
fname_trial_pairs = r'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_trial_pairs'

# Data on rate covariance and low-high-power trial pairs
dfg_rcov = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_rcov))
dfg_trial_pairs = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_trial_pairs))

# Channel, session, cell pair
chan_name = 'Pancake_20130923_1_ch99'
session = '20130923_1'
cell1_name = 'Pancake_20130923_1_ch98_c1'
cell2_name = 'Pancake_20130923_1_ch99_c4'

# Load trial pairs for the given channel
trial_pairs_entry = dfg_trial_pairs.get_table_entries_by_coords(
    {'chan_name': chan_name})[0]
trial_pairs_data = dfg_trial_pairs.load_inner_data(trial_pairs_entry)

# Load rate covariance data
rcov_entry = dfg_rcov.get_table_entries_by_coords(
    {'sess_id': session})[0]
rcov_data = dfg_rcov.load_inner_data(rcov_entry)
rcov = rcov_data.rcov

# Select covariance data for a given cell pair
ind = {'cell1_name': cell1_name, 'cell2_name': cell2_name}
rcov_cellpair = usf.xarray_select_xr(rcov, ind)

trial_idx_loval =  {'trial_num': trial_pairs_data.trial_id_loval.values}
trial_idx_hival = {'trial_num': trial_pairs_data.trial_id_hival.values}
rcov_low_TFpow = usf.xarray_select_xr(rcov_cellpair, trial_idx_loval)
rcov_high_TFpow = usf.xarray_select_xr(rcov_cellpair, trial_idx_hival)

# Test
lags_used = rcov_low_TFpow.lags.values
mask = (lags_used >= -0.01) & (lags_used <= 0.01)
lags_used = lags_used[mask]
lagROI = {'lags': lags_used}
rcov_low_TFpow_lagROI = usf.xarray_select_xr(rcov_low_TFpow, lagROI)
rcov_high_TFpow_lagROI = usf.xarray_select_xr(rcov_high_TFpow, lagROI)
#rcov_low_TFpow_lagROI = rcov_low_TFpow_lagROI.sum(dim='sample_num')
#rcov_high_TFpow_lagROI = rcov_high_TFpow_lagROI.sum(dim='sample_num')
x_low = rcov_low_TFpow_lagROI.values
x_high = rcov_high_TFpow_lagROI.values
#x_low = np.abs(x_low)
#x_high = np.abs(x_high)
x_low = np.sum(x_low, axis=0)
x_high = np.sum(x_high, axis=0)
    
t, p = sc.stats.ttest_rel(x_low, x_high, nan_policy='omit')  
print(f'p = {p}')

# =============================================================================
# plt.figure()
# N = len(x_low)
# plt.plot(-1 * np.ones(N), x_low, 'k.')
# plt.plot(1 * np.ones(N), x_high, 'k.')
# plt.xlim((-2, 2))
# =============================================================================


# Test 2
lags_used = rcov_low_TFpow.lags.values
mask = (lags_used >= -0.1) & (lags_used <= 0.1)
lags_used = lags_used[mask]
lagROI = {'lags': lags_used}
rcov_low_TFpow_lagROI = usf.xarray_select_xr(rcov_low_TFpow, lagROI)
rcov_high_TFpow_lagROI = usf.xarray_select_xr(rcov_high_TFpow, lagROI)
x_low = rcov_low_TFpow_lagROI.values
x_high = rcov_high_TFpow_lagROI.values
x_low = np.nanmean(x_low, axis=1)
x_high = np.nanmean(x_high, axis=1)
plt.figure()
plt.plot(lags_used, x_low, '.-')
plt.plot(lags_used, x_high, '.-')
plt.xlabel('Lag, s')
plt.title('Rate covariance')


# Plot rcov matrix
rcov_avg = rcov.mean(dim='trial_num', skipna=True)
lags_used = rcov.lags.values
mask = (lags_used >= -0.001) & (lags_used <= 0.001)
lagROI = {'lags': lags_used[mask]}
rcov_avg = usf.xarray_select_xr(rcov_avg, lagROI)
rcov_avg = rcov_avg.mean(dim='sample_num', skipna=True)
plt.figure()
plt.imshow(rcov_avg, vmin=0, vmax=6)
plt.colorbar()


# Try all channels

session = '20130923_1'
cell1_name = 'Pancake_20130923_1_ch83_c1'
cell2_name = 'Pancake_20130923_1_ch83_c2'

# Load rate covariance data
rcov_entry = dfg_rcov.get_table_entries_by_coords(
    {'sess_id': session})[0]
rcov_data = dfg_rcov.load_inner_data(rcov_entry)
rcov = rcov_data.rcov

# Select covariance data for a given cell pair
ind = {'cell1_name': cell1_name, 'cell2_name': cell2_name}
rcov_cellpair = usf.xarray_select_xr(rcov, ind)

entries = dfg_trial_pairs.get_table_entries()
N = len(entries)

P = np.nan * np.ones(N)
D = np.nan * np.ones(N)

for n, entry in enumerate(entries):
    
    if dfg_trial_pairs.outer_table.iloc[entry].sess_id != session:
        continue
    
    print(f'{entry}')
    
    # Load trial pairs for the given channel
    trial_pairs_data = dfg_trial_pairs.load_inner_data(entry)
    
    trial_idx_loval =  {'trial_num': trial_pairs_data.trial_id_loval.values}
    trial_idx_hival = {'trial_num': trial_pairs_data.trial_id_hival.values}
    rcov_low_TFpow = usf.xarray_select_xr(rcov_cellpair, trial_idx_loval)
    rcov_high_TFpow = usf.xarray_select_xr(rcov_cellpair, trial_idx_hival)
    
    lags_used = rcov_low_TFpow.lags.values
    mask = (lags_used >= -0.01) & (lags_used <= 0.01)
    lags_used = lags_used[mask]
    lagROI = {'lags': lags_used}
    rcov_low_TFpow_lagROI = usf.xarray_select_xr(rcov_low_TFpow, lagROI)
    rcov_high_TFpow_lagROI = usf.xarray_select_xr(rcov_high_TFpow, lagROI)
    x_low = rcov_low_TFpow_lagROI.values
    x_high = rcov_high_TFpow_lagROI.values
    x_low = np.sum(x_low, axis=0)
    x_high = np.sum(x_high, axis=0)
        
    t, p = sc.stats.ttest_rel(x_low, x_high, nan_policy='omit')  
    P[n] = p
    
    D[n] = np.nanmean(x_high - x_low)
    
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(P)
plt.plot([0, len(P)], [0.05, 0.05])
plt.title(f'{cell1_name} - {cell2_name}')
plt.ylabel('P-value)')
plt.xlabel('Channel')
plt.subplot(2, 1, 2)
plt.plot(D)
plt.plot([0, len(D)], [0, 0])
plt.ylabel('High - Low')
plt.xlabel('Channel')



# =============================================================================
# tROI_name = 'del1'
# fROI_name = 'beta'
# pthresh = 0.05
# rmax = 50
# 
# dirpath_root = r'D:\WORK\Camilo'
# dirpath_data = os.path.join(dirpath_root, 'data')
# dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')
# 
# Nchan_used = 25
# fname_in = f'tbl_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_tROI_fROI_pval_(nchan={Nchan_used})'
# fpath_in = os.path.join(dirpath_proc, fname_in)
# 
# tbl_spPLV_pval = dfg.DataTable(fpath_in)
# 
# 
# def select_cell_pairs_by_spPLV(
#         tbl_spPLV_pval: dfg.DataTable, tROI_name, fROI_name, pthresh, rmax):
#     
#     # Input table
#     tbl = tbl_spPLV_pval.outer_table
#     
#     # Select rows corresponding to a given time-frequency ROI,
#     # with significant PLV and the firing rate within a given range
#     mask = ((tbl['PLV_pval'] < pthresh) & (tbl['firing_rate'] < rmax) &
#             (tbl['tROI_name'] == tROI_name) & (tbl['fROI_name'] == fROI_name))
#     tbl = tbl[mask]
#     
#     # Channel-related columns
#     columns_common = ['chan_name', 'subj_name', 'sess_id', 'chan_id', 'fpath_lfp',
#                       'fpath_epoched', 'fpath_tf', 'fpath_spPLV_tROI',
#                       'fpath_spPLV_tROI_fROI', 'fROI_name', 'fROI_name2',
#                       'fROI_num', 'tROI_name', 'tROI_name2', 'tROI_num',
#                       'ROI_sz_fROI']
#     # Cell-related columns
#     columns_cell = ['cell_id', 'cell_name', 'PLV', 'Nspikes', 'PLV_pval',
#                     'firing_rate']
#     
#     #tbl_common = tbl[columns_common]
#     tbl_cell = tbl[columns_cell]
#     
#     # Here we accumulate output rows in a form of a list of dicts
#     rows_out = []
#     
#     for chan in tbl['chan_name'].unique():
#         # Find table rows corresponding to a given channel
#         # and randomly permute them
#         mask = (tbl['chan_name'] == chan)
#         idx = np.nonzero(mask.values)[0]
#         rand_idx = np.random.permutation(idx)
#         
#         # Copy channel-related column values to the output row
#         row_base = tbl.iloc[idx[0]][columns_common].to_dict()
#         
#         # Walk through pairs of rows, extract cell-related column values,
#         # and combine each pair into a single row:
#         # chan-related values + cell1-related values + cell2-related values
#         N = int(len(idx) / 2)
#         for n in range(N):
#             row_new = row_base.copy()
#             for m in range(2):
#                 ind = rand_idx[2 * n + m]
#                 row_cell = tbl_cell.iloc[ind].to_dict()
#                 # Add '_1' or '_2' to the names of cell-related fields
#                 row_cell = {(key + '_' + str(m + 1)): val
#                             for key, val in row_cell.items()}
#                 row_new.update(row_cell)
#             rows_out.append(row_new)
#     
#     # Collect output rows into a table
#     columns_cell_1 = [col + '_1' for col in columns_cell]
#     columns_cell_2 = [col + '_2' for col in columns_cell]        
#     columns_cell12 = list(itertools.chain(*zip(columns_cell_1, columns_cell_2)))
#     columnns_new = columns_common + columns_cell12        
#     tbl_out = pd.DataFrame(rows_out, columns=columnns_new)
#     
#     # Description of the processing step
#     proc_step_name = 'Select pairs of cells with significant PLV with the same channel'
#     proc_step_func = 'select_cell_pairs_by_spPLV()'
#     data_desc_out = {
#         'outer_dims':
#             ['chan_name', 'fROI_num', 'tROI_num', 'cell_id_1', 'cell_id_2'],
#         'outer_coords': {
#             'chan_name': 'Subject + session + channel',
#             'fROI_name': '',
#             'fROI_name2': '',
#             'fROI_num': 'Frequency ROI',
#             'tROI_name': 'Time ROI (name)',
#             'tROI_name2': 'Time ROI (limits)',
#             'tROI_num': 'Time ROI (number)',
#             'cell_id_1': 'Cell 1 number',
#             'cell_name_1': 'Cell 1 name (subject + session + channel)',
#             'cell_id_2': 'Cell 2 number',
#             'cell_name_2': 'Cell 2 name (subject + session + channel)',
#             },
#         'variables': {
#             'ROI_sz_fROI': '',
#             'PLV_1': 'Spike-LFP phase-locking value in a time ROI (cell 1)',
#             'Nspikes_1': 'Number of spikes in a time ROI (cell 1)',
#             'PLV_pval_1': 'P-value of absolute trial-averaged PLV (cell 1)',
#             'firing_rate_1': 'Firing rate (cell 1)',
#             'PLV_2': 'Spike-LFP phase-locking value in a time ROI (cell 2)',
#             'Nspikes_2': 'Number of spikes in a time ROI (cell 2)',
#             'PLV_pval_2': 'P-value of absolute trial-averaged PLV (cell 2)',
#             'firing_rate_2': 'Firing rate (cell 2)'
#             }
#         }
#     proc_params = {
#         'tROI_name': {
#             'desc': 'Name of the time ROI',
#             'value': tROI_name},
#         'fROI_name': {
#             'desc': 'Name of the frequency ROI',
#             'value': fROI_name},
#         'pthresh': {
#             'desc': 'PLV p-value threshold, below which a cell is used',
#             'value': pthresh},
#         'rmax': {
#             'desc': 'Max. firing rate, above which a cell is discarded',
#             'value': rmax}
#         }
#     
#     # Collect the result
#     tbl_res = dfg.DataTable()
#     tbl_res.outer_table = tbl_out
#     tbl_res.data_proc_tree = copy.deepcopy(tbl_spPLV_pval.data_proc_tree)
#     tbl_res.data_proc_tree.add_process_step(
#         proc_step_name, proc_step_func, proc_params, data_desc_out)
#     return tbl_res
# =============================================================================
    

    
    