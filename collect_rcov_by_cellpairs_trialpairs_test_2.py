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
#import xarray as xr

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

fname_rcov = r'dfg_rcov_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)_(bins=5_iter=50_lags=31)'
fname_trial_pairs = r'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_trial_pairs'

# Data on rate covariance and low-high-power trial pairs
dfg_rcov = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_rcov))
dfg_trial_pairs = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_trial_pairs))

# Load epoched spiketrains
fname_cell_epoched_info = 'cell_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_cell_epoched_info = os.path.join(dirpath_proc, fname_cell_epoched_info)
with open(fpath_cell_epoched_info, 'rb') as fid:
    cell_epoched_info = pk.load(fid)

session = '20130923_1'

# Load rate covariance data
rcov_entry = dfg_rcov.get_table_entries_by_coords(
    {'sess_id': session})[0]
rcov_data = dfg_rcov.load_inner_data(rcov_entry)
rcov = rcov_data.rcov


# Try all channels and cell pairs

def make_pairs(N):
    pairs = []
    for n1 in range(N):
        for n2 in range(n1 + 1, N):
            pairs.append((n1, n2))
    return pairs

mask_sess = (cell_epoched_info.sess_id == session)
cell_epoched_info_sess = cell_epoched_info[mask_sess]

chan_idx_uni = np.unique(cell_epoched_info_sess.chan_id)

cell_pairs = []

# Produce a list of cell pairs, such that both cells in a pair are recorded
# from the same channel (for a single session)
for chan_id in chan_idx_uni:
    
    mask_chan = (cell_epoched_info_sess.chan_id == chan_id)
    cell_epoched_info_chan = cell_epoched_info_sess[mask_chan]
    
    Ncells = len(cell_epoched_info_chan)
    cell_num_pairs = make_pairs(Ncells)
    
    for cell_num_pair in cell_num_pairs:
        id0, id1 = cell_num_pair
        cell_name_0 = cell_epoched_info_chan.iloc[id0].cell_name
        cell_name_1 = cell_epoched_info_chan.iloc[id1].cell_name
        cell_pair = (cell_name_0, cell_name_1)
        cell_pairs.append(cell_pair)
        
dfg_trial_pairs_sess = copy.deepcopy(dfg_trial_pairs)
mask = (dfg_trial_pairs_sess.outer_table.sess_id == session)
dfg_trial_pairs_sess.outer_table = dfg_trial_pairs_sess.outer_table[mask]
trial_pair_entries = dfg_trial_pairs_sess.get_table_entries()

Nchans = len(trial_pair_entries)
Npairs = len(cell_pairs)

exp_modes = {'normal': {}, 'shuffled': {}}

E = np.nan * np.ones((Npairs, Nchans))
for exp in exp_modes.values():
    exp['P'] = E.copy()     # p-values
    exp['T'] = E.copy()     # t-scores
    exp['D'] = E.copy()     # differences

N = Nchans * Npairs
pbar = tqdm.tqdm(total=N)

for chan_num, entry in enumerate(trial_pair_entries):
    
    # Load (hival - loval) trial pairs for the given channel
    trial_pairs_data = dfg_trial_pairs_sess.load_inner_data(entry)
    
    trial_idx_loval = trial_pairs_data.trial_id_loval.values
    trial_idx_hival = trial_pairs_data.trial_id_hival.values
    
    diff_difROI_vec = trial_pairs_data.diff_difROI.values
    
    # Select trial pairs with large difference in difROI
    perc = 75
    diff_thresh = np.percentile(diff_difROI_vec, perc)
    mask = (diff_difROI_vec > diff_thresh)
    diff_difROI_vec = diff_difROI_vec[mask]
    trial_idx_loval = trial_idx_loval[mask]
    trial_idx_hival = trial_idx_hival[mask]
    
    # Randomly shuffle trial_idx_loval and trial_idx_hival
    trial_idx_loval_sh = trial_idx_loval.copy()
    trial_idx_hival_sh= trial_idx_hival.copy()
    Ntrials = len(trial_idx_loval)
    mask = np.random.uniform(size=Ntrials) < 0.5
    #print(np.mean(mask))
    trial_idx_loval_sh[mask] = trial_idx_hival[mask]
    trial_idx_hival_sh[mask] = trial_idx_loval[mask]
    
    exp_modes['normal']['trial_idx_loval'] = trial_idx_loval
    exp_modes['normal']['trial_idx_hival'] = trial_idx_hival
    exp_modes['shuffled']['trial_idx_loval'] = trial_idx_loval_sh
    exp_modes['shuffled']['trial_idx_hival'] = trial_idx_hival_sh
    
    for pair_num, cell_pair in enumerate(cell_pairs):

        cell1_name, cell2_name = cell_pair
        
        # Select covariance data for a given cell pair
        ind = {'cell1_name': cell1_name, 'cell2_name': cell2_name}
        rcov_cellpair = usf.xarray_select_xr(rcov, ind)
        
        for exp_name, exp in exp_modes.items():
        
            # Get rate covariance values for hival and loval trials
            trial_idx_loval_sel = {'trial_num': exp['trial_idx_loval']}
            trial_idx_hival_sel = {'trial_num': exp['trial_idx_hival']}
            rcov_low_TFpow = usf.xarray_select_xr(rcov_cellpair,
                                                  trial_idx_loval_sel)
            rcov_high_TFpow = usf.xarray_select_xr(rcov_cellpair,
                                                  trial_idx_hival_sel)
            
            # Summate rate covariance signals over several lag bins around zero
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
            
            # Compare rate covariance values from loval and hival groups with
            # pairedd t-test
            t, p = stat.ttest_rel(x_low, x_high, nan_policy='omit')  
            exp['T'][pair_num, chan_num] = t
            exp['P'][pair_num, chan_num] = p
            
            # Difference between covariance values averaged over two groups
            exp['D'][pair_num, chan_num] = np.nanmean(x_high - x_low)
        
        pbar.update()
        
pbar.close()

# =============================================================================
# plt.figure()
# plt.subplot(2, 1, 1)
# Dmax = np.max(np.abs(D))
# plt.imshow(D, vmin=-Dmax, vmax=Dmax)
# plt.ylabel('Cell pair')
# plt.title('Effect')
# plt.colorbar()
# plt.subplot(2, 1, 2)
# plt.imshow(P, vmin=0, vmax=1)
# plt.ylabel('Cell pair')
# plt.xlabel('Channel')
# plt.title('P-value')
# plt.colorbar()
# =============================================================================

for exp_name, exp in exp_modes.items():    
    T = exp['T']
    P = exp['P']
    plt.figure()
    plt.subplot(2, 1, 1)
    Tmax = np.max(np.abs(T))
    plt.imshow(T, vmin=-Tmax, vmax=Tmax)
    plt.ylabel('Cell pair')
    plt.title(f'T-score ({exp_name})')
    plt.colorbar()
    plt.subplot(2, 1, 2)
    pthresh = 0.01
    Tsig = T.copy()
    Tsig[P >= pthresh] = np.nan
    plt.imshow(Tsig, vmin=-Tmax, vmax=Tmax)
    plt.ylabel('Cell pair')
    plt.xlabel('Channel')
    plt.title(f'T-score, p < {pthresh}')
    plt.colorbar()

# =============================================================================
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.plot(P)
# plt.plot([0, len(P)], [0.05, 0.05])
# plt.title(f'{cell1_name} - {cell2_name}')
# plt.ylabel('P-value)')
# plt.xlabel('Channel')
# plt.subplot(2, 1, 2)
# plt.plot(D)
# plt.plot([0, len(D)], [0, 0])
# plt.ylabel('High - Low')
# plt.xlabel('Channel')
# =============================================================================



# Load firing rates by stimuli
fpath_rates_by_stims = (r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan'
                        r'\tbl_rvec_(ev=stim1_t)_(t=-1.00-3.00)_'
                        '(t=500-1200_dt=10)_bystims_'
                        '(t=0.5-1.2_log=False_thresh=2.5)')
dtbl_rates_by_stims = dfg.DataTable(fpath_rates_by_stims)
tbl_rbystim = dtbl_rates_by_stims.outer_table   

sess_idx_uni = np.unique(tbl_rbystim.sess_id)

rstim_cols = ['r_stim6', 'r_stim7', 'r_stim8', 'r_stim10', 'r_stim12',
              'r_stim14', 'r_stim15', 'r_stim16']

sel_cov_mats = {}

# Calculate selectivity covaiance matrix (cells x cells) for each session
for sess_id in sess_idx_uni:
    
    # Get firing rates by stimuli as a matrix (cells x stims) for a current
    # session, subtract the mean over the stimuli
    tbl_rbystim_sess = tbl_rbystim[tbl_rbystim.sess_id == sess_id]    
    X = np.array(tbl_rbystim_sess[rstim_cols])
    X = X - np.mean(X, axis=1, keepdims=True)
    
    # Selectivity covariance values  for cell pairs: dot products of zero-mean
    # firing rate vectors (1 x stims)
    Ncells = X.shape[0]
    C = np.nan * np.ones((Ncells, Ncells))
    for n in range(Ncells):
        for m in range(n + 1, Ncells):
            C[m, n] = np.dot(X[m,:], X[n,:])
    sel_cov_mats[sess_id] = C

# =============================================================================
#     c = C.ravel()
#     c = c[~np.isnan(c)]
#     sigma = np.std(c)
#     
#     cell_id_pairs = np.argwhere(C > 1.5 * sigma)
#     
#     cvals = np.array([C[tuple(idd)] for idd in cell_id_pairs])
#     idx_sorted = np.flip(np.argsort(cvals))
#     cvals = cvals[idx_sorted]
#     cell_id_pairs = cell_id_pairs[idx_sorted]
#     
#     for cell_id_pair in cell_id_pairs:
#         cell_name_1 = tbl_rbystim.cell_name[cell_id_pair[0]]
#         cell_name_2 = tbl_rbystim.cell_name[cell_id_pair[1]]
#         print(f'{cell_name_1} - {cell_name_2}')
# =============================================================================
    
#plt.figure()
#plt.plot(c, '.')
#plt.plot(idx, c[idx], '.')

lag_range = (-0.001, 0.001)

rcov_mats = {}
rcov_cell_names = {}

# Get rate covariance matrix (cells x cells) for each session
for sess_id in sess_idx_uni:
    
    # Load rate covariance data for a given session
    rcov_entry = dfg_rcov.get_table_entries_by_coords(
        {'sess_id': sess_id})[0]
    rcov_data = dfg_rcov.load_inner_data(rcov_entry)
    rcov = rcov_data.rcov
    rcov_cell_names[sess_id] = rcov.cell1_name.data
    
    # Average rate covariance signals over several lag bins around zero
    lags_used = rcov.lags.values
    mask = (lags_used >= lag_range[0]) & (lags_used <= lag_range[1])
    lags_used = lags_used[mask]
    lagROI = {'lags': lags_used}
    rcov_lagROI = usf.xarray_select_xr(rcov, lagROI)
    rcov_lagROI = rcov_lagROI.mean(dim='sample_num', skipna=True)
    
    # Average rate covariances over trials
    rcov_lagROI_avg = rcov_lagROI.mean(dim='trial_num', skipna=True)
    
    rcov_mats[sess_id] = rcov_lagROI_avg.data
    

# Selectivity covariance vs. rate covariance

#sess_id = '20130923_1'

c_rcov = np.array([],  dtype=np.float64)
c_sel_cov = np.array([],  dtype=np.float64)

c_rcov_high = np.array([],  dtype=np.float64)

for sess_id in sess_idx_uni:

    c_rcov = np.append(c_rcov, rcov_mats[sess_id].T.ravel())
    c_sel_cov = np.append(c_sel_cov, sel_cov_mats[sess_id].ravel())

plt.figure()
plt.plot(c_sel_cov, c_rcov, '.')
plt.plot([np.nanmin(c_sel_cov), np.nanmax(c_sel_cov)], [0, 0], 'k')
plt.xlabel('Selectivity covariance')
plt.ylabel('Rate covariance')


#cell_num_pairs = make_pairs(Ncells)

sess_id = '20130923_1'

C_rcov = rcov_mats[sess_id].T
C_sel_cov = sel_cov_mats[sess_id]
cell_names = rcov_cell_names[sess_id]
mask = (C_rcov > 0.5) & ~np.isnan(C_sel_cov)
idx = np.argwhere(mask)
for ind in idx:
    cell_name_1 = cell_names[ind[0]]
    cell_name_2 = cell_names[ind[1]]
    rcov_val = C_rcov[ind[0], ind[1]]
    print(f'{cell_name_1} - {cell_name_2}: rcov = {rcov_val}')


# =============================================================================
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(rcov_mats[sess_id])
# plt.subplot(1,2,2)
# plt.imshow(sel_cov_mats[sess_id])
#     
# =============================================================================
