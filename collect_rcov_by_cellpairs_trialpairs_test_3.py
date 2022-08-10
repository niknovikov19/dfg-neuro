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
    
    
#### Parameters
params = {}
#params['cell_pairs_type'] = 'high_rcov' # Use cell pairs with high rate cov. (trial-avg)
params['cell_pairs_type'] = 'high_sel_cov' # Use cell pairs with similar selectivity 
params['all_chan_test'] = False    # Perform analysis with TFpow taken from each channel
params['chan_sel_type'] = 'cell_rec_chan' # TFpow taken from one of the two channels,
                                          # at which the cells of a pair were recorded
params['plot_cell_pair_test'] = False  # Plot statistics (over trials) for each cell pair
params['lag_range'] = (-0.01, 0.01)
params['lag_ROI_type'] = 'symm'
#params['lag_ROI_type'] = 'antisymm_abs'

rcov_threh = 0.5
sel_thresh = 0
difROI_thresh = 0

sess_idx_uni = np.unique(cell_epoched_info.sess_id)

sess_data = {}
for sess_id in sess_idx_uni:
    sess_data[sess_id] = {}

#sys.exit()


def calc_lag_ROI_symm(X, lag_range):
    lags_used = X.lags.values
    mask = (lags_used >= lag_range[0]) & (lags_used <= lag_range[1])
    lags_used = lags_used[mask]
    lagROI = {'lags': lags_used}
    X_lagROI = usf.xarray_select_xr(X, lagROI)
    X_lagROI = X_lagROI.mean(dim='sample_num', skipna=True)
    return X_lagROI

def calc_lag_ROI_antisymm_abs(X, lag_range):
    lags_used = X.lags.values
    mask_left = (lags_used >= lag_range[0]) & (lags_used <= 0)
    mask_right = (lags_used <= lag_range[1]) & (lags_used >= 0)
    lags_used_left = lags_used[mask_left]
    lagROI_left = {'lags': lags_used_left}
    lags_used_right = lags_used[mask_right]    
    lagROI_right = {'lags': lags_used_right}
    X_lagROI_left = usf.xarray_select_xr(X, lagROI_left)
    X_lagROI_left = X_lagROI_left.sum(dim='sample_num', skipna=True)
    X_lagROI_right = usf.xarray_select_xr(X, lagROI_right)
    X_lagROI_right = X_lagROI_right.sum(dim='sample_num', skipna=True)
    X_lagROI = np.abs(X_lagROI_right - X_lagROI_left)
    return X_lagROI

def calc_lag_ROI(X, lag_range, ROI_type):
    if ROI_type == 'symm':
        return calc_lag_ROI_symm(X, lag_range)
    elif ROI_type == 'antisymm_abs':
        return calc_lag_ROI_antisymm_abs(X, lag_range)
    return None


####  Average rate covariance over trials and central lag bins

lag_range = params['lag_range']
lag_ROI_type = params['lag_ROI_type']

# Get rate covariance matrix (cells x cells) for each session
for sess_id in sess_idx_uni:
    
    # Load rate covariance data for a given session
    rcov_entry = dfg_rcov.get_table_entries_by_coords(
        {'sess_id': sess_id})[0]
    rcov_data = dfg_rcov.load_inner_data(rcov_entry)
    rcov = rcov_data.rcov
    
    # Average rate covariance signals over several lag bins around zero
    #rcov_lagROI = calc_lag_ROI(rcov, lag_range, lag_ROI_type)
    rcov_lagROI = calc_lag_ROI(rcov, lag_range, ROI_type='symm')
    
    # Average rate covariances over trials
    rcov_lagROI_avg = rcov_lagROI.mean(dim='trial_num', skipna=True)
    
    sess_data[sess_id]['rcov_mat'] = rcov_lagROI_avg.data
    sess_data[sess_id]['rcov_cell_names'] = rcov.cell1_name.data
    

#### Calculate cell selectivity covariance matrix

# Load firing rates by stimuli
fpath_rates_by_stims = (r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan'
                        r'\tbl_rvec_(ev=stim1_t)_(t=-1.00-3.00)_'
                        '(t=500-1200_dt=10)_bystims_'
                        '(t=0.5-1.2_log=False_thresh=2.5)')
dtbl_rates_by_stims = dfg.DataTable(fpath_rates_by_stims)
tbl_rbystim = dtbl_rates_by_stims.outer_table   

rstim_cols = ['r_stim6', 'r_stim7', 'r_stim8', 'r_stim10', 'r_stim12',
              'r_stim14', 'r_stim15', 'r_stim16']

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
            C[n, m] = np.dot(X[m,:], X[n,:])
    sess_data[sess_id]['sel_cov_mat'] = C
    sess_data[sess_id]['sel_cov_cell_names'] = tbl_rbystim_sess.cell_name.values
    

####  Create a list of cell pairs

def make_pairs(N):
    pairs = []
    for n1 in range(N):
        for n2 in range(n1 + 1, N):
            pairs.append((n1, n2))
    return pairs

# Produce a list of cell pairs with high rcov
for sess_id in sess_idx_uni:
    
    # Get cell pairs with high rate covariance or high selectivity similarity
    if params['cell_pairs_type'] == 'high_rcov':
        C = np.triu(sess_data[sess_id]['rcov_mat'], 1) # nullified diagonal and below
        cell_names = sess_data[sess_id]['rcov_cell_names']    
        idx = np.argwhere(C > rcov_threh)
    elif params['cell_pairs_type'] == 'high_sel_cov':    
        C = sess_data[sess_id]['sel_cov_mat']
        cell_names = sess_data[sess_id]['sel_cov_cell_names']
        idx = np.argwhere(C > sel_thresh * np.nanstd(C))
    
    sess_data[sess_id]['cell_pairs'] = []    
    for ind in idx:
        cell_name_0 = cell_names[ind[0]]
        cell_name_1 = cell_names[ind[1]]
        cell_pair = (cell_name_0, cell_name_1)
        sess_data[sess_id]['cell_pairs'].append(cell_pair)
    

if params['all_chan_test']:
        
    ####  Compare rcov values for the cell pairs from the list between
    # low- and high-beta trials, where beta is taken from various channels
    # The result is a matrix (cell_pairs x channels)
    for sess_id in sess_idx_uni:
            
        dfg_trial_pairs_sess = copy.deepcopy(dfg_trial_pairs)
        mask = (dfg_trial_pairs_sess.outer_table.sess_id == sess_id)
        dfg_trial_pairs_sess.outer_table = dfg_trial_pairs_sess.outer_table[mask]
        trial_pair_entries = dfg_trial_pairs_sess.get_table_entries()
        
        cell_pairs = sess_data[sess_id]['cell_pairs']
        
        # Load rate covariance data for a given session
        rcov_entry = dfg_rcov.get_table_entries_by_coords(
            {'sess_id': sess_id})[0]
        rcov_data = dfg_rcov.load_inner_data(rcov_entry)
        rcov = rcov_data.rcov
        
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
            
            # Load (hival - loval) trial pairs for a given channel
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
                    
                    # Average rate covariance signals over several lag bins around zero
                    rcov_low_TFpow_lagROI = calc_lag_ROI(
                        rcov_low_TFpow, lag_range, lag_ROI_type)
                    rcov_high_TFpow_lagROI = calc_lag_ROI(
                        rcov_high_TFpow, lag_range, lag_ROI_type)

                    x_low = rcov_low_TFpow_lagROI.values
                    x_high = rcov_high_TFpow_lagROI.values
                    x_low_avg = np.nanmean(x_low, axis=0)
                    x_high_avg = np.nanmean(x_high, axis=0)
                    
                    # Compare rate covariance values from loval and hival groups with
                    # paired t-test
                    t, p = stat.ttest_rel(x_low, x_high, nan_policy='omit')  
                    exp['T'][pair_num, chan_num] = t
                    exp['P'][pair_num, chan_num] = p
                    
                    # Difference between covariance values averaged over two groups
                    exp['D'][pair_num, chan_num] = np.nanmean(x_high_avg - x_low_avg)
                
                pbar.update()
                
        pbar.close()
        
        sess_data[sess_id]['exp_data'] = exp_modes    
    
    #### Visualize the results
    
    for sess_id in sess_idx_uni:
        exp_data = sess_data[sess_id]['exp_data']
        var_name_vis = 'T'
        Xnorm = exp_data['normal'][var_name_vis]
        Xmax = np.max(np.abs(Xnorm))
        for exp_name, exp in exp_data.items():    # normal, shuffled
            X = exp[var_name_vis]
            P = exp['P']
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.imshow(X, vmin=-Xmax, vmax=Xmax)
            plt.ylabel('Cell pair')
            plt.title(f'{var_name_vis} ({sess_id}, {exp_name})')
            plt.colorbar()
            plt.subplot(2, 1, 2)
            pthresh = 0.05
            Xsig = X.copy()
            Xsig[P >= pthresh] = np.nan
            S = np.nansum(np.abs(Xsig))
            plt.imshow(Xsig, vmin=-Xmax, vmax=Xmax)
            plt.ylabel('Cell pair')
            plt.xlabel('Channel')
            plt.title(f'{var_name_vis}, p < {pthresh}, S = {S}')
            plt.colorbar()


####  Compare rcov values for the cell pairs from the list between
# low- and high-TFpow trials
for sess_id in sess_idx_uni:
    
    # DataFileGroup containing trial pairs for the channels in a given session
    dfg_trial_pairs_sess = copy.deepcopy(dfg_trial_pairs)
    mask = (dfg_trial_pairs_sess.outer_table.sess_id == sess_id)
    dfg_trial_pairs_sess.outer_table = dfg_trial_pairs_sess.outer_table[mask]
    trial_pair_entries = dfg_trial_pairs_sess.get_table_entries()
    
    cell_pairs = sess_data[sess_id]['cell_pairs']
    
    # Load rate covariance data for a given session
    rcov_entry = dfg_rcov.get_table_entries_by_coords(
        {'sess_id': sess_id})[0]
    rcov_data = dfg_rcov.load_inner_data(rcov_entry)
    rcov = rcov_data.rcov
    
    Npairs = len(cell_pairs)
    E = np.nan * np.ones((Npairs))    
    exp = {}
    exp['P'] = E.copy()     # p-values
    exp['T'] = E.copy()     # t-scores
    exp['D'] = E.copy()     # differences
    
    for pair_num, cell_pair in enumerate(cell_pairs):

        cell1_name, cell2_name = cell_pair

        if params['chan_sel_type'] == 'cell_rec_chan':
            # TFpow is taken from the same channel one of the cells of a pair
            
            # Get two channels, in which the cells were recorded
            chan1_name = usf.get_chan_by_cell(cell1_name, cell_epoched_info,
                                              dfg_trial_pairs_sess.outer_table)
            chan2_name = usf.get_chan_by_cell(cell2_name, cell_epoched_info,
                                              dfg_trial_pairs_sess.outer_table)
            chan_names = [chan1_name, chan2_name]
            
            # Load trial pairs and TFpow difference for each of the two channels
            chan_data = {}
            for chan_name in chan_names:            
                chan_index = dfg_trial_pairs_sess.get_table_entries_by_coords(
                    {'chan_name': chan_name})[0]
                X = dfg_trial_pairs_sess.load_inner_data(chan_index)
                chan_data[chan_name] = {
                    'trial_idx_hival': X.trial_id_hival,
                    'trial_idx_loval': X.trial_id_loval,
                    'TFpow_diffs_difROI': X.diff_difROI,
                    }
                            
            # Find the channel (out of two) with larger TFpow difference
            diff1 = np.mean(chan_data[chan1_name]['TFpow_diffs_difROI'])
            diff2 = np.mean(chan_data[chan2_name]['TFpow_diffs_difROI'])
            if diff1 > diff2:
                chan_name_used = chan1_name
            else:
                chan_name_used = chan2_name
        
        # Low- and high-TFpow trials for the selected channel
        trial_idx_loval = chan_data[chan_name_used]['trial_idx_loval']
        trial_idx_hival = chan_data[chan_name_used]['trial_idx_hival']
        
        # Select trial pairs with large difference in difROI
        perc = difROI_thresh
        diff_difROI_vec = chan_data[chan_name_used]['TFpow_diffs_difROI']
        diff_thresh = np.percentile(diff_difROI_vec, perc)
        mask = (diff_difROI_vec > diff_thresh)
        diff_difROI_vec = diff_difROI_vec[mask]
        trial_idx_loval = trial_idx_loval[mask].values
        trial_idx_hival = trial_idx_hival[mask].values

        # Select rate covariance data for a given cell pair
        ind = {'cell1_name': cell1_name, 'cell2_name': cell2_name}
        rcov_cellpair = usf.xarray_select_xr(rcov, ind)
        
        # Average rate covariance vectors over several lag bins around zero
        rcov_cellpair_avgbins = calc_lag_ROI(
            rcov_cellpair, lag_range, lag_ROI_type)
       
        # Get rate covariance values for low- and high-TFpow trials
        trial_idx_loval_sel = {'trial_num': trial_idx_loval}
        trial_idx_hival_sel = {'trial_num': trial_idx_hival}
        rcov_low_TFpow = usf.xarray_select_xr(rcov_cellpair_avgbins,
                                              trial_idx_loval_sel)
        rcov_high_TFpow = usf.xarray_select_xr(rcov_cellpair_avgbins,
                                              trial_idx_hival_sel)
        x_low = rcov_low_TFpow.values
        x_high = rcov_high_TFpow.values
        
        # Compare rate covariance values from low- and high-TFpow trial groups
        # with paired t-test
        t, p = stat.ttest_rel(x_low, x_high, nan_policy='omit')  
        exp['T'][pair_num] = t
        exp['P'][pair_num] = p
        
        # Difference between covariance values averaged over two groups
        exp['D'][pair_num] = np.nanmean(x_high - x_low)
        
        if params['plot_cell_pair_test']:
            plt.figure(101)
            plt.clf()
            E = np.ones((len(x_low)))
            plt.plot(1 * E, x_low, '.')
            plt.plot(2 * E, x_high, '.')
            plt.legend('Low TFpow', 'High TFpow')
            plt.title(f'{cell1_name} - {cell2_name} (p = {p})')
            plt.ylabel('Rate covariance')
            plt.xlim((0.5, 2.5))
            plt.draw()
            plt.waitforbuttonpress()
    
    sess_data[sess_id]['exp2_data'] = exp
    

d_vec = np.array([], np.float64)
for sess_id in sess_idx_uni:
    d_vec = np.concatenate((d_vec, sess_data[sess_id]['exp2_data']['D']))
#d_vec[np.nanargmax(d_vec)] = np.nan
#d_vec[np.nanargmin(d_vec)] = np.nan
#d_vec[np.nanargmin(d_vec)] = np.nan
z_vec = np.zeros((len(d_vec)))
t, p = stat.ttest_rel(d_vec, z_vec, nan_policy='omit')  

plt.figure()
plt.plot(z_vec, d_vec, '.')
plt.plot([-1, 1], [0, 0], 'k')
plt.ylabel('Cell pair')
plt.title(f'rcov (High TFpow - Low TFpow trials), p = {p}')
    
    