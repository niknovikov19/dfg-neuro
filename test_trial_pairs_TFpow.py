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

# Data on and low-high-power trial pairs
fname_trial_pairs = ('dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_'
                     'trial_pairs2_(perc_sameROI=0.5r)')
dfg_trial_pairs = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_trial_pairs))

# Time-frequency power
fname_TFpow = 'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)'
dfg_TFpow = dfg.DataFileGroup(os.path.join(dirpath_proc, fname_TFpow))

chan_names = dfg_TFpow.outer_table.chan_name
Nchan = len(chan_names)

diff_difROI_avg_vec = np.ndarray((Nchan), dtype=np.float64)
diff_sameROI_avg_vec = np.ndarray((Nchan), dtype=np.float64)
diff_difROI_avg_vec_m = np.ndarray((Nchan), dtype=np.float64)
diff_sameROI_avg_vec_m = np.ndarray((Nchan), dtype=np.float64)

TFpow_ROIs_lst = []

pbar = tqdm.tqdm(total=Nchan)

freq_ROIs = {
    'beta': (15, 40),
    'gamma': (60, 100)
    }

time_ROIs = {
    'del11': (0.5, 0.85),
    'del12': (0.85, 1.2)
    }

for n, chan_name in enumerate(chan_names):
#for n, chan_name in enumerate([chan_names[0]]):
    
    # Get entries
    outer_coords = {'chan_name': chan_name}
    TFpow_entry = dfg_TFpow.get_table_entries_by_coords(outer_coords)[0]
    trpairs_entry = dfg_trial_pairs.get_table_entries_by_coords(outer_coords)[0]
    
    # Load inner data
    TFpow = dfg_TFpow.load_inner_data(TFpow_entry).TFpow
    X_trpairs = dfg_trial_pairs.load_inner_data(trpairs_entry)
    
    # Trial pairs
    trial_idx_loval = X_trpairs.trial_id_loval.data
    trial_idx_hival = X_trpairs.trial_id_hival.data
    
    # Power difference between pairs of trials
    diff_difROI_vec = X_trpairs.diff_difROI.data
    diff_sameROI_vec = X_trpairs.diff_sameROI.data
    
    # Select trial pairs with large power difference in difROI
    diff_perc = 75
    diff_thresh = np.percentile(diff_difROI_vec, diff_perc)
    mask = (diff_difROI_vec > diff_thresh)
    diff_difROI_vec_m = diff_difROI_vec[mask]
    diff_sameROI_vec_m = diff_sameROI_vec[mask]
    trial_idx_loval_m = trial_idx_loval[mask]
    trial_idx_hival_m = trial_idx_hival[mask]
    
    # Trial-averaged power differences between pairs of cells
    diff_difROI_avg = np.mean(diff_difROI_vec)          # All trials
    diff_sameROI_avg = np.mean(diff_sameROI_vec)
    diff_difROI_avg_vec[n] = diff_difROI_avg
    diff_sameROI_avg_vec[n] = diff_sameROI_avg    
    diff_difROI_avg_m = np.mean(diff_difROI_vec_m)      # Selected trials
    diff_sameROI_avg_m = np.mean(diff_sameROI_vec_m)
    diff_difROI_avg_vec_m[n] = diff_difROI_avg_m
    diff_sameROI_avg_vec_m[n] = diff_sameROI_avg_m
    
    # Calculate TFpow difference between high- and low-power trials
    TFpow_low = usf.xarray_select_xr(TFpow, {'trial_num': trial_idx_loval})
    TFpow_high = usf.xarray_select_xr(TFpow, {'trial_num': trial_idx_hival})
    TFpow_low_m = usf.xarray_select_xr(TFpow, {'trial_num': trial_idx_loval_m})
    TFpow_high_m = usf.xarray_select_xr(TFpow, {'trial_num': trial_idx_hival_m})
    TFpow_avg = TFpow.mean(dim='trial_num', skipna=True)
    TFpow_low_avg_m = TFpow_low_m.mean(dim='trial_num', skipna=True)
    TFpow_high_avg_m = TFpow_high_m.mean(dim='trial_num', skipna=True)
    TFpow_diff_avg_m = np.log(TFpow_high_avg_m) - np.log(TFpow_low_avg_m)
    
    freq_vals = TFpow_low_m.freq.values
    time_vals = TFpow_low_m.time.values
    
    TFpow_ROIs = {'all': {}, 'low': {}, 'high': {}}
    TFpow_ROIs_m = {'all': {}, 'low': {}, 'high': {}}
    TFpow_avg_ROIs_m = {'all': {}, 'low': {}, 'high': {}}
    TFpow_lh = {'all': TFpow, 'low': TFpow_low, 'high': TFpow_high}
    TFpow_lh_m = {'all': TFpow, 'low': TFpow_low_m, 'high': TFpow_high_m}
    TFpow_lh_avg_m = {'all': TFpow_avg, 'low': TFpow_low_avg_m, 'high': TFpow_high_avg_m}
    
    for freq_ROI_name in ('beta', 'gamma'):
        for time_ROI_name in ('del11', 'del12'):
            freq_ROI = freq_ROIs[freq_ROI_name]
            time_ROI = time_ROIs[time_ROI_name]
            freq_mask = (freq_vals >= freq_ROI[0]) & (freq_vals <= freq_ROI[1])            
            time_mask = (time_vals >= time_ROI[0]) & (time_vals <= time_ROI[1])
            tf_sel = {'time': time_vals[time_mask],
                      'freq': freq_vals[freq_mask]}
            tf_ROI_name = f'{time_ROI_name}_{freq_ROI_name}'
            for trial_set in ('all', 'low', 'high'):
                TFpow_ROI = usf.xarray_select_xr(TFpow_lh[trial_set], tf_sel)
                TFpow_ROI = TFpow_ROI.mean(dim='time', skipna=True)
                TFpow_ROI = TFpow_ROI.mean(dim='freq', skipna=True)
                TFpow_ROIs[trial_set][tf_ROI_name] = TFpow_ROI
                TFpow_ROI_m = usf.xarray_select_xr(TFpow_lh_m[trial_set], tf_sel)
                TFpow_ROI_m = TFpow_ROI_m.mean(dim='time', skipna=True)
                TFpow_ROI_m = TFpow_ROI_m.mean(dim='freq', skipna=True)
                TFpow_ROIs_m[trial_set][tf_ROI_name] = TFpow_ROI_m                
                TFpow_avg_ROI_m = usf.xarray_select_xr(TFpow_lh_avg_m[trial_set], tf_sel)
                TFpow_avg_ROI_m = TFpow_avg_ROI_m.mean(dim='time', skipna=True)
                TFpow_avg_ROI_m = TFpow_avg_ROI_m.mean(dim='freq', skipna=True)
                TFpow_avg_ROIs_m[trial_set][tf_ROI_name] = TFpow_avg_ROI_m
                
    TFpow_ROIs_lst.append(TFpow_avg_ROIs_m)
    
    x_same = TFpow_ROIs['all']['del12_beta']
    x_dif = TFpow_ROIs['all']['del11_beta']
    
    x_same_hival = usf.xarray_select_xr(x_same, {'trial_num': trial_idx_hival})
    x_same_loval = usf.xarray_select_xr(x_same, {'trial_num': trial_idx_loval})
    x_dif_hival = usf.xarray_select_xr(x_dif, {'trial_num': trial_idx_hival})
    x_dif_loval = usf.xarray_select_xr(x_dif, {'trial_num': trial_idx_loval})
    
    pbar.update()
    
    f = lambda x: x
    beta_del11_low_vec = f(TFpow_ROIs['low']['del11_beta'])
    beta_del11_high_vec = f(TFpow_ROIs['high']['del11_beta'])
    beta_del12_low_vec = f(TFpow_ROIs['low']['del12_beta'])
    beta_del12_high_vec = f(TFpow_ROIs['high']['del12_beta'])
    beta_del11_low_vec_m = f(TFpow_ROIs_m['low']['del11_beta'])
    beta_del11_high_vec_m = f(TFpow_ROIs_m['high']['del11_beta'])
    beta_del12_low_vec_m = f(TFpow_ROIs_m['low']['del12_beta'])
    beta_del12_high_vec_m = f(TFpow_ROIs_m['high']['del12_beta'])
    
    plt.figure(900)
    plt.clf()
    
    # Trial-averaged high minus low TFpow spectrogram
    plt.subplot(1,2,1)
    tvec = TFpow_diff_avg_m.time.data
    fvec = TFpow_diff_avg_m.freq.data
    rc = (np.min(tvec), np.max(tvec), np.min(fvec), np.max(fvec))
    cmax = np.max(np.abs(TFpow_diff_avg_m.data))
    plt.imshow(TFpow_diff_avg_m.data, origin='lower', vmin=-cmax, vmax=cmax,
               extent=rc, aspect='auto')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title((f'{chan_name}: del11 = {diff_difROI_avg:.02e}, '
               f'del12 = {diff_sameROI_avg:.02e}'))
    plt.colorbar()
    
    # Beta power in high and low trials, in two time ROIs
    plt.subplot(1,2,2)
    #plt.plot(beta_del12_low_vec, beta_del11_low_vec, '.', label='Low', markersize=8)
    #plt.plot(beta_del12_high_vec, beta_del11_high_vec, '.', label='High', markersize=8)
    #plt.plot(beta_del12_low_vec_m, beta_del11_low_vec_m, '.', label='LOW', markersize=12)
    #plt.plot(beta_del12_high_vec_m, beta_del11_high_vec_m, '.', label='HIGH', markersize=12)
    Npts = len(beta_del12_low_vec)
    Npts_m = len(beta_del12_low_vec_m)
    #Npts = 25
    for n in range(Npts):
        plt.plot([beta_del12_low_vec[n], beta_del12_high_vec[n]],
                 [beta_del11_low_vec[n], beta_del11_high_vec[n]], '.-')
    for n in range(Npts_m):
        plt.plot([beta_del12_low_vec_m[n], beta_del12_high_vec_m[n]],
                 [beta_del11_low_vec_m[n], beta_del11_high_vec_m[n]], '.--',
                 markersize=15, linewidth=4)
    xmax = np.max(beta_del11_high_vec)
    plt.plot([0, xmax], [0, xmax], 'k')
    plt.xlabel('del12 (low diff)')
    plt.ylabel('del11 (high diff)')
    plt.legend()
    
# =============================================================================
#     plt.subplot(1,2,2)
#     plt.plot(diff_sameROI_vec, diff_difROI_vec, '.')
#     plt.plot(diff_sameROI_vec_m, diff_difROI_vec_m, '.')
#     plt.plot([0, 0], [np.min(diff_difROI_vec), np.max(diff_difROI_vec)], 'k')
#     plt.xlabel('diff_sameROI')
#     plt.ylabel('diff_difROI')
# =============================================================================
    
    plt.draw()
    #break
    if plt.waitforbuttonpress():
        break

pbar.close()


# =============================================================================
# plt.figure()
# #plt.subplot(1,2,1)
# plt.plot(diff_sameROI_avg_vec, diff_difROI_avg_vec, '.')
# #plt.plot([0, 0], [0, np.max(diff_difROI_avg_vec)], 'k')
# #plt.subplot(1,2,2)
# plt.plot(diff_sameROI_avg_vec_m, diff_difROI_avg_vec_m, '.')
# plt.plot([0, 0], [0, np.max(diff_difROI_avg_vec_m)], 'k')
# 
# plt.figure()
# plt.plot(diff_difROI_avg_vec, diff_difROI_avg_vec_m, '.')
# plt.plot([0, np.max(diff_difROI_avg_vec)], [0, np.max(diff_difROI_avg_vec)], 'k')
# plt.xlabel('All trials')
# plt.ylabel('High-difference trials')
# 
# plt.figure()
# plt.plot(diff_sameROI_avg_vec, diff_sameROI_avg_vec_m, '.')
# plt.plot([0, np.max(diff_sameROI_avg_vec)], [0, np.max(diff_sameROI_avg_vec)], 'k')
# plt.xlabel('All trials')
# plt.ylabel('High-difference trials')
# 
# beta_del11_low_vec = np.array([TFpow_ROIs['low']['del11_beta']
#                                 for TFpow_ROIs in TFpow_ROIs_lst])
# beta_del11_high_vec = np.array([TFpow_ROIs['high']['del11_beta']
#                                 for TFpow_ROIs in TFpow_ROIs_lst])
# beta_del12_low_vec = np.array([TFpow_ROIs['low']['del12_beta']
#                                 for TFpow_ROIs in TFpow_ROIs_lst])
# beta_del12_high_vec = np.array([TFpow_ROIs['high']['del12_beta']
#                                 for TFpow_ROIs in TFpow_ROIs_lst])
# 
# f = lambda x: np.log(x)
# beta_del11_low_vec = f(beta_del11_low_vec)
# beta_del11_high_vec = f(beta_del11_high_vec)
# beta_del12_low_vec = f(beta_del12_low_vec)
# beta_del12_high_vec = f(beta_del12_high_vec)
# 
# plt.figure()
# plt.plot(beta_del12_low_vec, beta_del11_low_vec, '.', label='Low')
# plt.plot(beta_del12_high_vec, beta_del11_high_vec, '.', label='High')
# xmax = np.max(beta_del11_high_vec)
# plt.plot([0, xmax], [0, xmax], 'k')
# plt.xlabel('del12 (low-diff)')
# plt.ylabel('del11 (high-diff)')
# plt.legend()
# 
# beta_del11_diff_vec = beta_del11_high_vec - beta_del11_low_vec
# beta_del12_diff_vec = beta_del12_high_vec - beta_del12_low_vec
# 
# dd1 = ((beta_del11_diff_vec - diff_difROI_avg_vec_m))
# dd2 = ((beta_del12_diff_vec - diff_sameROI_avg_vec_m))
# 
# print(np.max(np.abs(dd1)))
# print(np.max(np.abs(dd2)))
# =============================================================================

#z = np.zeros((Nchan))
#t, p = stat.ttest_rel(diff_sameROI_avg_vec, z, nan_policy='omit')


# TODO:
# 1. Check that beta ROIs in TFpow and trial_pairs match
# 2. Calculate gamma ROIs and make statistics
# 3. Permutations instead of t-test (also in the main analysis!!!)
# +4. Plot diff_difROI and diff_sameROI for each trial in each channel
# 5. Plot each trial pair as a line on the plain beta(del11) - beta(del12)
# 6. Calc and plot trial-averaged diff_difROI and diff_sameROI using mask
# 7. baseline instead of log


