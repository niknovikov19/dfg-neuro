# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 22:03:10 2021

@author: Nikita
"""

import copy
import os
#import sys

#import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pk
#import scipy as sc
#import scipy.signal as sig
import scipy.stats as stat
from tqdm import tqdm
import xarray as xr

#import pycorrelate as pycor

import data_file_group_2 as dfg
#import spiketrain_manager as spk
import firing_rate as fr
import useful as usf


def calc_dfg_firing_rates_by_stim_types(dfg_rate, trial_info, time_win,
                                      use_log=False, outlier_thresh=None,
                                      dirpath_out_img=None):
    
    if dirpath_out_img is not None:
        plt.figure()
    
    Nsess = len(trial_info)
    
    # Get all stimulus codes
    stim_codes_all = []
    for sess_num in range(Nsess):
        stim_codes_all += trial_info[sess_num]['trial_tbl'].stim1_code.tolist()
    stim_codes_all = [code for code in stim_codes_all if code is not None]
    stim_codes_all_uni = np.unique(stim_codes_all)
    
    # Output variables (will be stored in a table)
    vars_out = {f'r_stim{x}': f'Mean firing rate for stim {x}'
                for x in stim_codes_all_uni}
    vars_out.update({
        'stim_code_rmax': 'Stimulus with maximal firing rate',
        'stim_code_rmin': 'Stimulus with minimal firing rate',
        'pF_anova': 'P-value of selectivity (ANOVA across all stimuli)',
        'pT_minmax': 'P-value from t-test comparing responses to '
                     'stim_code_rmax and stim_code_rmin'
        })
    
    # Description of the processing step
    proc_step_new =  {
        'name': 'Firing rates by stimuli',
        'function': 'calc_dfg_firing_rates_by_stim_types()',
        'params': {
            'time_win': time_win,
            'use_log': use_log,
            'outlier_thresh': outlier_thresh
            },
        'data_desc_out': {
            'variables': vars_out,
            'outer_dims': ['cell_name'],
            'outer_coords': {
                'cell_name': 'Subject + session + channel + cell',
            },
            'fpath_data_column': None,
            'inner_dims': [],
            'inner_coords': []
            }
        }
    
    # Initialize output DataTable object
    dtbl_out = dfg.DataTable()
    rvec_info = dfg_rate.outer_table
    dtbl_out.outer_table = copy.deepcopy(rvec_info)
    dtbl_out._init_outer_indices()
    dtbl_out.data_proc_tree = copy.deepcopy(dfg_rate.data_proc_tree)
    dtbl_out.data_proc_tree.add_process_step(
        proc_step_new['name'],
        proc_step_new['function'],
        proc_step_new['params'],
        proc_step_new['data_desc_out'])
    col_names_new = list(vars_out.keys())
    dtbl_out.outer_table[col_names_new] = np.nan
    
    for sess_num in range(Nsess):
        
        sess = trial_info[sess_num]
        
        # Trials in a session
        trial_tbl = sess['trial_tbl']
        codes = [code for code in trial_tbl.stim1_code if code is not None]
        stim_codes_uni = np.unique(codes)
        Ncodes = len(stim_codes_uni)
        
        # Cells in a session
        cell_mask = (rvec_info.sess_id == sess['sess_id'])
        cell_idx = rvec_info.index[cell_mask]
        Ncells = len(cell_idx)
        
        for cell_num, cell_id in enumerate(cell_idx):
            
            print(f'sess: {sess_num} / {Nsess}  cell: {cell_num} / {Ncells}')
            cell = rvec_info.iloc[cell_id]
            
            # Load firing rate data
            R = dfg_rate.load_inner_data(cell_id)
            
            # Here we will store firing rates for each stimulus code
            RR = []
            RR_avg = np.zeros(Ncodes)
            L = []
            #RR_isnorm = np.zeros(Ncodes)
            
            for code_num, code in enumerate(stim_codes_uni):
                
                # Trials with the current stimulus code
                trial_mask_tbl = (trial_tbl.stim1_code == code)
                trial_idx = trial_tbl[trial_mask_tbl].trial_id
                
                # Get firing rates: average over the time win and select trials 
                trial_mask_R = R.trial_id.isin(trial_idx)
                sample_mask_R = ((R.time >= time_win[0]) &
                                 (R.time <= time_win[1]))
                rvec = R.r[trial_mask_R, sample_mask_R].mean(dim='sample_num').data
                
                # Log
                if use_log:
                    rvec = np.log(rvec)
                
                # Remove outliers
                if outlier_thresh is not None:
                    out_mask = (np.abs(stat.zscore(rvec)) < outlier_thresh)
                    rvec = rvec[out_mask]
                
                RR.append(rvec)
                RR_avg[code_num] = np.nanmean(rvec)
                L.append(len(rvec))
                #st, RR_isnorm[code_num] = stat.normaltest(rvec)
                
            print(L)
                
            # ANOVA on firing rates over the stimulus codes
            F, pF = stat.f_oneway(*RR)
            
            # T-test between the stimuli with the min and max firing rate
            id_min = np.argmin(RR_avg)
            id_max = np.argmax(RR_avg)
            T, pT = stat.ttest_ind(RR[id_min], RR[id_max])
            
            # Visualize the result
            if dirpath_out_img is not None:                
                plt.clf()                
                for code_num in range(Ncodes):
                    plt.plot([stim_codes_uni[code_num]] * len(RR[code_num]),
                             RR[code_num], 'k.')
                plt.xlabel('Stimulus code')
                plt.ylabel('Firing rate')
                plt.title(f'{cell.cell_name}  '
                          f't = ({time_win[0]} - {time_win[1]})  '
                          f'F = {F:.03f}, p = {pF:.08f}')
                fname_out = f'{cell.cell_name}.png'
                fpath_out = os.path.join(dirpath_out_img, fname_out)
                plt.savefig(fpath_out)
                #plt.show()
                #plt.waitforbuttonpress()
                
            # Add the results to the output table
            tbl_out = dtbl_out.outer_table
            row_id = np.where(tbl_out.cell_name == cell.cell_name)[0][0]
            for code_num in range(Ncodes):
                col_name = f'r_stim{stim_codes_uni[code_num]}'
                tbl_out.iloc[row_id, tbl_out.columns.get_loc(col_name)] = (
                    RR_avg[code_num])
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('stim_code_rmax')] = (
                stim_codes_uni[np.nanargmax(RR_avg)])
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('stim_code_rmin')] = (
                stim_codes_uni[np.nanargmin(RR_avg)])
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('pF_anova')] = pF
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('pT_minmax')] = pT
            
    return dtbl_out


# =============================================================================
# fpath_rvec = r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan\dfg_rvec_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)'
# dfg_rate = dfg.DataFileGroup(fpath_rvec)
# 
# fpath_trial_info = r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan\trial_info'
# with open(fpath_trial_info, 'rb') as fid:
#     trial_info = pk.load(fid)
# =============================================================================
            
                
# =============================================================================
# dirpath_rate_img = r'D:\WORK\Camilo\TEST\rate_by_stims'
# 
# tbl_out = calc_dfg_firing_rates_by_stim_types(
#     dfg_rate, trial_info, time_win=(0.5, 1.2), use_log=False,
#     outlier_thresh=2.5, dirpath_out_img=None)
# 
# cols = ['r_stim6', 'r_stim7', 'r_stim8', 'r_stim10', 'r_stim12',
#         'r_stim14', 'r_stim15', 'r_stim16']
# X = np.array(tbl_out.outer_table[cols])
# X = X - np.mean(X, axis=1, keepdims=True)
# 
# X = X[:55, :]
# 
# Ncells = X.shape[0]
# 
# C = np.nan * np.ones((Ncells, Ncells))
# 
# for n in range(Ncells):
#     for m in range(n + 1, Ncells):
#         C[m, n] = np.dot(X[m,:], X[n,:])
# 
# c = C.ravel()
# c = c[~np.isnan(c)]
# 
# idx = np.argwhere(c > (1.5 * np.std(c)))
# 
# plt.figure()
# plt.plot(c, '.')
# plt.plot(idx, c[idx], '.')
# =============================================================================
        
        
    
    
    
    
    

