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
import spike_corr as spcor
import vis

import data_file_group_2 as dfg
import roi_utils as roi
import spike_TF_PLV as spPLV
import useful as usf



def collect_rcov_by_cellpairs_trialpairs(
        dfg_rcov, dfg_trial_pairs, tbl):
    pass


tROI_name = 'del1'
fROI_name = 'beta'
pthresh = 0.05
rmax = 50

dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

Nchan_used = 25
fname_in = f'tbl_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_tROI_fROI_pval_(nchan={Nchan_used})'
fpath_in = os.path.join(dirpath_proc, fname_in)

tbl_spPLV_pval = dfg.DataTable(fpath_in)


def select_cell_pairs_by_spPLV(
        tbl_spPLV_pval: dfg.DataTable, tROI_name, fROI_name, pthresh, rmax):
    
    # Input table
    tbl = tbl_spPLV_pval.outer_table
    
    # Select rows corresponding to a given time-frequency ROI,
    # with significant PLV and the firing rate within a given range
    mask = ((tbl['PLV_pval'] < pthresh) & (tbl['firing_rate'] < rmax) &
            (tbl['tROI_name'] == tROI_name) & (tbl['fROI_name'] == fROI_name))
    tbl = tbl[mask]
    
    # Channel-related columns
    columns_common = ['chan_name', 'subj_name', 'sess_id', 'chan_id', 'fpath_lfp',
                      'fpath_epoched', 'fpath_tf', 'fpath_spPLV_tROI',
                      'fpath_spPLV_tROI_fROI', 'fROI_name', 'fROI_name2',
                      'fROI_num', 'tROI_name', 'tROI_name2', 'tROI_num',
                      'ROI_sz_fROI']
    # Cell-related columns
    columns_cell = ['cell_id', 'cell_name', 'PLV', 'Nspikes', 'PLV_pval',
                    'firing_rate']
    
    #tbl_common = tbl[columns_common]
    tbl_cell = tbl[columns_cell]
    
    # Here we accumulate output rows in a form of a list of dicts
    rows_out = []
    
    for chan in tbl['chan_name'].unique():
        # Find table rows corresponding to a given channel
        # and randomly permute them
        mask = (tbl['chan_name'] == chan)
        idx = np.nonzero(mask.values)[0]
        rand_idx = np.random.permutation(idx)
        
        # Copy channel-related column values to the output row
        row_base = tbl.iloc[idx[0]][columns_common].to_dict()
        
        # Walk through pairs of rows, extract cell-related column values,
        # and combine each pair into a single row:
        # chan-related values + cell1-related values + cell2-related values
        N = int(len(idx) / 2)
        for n in range(N):
            row_new = row_base.copy()
            for m in range(2):
                ind = rand_idx[2 * n + m]
                row_cell = tbl_cell.iloc[ind].to_dict()
                # Add '_1' or '_2' to the names of cell-related fields
                row_cell = {(key + '_' + str(m + 1)): val
                            for key, val in row_cell.items()}
                row_new.update(row_cell)
            rows_out.append(row_new)
    
    # Collect output rows into a table
    columns_cell_1 = [col + '_1' for col in columns_cell]
    columns_cell_2 = [col + '_2' for col in columns_cell]        
    columns_cell12 = list(itertools.chain(*zip(columns_cell_1, columns_cell_2)))
    columnns_new = columns_common + columns_cell12        
    tbl_out = pd.DataFrame(rows_out, columns=columnns_new)
    
    # Description of the processing step
    proc_step_name = 'Select pairs of cells with significant PLV with the same channel'
    proc_step_func = 'select_cell_pairs_by_spPLV()'
    data_desc_out = {
        'outer_dims':
            ['chan_name', 'fROI_num', 'tROI_num', 'cell_id_1', 'cell_id_2'],
        'outer_coords': {
            'chan_name': 'Subject + session + channel',
            'fROI_name': '',
            'fROI_name2': '',
            'fROI_num': 'Frequency ROI',
            'tROI_name': 'Time ROI (name)',
            'tROI_name2': 'Time ROI (limits)',
            'tROI_num': 'Time ROI (number)',
            'cell_id_1': 'Cell 1 number',
            'cell_name_1': 'Cell 1 name (subject + session + channel)',
            'cell_id_2': 'Cell 2 number',
            'cell_name_2': 'Cell 2 name (subject + session + channel)',
            },
        'variables': {
            'ROI_sz_fROI': '',
            'PLV_1': 'Spike-LFP phase-locking value in a time ROI (cell 1)',
            'Nspikes_1': 'Number of spikes in a time ROI (cell 1)',
            'PLV_pval_1': 'P-value of absolute trial-averaged PLV (cell 1)',
            'firing_rate_1': 'Firing rate (cell 1)',
            'PLV_2': 'Spike-LFP phase-locking value in a time ROI (cell 2)',
            'Nspikes_2': 'Number of spikes in a time ROI (cell 2)',
            'PLV_pval_2': 'P-value of absolute trial-averaged PLV (cell 2)',
            'firing_rate_2': 'Firing rate (cell 2)'
            }
        }
    proc_params = {
        'tROI_name': {
            'desc': 'Name of the time ROI',
            'value': tROI_name},
        'fROI_name': {
            'desc': 'Name of the frequency ROI',
            'value': fROI_name},
        'pthresh': {
            'desc': 'PLV p-value threshold, below which a cell is used',
            'value': pthresh},
        'rmax': {
            'desc': 'Max. firing rate, above which a cell is discarded',
            'value': rmax}
        }
    
    # Collect the result
    tbl_res = dfg.DataTable()
    tbl_res.outer_table = tbl_out
    tbl_res.data_proc_tree = copy.deepcopy(tbl_spPLV_pval.data_proc_tree)
    tbl_res.data_proc_tree.add_process_step(
        proc_step_name, proc_step_func, proc_params, data_desc_out)
    return tbl_res
    

    
    