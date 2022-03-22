# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

#import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import xarray as xr
import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
import spike_TF_PLV as spPLV

# Root paths for the data and the processing results
dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')


# Load Time-Frequency data
fname_dfg_tf = 'dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0)'
fpath_dfg_tf = os.path.join(dirpath_proc, fname_dfg_tf)
dfg_tf = dfg.DataFileGroup(fpath_dfg_tf)

# Load epoched spiketrains
fname_cell_epoched_info = 'cell_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_cell_epoched_info = os.path.join(dirpath_proc, fname_cell_epoched_info)
with open(fpath_cell_epoched_info, 'rb') as fid:
    cell_epoched_info = pk.load(fid)

# Time ROIs to calculate Spike-TF PLV
tROI_descs = [
        {'name': 'del1', 'limits': {'time': [0.5, 1.2]}},
        {'name': 'del11', 'limits': {'time': [0.5, 0.85]}},
]
tROIset_name = 'tROIset_test'

# Number of channels and cells to work with
Nchan_used = 2 
Ncell_used = 3

# Calculate PLV
dfg_tf.outer_table = dfg_tf.outer_table[:Nchan_used]
dfg_spPLV = spPLV.calc_dfg_spike_TF_PLV(dfg_tf, cell_epoched_info, tROI_descs,
                                        tROIset_name, Ncell_used)

# Save the result
fpath_out = r'D:\WORK\Camilo\TEST\spPLV_test\spPLV_test_2'
dfg_spPLV.save(fpath_out)

# Compare with the old result
fpath_old = r'D:\WORK\Camilo\TEST\spPLV_test\spPLV_test'
dfg_spPLV_old = dfg.DataFileGroup(fpath_old)

for entry in range(len(dfg_spPLV_old.outer_table)):

    fpath_data = dfg_spPLV_old.outer_table.at[entry, 'fpath_spPLV_tROI']
    fpath_data = fpath_data.replace('H:', 'D:')
    dfg_spPLV_old.outer_table.at[entry, 'fpath_spPLV_tROI'] = fpath_data
    
    X_new = dfg_spPLV.load_inner_data(entry)
    X_old = dfg_spPLV_old.load_inner_data(entry)
    
    for var_name in X_new.data_vars:
    
        x_new = X_new[var_name].data 
        x_old = X_old[var_name].data 
        
        b = (x_new != x_old)
        print(np.all(np.isnan(x_new[b])))
        print(np.all(np.isnan(x_old[b])))



