# -*- coding: utf-8 -*-
"""Tests for ROI reducing functions on DataFileGroup.

"""

import importlib
import itertools
import os
import sys

import numpy as np
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import useful as usf
import roi_utils as roi
import data_file_group as dfg
import test_utils as test


dirpath_root = r'H:\WORK\Camilo\TEST\tfROI_test'
fname_in = 'TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0).nc'

# TF ROIs
ROIset_name = 'ROIs_beta_gamma'
ROI_descs = [
        {'name': 'beta_del1',
         'limits': {'freq': [15, 40],  'time': [0.5, 1.2]}},
        {'name': 'beta_del11',
         'limits': {'freq': [15, 40],  'time': [0.5, 0.85]}},
        {'name': 'beta_del12',
         'limits': {'freq': [15, 40],  'time': [0.85, 1.2]}},
        {'name': 'gamma_del1',
         'limits': {'freq': [60, 100], 'time': [0.5, 1.2]}},
        {'name': 'gamma_del11',
         'limits': {'freq': [60, 100], 'time': [0.5, 0.85]}},
        {'name': 'gamma_del12',
         'limits': {'freq': [60, 100], 'time': [0.85, 1.2]}},
        {'name': 'beta_stim',
         'limits': {'freq': [15, 40],  'time': [0.05, 0.3]}},
        {'name': 'beta1_stim',
         'limits': {'freq': [15, 25],  'time': [0.05, 0.3]}},
        {'name': 'beta2_stim',
         'limits': {'freq': [25, 40],  'time': [0.05, 0.3]}},
        {'name': 'gamma_stim',
         'limits': {'freq': [60, 100], 'time': [0.05, 0.3]}},
        {'name': 'beta_bl',
         'limits': {'freq': [15, 40],  'time': [-0.7, -0.2]}},
        {'name': 'beta1_bl',
         'limits': {'freq': [15, 25],  'time': [-0.7, -0.2]}},
        {'name': 'beta2_bl',
         'limits': {'freq': [25, 40],  'time': [-0.7, -0.2]}},
        {'name': 'gamma_bl',
         'limits': {'freq': [60, 100], 'time': [-0.7, -0.2]}}
]

# =============================================================================
# TFpow_mode = 'induced'
# fname_out_tfROI = '%s_(ROIset=%s_pow=%s)' % (fname_out_tf, ROIset_name, TFpow_mode)
# tfROI_info = run_or_load(
#         lambda: lfp.calc_TF_ROIs(chan_tf_info, ROI_vec, ROIset_name, TFpow_mode),
#         fname_out_tfROI, recalc=False)
# =============================================================================

# Create a DataFileGroup object
dfg_tf = dfg.DataFileGroup()
initialized = False

for dirname in os.listdir(dirpath_root):
    
    dirpath = os.path.join(dirpath_root, dirname)
    if not os.path.isdir(dirpath):
        continue
    
    # Load dataset
    fpath_in = os.path.join(dirpath, fname_in)
    X = xr.load_dataset(fpath_in, engine='h5netcdf')
    attrs = usf.unflatten_dict(X.attrs)
    
    # Initialize DataFileGroup
    if not initialized:
        proc_steps = attrs['proc_steps']
        dfg_tf.create2(proc_steps)
        initialized = True
        
    # Add the dataset to the DataFileGroup
    outer_coords = attrs['outer_coord_vals']
    dfg_tf.add_entry(outer_coords, X, fpath_in)
    
# Save the DataFileGroup
fpath_out = os.path.join(dirpath_root, 'dfg_TF')
dfg_tf.save(fpath_out)

def reduce_fun(X, dims):
    return (np.abs(X)**2).mean(dim=dims)

# Calculate ROIs    
ROI_coords = list(ROI_descs[0]['limits'].keys())
var_renamings = {'TF': {'name': 'TFpow', 'desc': 'Time-frequency power'}}
coords_new_descs = {'tfROI_num': 'Time-frequency ROI'}
dfg_tfroi = roi.calc_data_file_group_ROIs(
        dfg_tf, ROI_coords, ROI_descs, reduce_fun,
        ROIset_dim_to_combine=None, ROIset_dim_name='tfROI',
        fpath_data_column='fpath_tfROI', fpath_data_postfix='tfROI',
        var_renamings=var_renamings, coords_new_descs=coords_new_descs)

# Load datasets, one by one, and print them into a file
fpath_txt = r'H:\WORK\Camilo\TEST\tfROI_test\tfROI_test.txt'
test.print_dfg(dfg_tfroi, fpath_txt)
    
    



