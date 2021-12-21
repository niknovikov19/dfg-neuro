# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

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
import spike_corr as spcor
import vis

import data_file_group as dfg
import roi_utils as roi
import spike_TF_PLV as spPLV
import useful as usf

# Root paths for the data and the processing results
dirpath_root = r'H:\WORK\Camilo'
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
        {'name': 'del12', 'limits': {'time': [0.85, 1.2]}},
        {'name': 'stim', 'limits': {'time': [0.05, 0.3]}},
        {'name': 'bl', 'limits': {'time': [-0.7, -0.2]}},
]
tROIset_name = 'tROIset1'

# Use a subset of channels
Nchan_used = 25
dfg_tf.outer_table = dfg_tf.outer_table[:Nchan_used]

# Calculate PLV
dfg_spPLV = spPLV.calc_dfg_spike_TF_PLV(dfg_tf, cell_epoched_info, tROI_descs,
                                        tROIset_name)

# Save the result
fname_dfg_spPLV = f'dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_tROI_(nchan={Nchan_used})'
fpath_dfg_spPLV = os.path.join(dirpath_proc, fname_dfg_spPLV)
dfg_spPLV.save(fpath_dfg_spPLV)

