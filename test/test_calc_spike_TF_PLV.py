# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys

#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#import xarray as xr
import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group as dfg
import spike_TF_PLV as spPLV

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
fpath_out = r'H:\WORK\Camilo\TEST\spPLV_test\spPLV_test'
dfg_spPLV.save(fpath_out)


