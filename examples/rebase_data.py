# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import xarray as xr
import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
import TF
import useful as usf


dirpath_base_old = r'D:\WORK\Camilo\data'
dirpath_base_new = r'D:\WORK\Camilo\data_copy'

# Copy TF data
fpath_dfg_in = (r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan' 
                r'\dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0)')
dfg_tf = dfg.DataFileGroup(fpath_dfg_in)
tf_files = dfg_tf.outer_table.fpath_tf.tolist()
usf.copy_files(tf_files, dirpath_base_old, dirpath_base_new)

# Copy epoched spiketrains
fpath_cell_epoched_info = (r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan'
                           r'\cell_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)')
with open(fpath_cell_epoched_info, 'rb') as fid:
    cell_epoched_info = pk.load(fid)    
spike_files = cell_epoched_info.fpath_epoched.tolist()
usf.copy_files(spike_files, dirpath_base_old, dirpath_base_new)
