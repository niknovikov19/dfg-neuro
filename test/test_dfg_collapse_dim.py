# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys
import time
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

# Root paths for the data and the processing results
dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

# Load input data
fname_dfg_in = ('dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_'
                'tROI=del1_(nchan=all_npl)')
fpath_dfg_in = os.path.join(dirpath_proc, fname_dfg_in)
dfg_in = dfg.DataFileGroup(fpath_dfg_in)

Nchan = 3
dfg_in.outer_table = dfg_in.outer_table[:Nchan]

def fun_mean(X, dims):
    return X.mean(dim=dims)

def fun_sum(X, dims):
    return X.sum(dim=dims)

dim_name = 'trial_num'
coord_names = ['trial_num', 'trial_id']
proc_info = [
    {'var_name_old': 'PLV', 'proc': fun_mean, 'var_name_new': 'PLV',
     'var_desc_new': 'Spike-field coherence'},
    {'var_name_old': 'Nspikes', 'proc': fun_sum, 'var_name_new': 'Nspikes',
     'var_desc_new': 'Total number of spikes'},
    {'var_name_old': 'Nspikes', 'proc': fun_mean, 'var_name_new': 'Nspikes_avg',
     'var_desc_new': 'Average number of spikes over trials'}
    ]
postfix = 'avgtrials_test'

dfg_out = dfg.dfg_collapse_dim(dfg_in, dim_name, coord_names, proc_info, postfix)

