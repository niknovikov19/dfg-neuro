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
                'tROI=del1_(nchan=all_npl)_avgtrials')
fpath_dfg_in = os.path.join(dirpath_proc, fname_dfg_in)
dfg_in = dfg.DataFileGroup(fpath_dfg_in)

Nchan = 3
dfg_in.outer_table = dfg_in.outer_table[:Nchan]

def fun_abs(X):
    return np.abs(X)

def fun_neg(X):
    return -X

proc_info = [
    {'var_name_old': 'PLV', 'proc': fun_abs, 'var_name_new': 'PLVabs',
     'var_desc_new': 'Spike-field coherence (absolute value)'},
    {'var_name_old': 'Nspikes', 'proc': None, 'var_name_new': 'Nspikes',
     'var_desc_new': None},
    {'var_name_old': 'Nspikes_avg', 'proc': None, 'var_name_new': 'Nspikes_avg',
     'var_desc_new': None}
    ]

postfix = 'abs'

dataset_proc = None
#dataset_proc = fun_neg

dfg_out = dfg.dfg_elementwise_proc(dfg_in, proc_info, postfix, dataset_proc)

