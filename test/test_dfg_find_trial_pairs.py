# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
from find_trial_pairs_by_samedif_TF_tfROI import _dfg_find_trial_pairs_by_samedif_tfpow_inner
import useful as usf


dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

fname_TFpow = 'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_tROI_fROI'
fpath_TFpow = os.path.join(dirpath_proc, fname_TFpow)
dfg_TFpow = dfg.DataFileGroup(fpath_TFpow)
dfg_TFpow.change_root('H:', 'D:')

X = dfg_TFpow.load_inner_data(1)

ROIset_same = {'fROI_name': 'beta', 'tROI_name': 'del12'}
ROIset_dif = {'fROI_name': 'beta', 'tROI_name': 'del11'}
Y = _dfg_find_trial_pairs_by_samedif_tfpow_inner(X, ROIset_same, ROIset_dif)


fname_dfg = 'dfg_TFpow_noERP_(ev=stim1_t)_(t=-1.00-3.00)_(TF_0.5_0.4_100)_trial_pairs'
fpath_dfg = os.path.join(dirpath_proc, fname_dfg)

dfg_trpairs = dfg.DataFileGroup(fpath_dfg)

for entry in dfg_trpairs.get_table_entries():
    
    X = dfg_trpairs.load_inner_data(entry)
    print(entry)

    plt.figure(100)
    plt.clf()
    x = X.diff_sameROI.data
    y = X.diff_difROI.data
    plt.plot(x, y, '.')
    plt.xlabel(str(ROIset_same))
    plt.ylabel(str(ROIset_dif))
    plt.title(f'TF power difference in trial pairs (entry = {entry})')
    plt.draw()
    plt.ylim((0, max(y)))
    plt.xlim((-1e-6, 1e-6))
    if not plt.waitforbuttonpress():
        break


