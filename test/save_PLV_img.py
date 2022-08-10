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
#import xarray as xr
#import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg

# Root paths for the data and the processing results
dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

# Load Time-Frequency data
fname_dfg_in = ('dfg_spPLV_(ev=stim1_t)_(wlen=0.500_wover=0.450_fmax=100.0)_'
                'tROI=del1_(nchan=all_npl)_avgtrials_abs_pval')
fpath_dfg_in = os.path.join(dirpath_proc, fname_dfg_in)
dfg_in = dfg.DataFileGroup(fpath_dfg_in)

dirpath_out = r'D:\WORK\Camilo\TEST\PLV_pval_IMG2'

for entry in dfg_in.get_table_entries():
    print(f'Entry: {entry}')
    X = dfg_in.load_inner_data(entry)
    plt.figure(100)
    plt.clf()
    ext = (float(np.min(X.coords['cell_id'])),
           float(np.max(X.coords['cell_id'])),
           float(np.min(X.coords['freq'])),
           float(np.max(X.coords['freq'])))
    chan_name = X.attrs['outer_coord_vals.chan_name']
    plt.subplot(1, 2, 1)
    plt.imshow(X.pval[:,0,:], aspect='auto', origin='lower', extent=ext,
               vmin=0, vmax=1)
    plt.xlabel('Cell')
    plt.ylabel('Frequency')
    plt.title(f'P-value, {chan_name}')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(X.PLVabs[:,0,:], aspect='auto', origin='lower', extent=ext,
               vmin=0)
    plt.xlabel('Cell')
    plt.title('PLVabs')
    plt.colorbar()
    plt.draw()
    plt.savefig(os.path.join(dirpath_out, chan_name + '.png'))


