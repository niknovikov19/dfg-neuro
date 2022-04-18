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

dirpath_out = 'D:\WORK\Camilo\TEST\TF_nonPL_IMG'

for entry in dfg_tf.get_table_entries():
    X = dfg_tf.load_inner_data(entry)
    Y = X.TF
    Y -= np.mean(Y, 2)
    Y = np.abs(Y)
    Y = np.mean(Y, 2)
    plt.figure(100)
    plt.clf()
    plt.imshow(Y, aspect='auto')
    plt.xlabel('Time')
    plt.ylabel('Freq')
    chan_name = X.attrs['outer_coord_vals.chan_name']
    plt.title(chan_name)
    plt.draw()
    plt.savefig(os.path.join(dirpath_out, chan_name + '.png'))


