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

# Root paths for the data and the processing results
dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

# Load epoched LFP data
fname_dfg_lfp = 'dfg_LFP_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_dfg_lfp = os.path.join(dirpath_proc, fname_dfg_lfp)
dfg_in = dfg.DataFileGroup(fpath_dfg_lfp)

# Number of channels to work with
Nchan_used = 2
dfg_in.outer_table = dfg_in.outer_table[:Nchan_used]

# Calculate TF
win_len = 0.5
win_overlap = 0.475
fmax = 100
dfg_tf = TF.calc_dfg_TF(dfg_in, win_len, win_overlap, fmax,
                        need_recalc=False)

# Save the result
fname_out = ('dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_'
             f'(wlen={win_len}_wover={win_overlap}_fmax={fmax})_'
             f'(nchan={Nchan_used})')
fpath_out = os.path.join(dirpath_proc, fname_out)
dfg_tf.save(fpath_out)

X = dfg_tf.load_inner_data(1)

W = np.angle(X.TF[:,:,0])
v = np.mean(np.abs(W[:, 1:] - W[:, :-1]), axis=1)

plt.figure()
ext = (X.time[0], X.time[-1], X.freq[0], X.freq[-1])
#plt.imshow(W, extent=ext, aspect='auto', origin='lower')
plt.plot(X.freq, v)
plt.xlabel('Frequency')
plt.title('Mean phase change')

S = np.mean(np.abs(X.TF), axis=2)

plt.figure()
ext = (X.time[0], X.time[-1], X.freq[0], X.freq[-1])
plt.imshow(S, extent=ext, aspect='auto', origin='lower')
plt.xlabel('Time')
plt.ylabel('Frequency')

