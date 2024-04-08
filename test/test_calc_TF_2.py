# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
from pathlib import Path
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
dirpath_proc = Path('E:/M1_exp/Proc')

# Load epoched LFP data
fname_dfg_lfp = 'dfg_lfp_epoched.pkl'
fpath_dfg_lfp = os.path.join(dirpath_proc, fname_dfg_lfp)
dfg_in = dfg.DataFileGroup(fpath_dfg_lfp)

# Number of sessions to work with
#nsess_used = 1
#dfg_in.outer_table = dfg_in.outer_table[:nsess_used]

# Calculate TF
win_len = 0.25
win_overlap = 0.25 * 0.9
fmax = 100
dfg_tf = TF.calc_dfg_TF(dfg_in, win_len, win_overlap, fmax,
                        need_recalc=False)

# Save the result
fname_out = ('dfg_TF.pkl')
fpath_out = os.path.join(dirpath_proc, fname_out)
dfg_tf.save(fpath_out)

X = dfg_tf.load_inner_data(0)

# =============================================================================
# W = np.angle(X.TF[:,:,0])
# v = np.mean(np.abs(W[:, 1:] - W[:, :-1]), axis=1)
# 
# plt.figure()
# ext = (X.time[0], X.time[-1], X.freq[0], X.freq[-1])
# #plt.imshow(W, extent=ext, aspect='auto', origin='lower')
# plt.plot(X.freq, v)
# plt.xlabel('Frequency')
# plt.title('Mean phase change')
# 
# S = np.mean(np.abs(X.TF), axis=2)
# 
# plt.figure()
# ext = (X.time[0], X.time[-1], X.freq[0], X.freq[-1])
# plt.imshow(S, extent=ext, aspect='auto', origin='lower')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# =============================================================================

