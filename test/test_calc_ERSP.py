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
import ersp

# Root paths for the data and the processing results
dirpath_proc = Path('E:/M1_exp/Proc')

# Load epoched LFP data
#fpath_dfg_lfp = dirpath_proc / 'dfg_TF_test.pkl'
fpath_dfg_lfp = dirpath_proc / 'dfg_TF.pkl'
dfg_in = dfg.DataFileGroup(fpath_dfg_lfp)

# Number of sessions to work with
#nsess_used = 1
#dfg_in.outer_table = dfg_in.outer_table[:nsess_used]
# Session to work with
# =============================================================================
# sess = dfg_in.get_table_entry_by_coords({'session': '230517_2759_1315VAL'})
# dfg_in.outer_table = dfg_in.outer_table[sess : sess + 1]
# dfg_in.outer_table.reset_index(drop=True, inplace=True)
# =============================================================================

# Calculate TF
baseline = (-3.5, -2.5)
time_win = (-1, 2)
dfg_out = ersp.calc_dfg_ersp(dfg_in, baseline, time_win=time_win, need_recalc=False )

# Save the result
fpath_out = dirpath_proc / 'dfg_ERSP_test.pkl'
dfg_out.save(fpath_out)

X = dfg_out.load_inner_data(0)

S = X.ERSP.sel(chan=130).mean(dim='trial')
plt.figure()
ext = (X.time[0], X.time[-1], X.freq[0], X.freq[-1])
plt.imshow(S, extent=ext, aspect='auto', origin='lower', vmin=-1.5, vmax=2)
plt.xlabel('Time')
plt.ylabel('Frequency')
#plt.xlim((0, 1))
plt.colorbar()
