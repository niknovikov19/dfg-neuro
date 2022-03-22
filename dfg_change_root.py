# -*- coding: utf-8 -*-

import os
import pickle as pk

import numpy as np
#import pandas as pd

import data_file_group_2 as dfg


dirpath_in = r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan'
fname_in = 'dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0)'

fpath_in = os.path.join(dirpath_in, fname_in)
dfg_in = dfg.DataFileGroup(fpath_in)

dfg_in.change_root('H:', 'D:')
dfg_in.save(fpath_in)


fname_cell_epoched_info = 'cell_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_cell_epoched_info = os.path.join(dirpath_in, fname_cell_epoched_info)
with open(fpath_cell_epoched_info, 'rb') as fid:
    cell_epoched_info = pk.load(fid)
num_entries = len(cell_epoched_info)
cell_epoched_info.set_index(np.arange(num_entries), inplace=True)
fpath_col_name = 'fpath_epoched'
for entry in range(num_entries):   
    fpath_old = cell_epoched_info.at[entry, fpath_col_name]
    fpath_new = fpath_old.replace('H:', 'D:')
    cell_epoched_info.at[entry, fpath_col_name] = fpath_new
    with open(fpath_cell_epoched_info, 'wb') as fid:
        pk.dump(cell_epoched_info, fid)
