# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 18:43:56 2024

@author: Nikita
"""

from pathlib import Path
import sys
from time import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
#warnings.simplefilter(action='ignore', category=FutureWarning)

#import matplotlib.pyplot as plt
#import xarray as xr

# Add dfg folder to path
parent_dir = Path(f'{__file__}/../../..')
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

#import data_file_group_2 as dfg
import dfg
#import tf_power as tfpow

# Root paths for the data and the processing results
#dirpath_proc = Path(r'D:\WORK\Salvador\M1_project\M1_thal_analysis\data\nikita\dfg_test')
dirpath_proc = Path(r'E:\M1_exp\Proc')

# Load complex TF data
fpath_dfg_in = dirpath_proc / 'dfg_TF_(wlen=0.25_wover=0.225_fmax=200).pkl'
dfg_in = dfg.DataFileGroup(fpath_dfg_in)

# Change root for testing (if needed)
#dfg_in.change_root(r'E:\M1_exp\Proc', str(dirpath_proc))

# Number of sessions to work with
nsess_used = 2
dfg_in.outer_table = dfg_in.outer_table[:nsess_used]
# Session to work with
# =============================================================================
# sess = dfg_in.get_table_entry_by_coords({'session': '230517_2759_1315VAL'})
# dfg_in.outer_table = dfg_in.outer_table[sess : sess + 1]
# dfg_in.outer_table.reset_index(drop=True, inplace=True)
# =============================================================================

# Calculate TF power
t0 = time()
dfg_out = dfg.tfpow.calc_dfg_tfpow(dfg_in, need_recalc=True)
print(f'dt = {time() - t0}')

# Save the result
fpath_out = dirpath_proc / 'dfg_TFpow_test.pkl'
dfg_out.save(fpath_out)

#X = dfg_out.load_inner_data(0)

# =============================================================================
# S = X.ERSP.sel(chan=130).mean(dim='trial')
# plt.figure()
# ext = (X.time[0], X.time[-1], X.freq[0], X.freq[-1])
# plt.imshow(S, extent=ext, aspect='auto', origin='lower', vmin=-1.5, vmax=2)
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# #plt.xlim((0, 1))
# plt.colorbar()
# =============================================================================
