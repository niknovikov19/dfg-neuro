# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys

#import matplotlib.pyplot as plt
#import numpy as np
#import pandas as pd
#import xarray as xr
import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
import spike_TF_PLV as spPLV

# Root paths for the data and the processing results
#dirpath_root = r'D:\WORK\Camilo'
#dirpath_data = os.path.join(dirpath_root, 'data')
#dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')


fpath_in = r'D:\WORK\Camilo\TEST\tbl_test\dfg_spPLV_tROI_fROI_pval'
dfg_in = dfg.DataFileGroup(fpath_in)

dfg_in.change_root('H:', 'D:')

dtbl_out = dfg.dfg_to_table(dfg_in)