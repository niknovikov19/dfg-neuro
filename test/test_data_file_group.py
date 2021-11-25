# -*- coding: utf-8 -*-

#import importlib
import os
import sys

import numpy as np
from pprint import pprint
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group


fpath_dfg = (r'H:\WORK\Camilo\Processing_Pancake_2sess_allchan' 
              '\dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0)')

dfg = data_file_group.DataFileGroup()
dfg.load(fpath_dfg)

entries = dfg.get_table_entries()

def f(x, a, b):
    print(f'{x} {a} {b}')
    
par = {'a': 2, 'b': 1}

f(10, **par)


