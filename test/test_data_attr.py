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

from data_file_group import DataProcTree
import useful as usf


# Create an arbitrary Dataset with empty attrs

dims = ['dim0', 'dim1']
sz = (3, 2)
coords = {
        'dim0':     np.arange(sz[0]),    # 0, 1, 2
        'dim1':     np.arange(sz[1]),    # 0, 1
        }

x = np.random.randint(0, 100, size=sz)
y = np.random.randint(0, 100, size=sz)
dim_coords = [coord_val for coord_name, coord_val in coords.items()
              if coord_name in dims]

X = xr.DataArray(x, coords=coords, dims=dims)
Y = xr.DataArray(y, coords=coords, dims=dims)


# Zero step: 'data creation'

Q = {}
Q[0] = xr.Dataset({'X0': X, 'Y0': Y})

proc_tree = DataProcTree()

params = {
    'par': 0,
    'par2': {'par21': 0, 'par22': 0},
    'par3': {'par3a': {'par3b': 0}}
}
data_info = {
    'variables': {
        'X0': 'The variable X0',
        'Y0': 'The variable Y0'
    }
}
proc_tree.add_process_step('Step 0', 'step0_func()', params, data_info)

Q[0].attrs = usf.flatten_dict(proc_tree.proc_steps)


# Step 1: Q0 -> Q1

Q[1] = xr.Dataset({'X1': X, 'Y1': Y})

#attr.from_xarray_attrs(Q[0].attrs)

params = {
    'par': 1,
    'par2': {'par21': 11, 'par22': 111},
    'par3': {'par3a': {'par3b': 111}}
}
data_info = {
    'variables': {
        'X1': 'The variable X1',
        'Y1': 'The variable Y1'
    }
}
proc_tree.add_process_step('Step 1', 'step1_func()', params, data_info)

Q[1].attrs = usf.flatten_dict(proc_tree.proc_steps)


# Print

print('\n==== Step 0 ====')
attr = usf.unflatten_dict(Q[0].attrs)
pprint(attr)

print('\n==== Step 1 ====')
attr = usf.unflatten_dict(Q[1].attrs)
pprint(attr)



