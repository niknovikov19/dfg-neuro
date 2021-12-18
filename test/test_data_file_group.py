# -*- coding: utf-8 -*-

import importlib
import itertools
import os
import sys

import numpy as np
from pprint import pprint
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group as dfg
import test_utils as test
#importlib.reload(data_file_group)


# =============================================================================
# fpath_dfg = (r'H:\WORK\Camilo\Processing_Pancake_2sess_allchan' 
#               '\dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0)')
# 
# dfg_test = dfg.DataFileGroup()
# dfg_test.load(fpath_dfg)
# 
# entries = dfg.get_table_entries()
# =============================================================================


def make_dataset(dims, coord_list, sz):
    
    # Description of the coordinates
    coords = {}
    for n, coord in enumerate(coord_list):
        dim = coord if isinstance(coord, str) else coord[1]
        dim_num = dims.index(dim)
        coord_vals = np.arange(sz[dim_num])
        if isinstance(coord, str):           
            coords[coord] = coord_vals
        else:
            coords[coord[0]] = (coord[1], coord_vals * 10)
    
    # Create a numeric numpy.ndarray, in which the value of each element
    # encodes the position of this element
    x = np.ndarray(sz, dtype=np.int)
    dim_coords = [coord_val for coord_name, coord_val in coords.items()
                  if coord_name in dims]
    idx = itertools.product(*dim_coords)
    for id in idx:
        x[id] = test.lst_encode(list(id))
    
    # Convert numeric ndarray to string ndarray    
    xs = x.astype(np.str)
    xs = xs.astype(np.object)
    
    # Create xarray.DataArray from the ndarray
    Qxyz = xr.DataArray(xs, coords=coords, dims=dims)
    
    # Create several slices of the DataArray
    Qyz = Qxyz.sel(x=0, drop=True).copy()
    Qxz = Qxyz.sel(y=0, drop=True).copy()
    Qxy = Qxyz.sel(z=0, drop=True).copy()
    
    # Put the created DataArray and its slices into a xarray.Dataset
    Q = xr.Dataset({'Qxyz': Qxyz, 'Qyz': Qyz, 'Qxz': Qxz, 'Qxy': Qxy})
    return Q


dims = ['x', 'y', 'z']
coord_list = ['x', 'y', 'z', ('ya', 'y')]

coord_names = [c if isinstance(c, str) else c[0] for c in coord_list]

sz_values = [
        [3, 2, 4],
        [2, 2, 3],
        [4, 2, 3]
]

dirpath_data_base = 'H:\WORK\Camilo\TEST\dfg_test'
fpath_data_list = [os.path.join(dirpath_data_base, f'entry_{n}', 'data_root')
                   for n in range(len(sz_values))]

root_proc_step = {
    'name': 'Root',
    'function': '',
    'data_desc_out':  {
        'variables': {v: v + ' variable'
                      for v in ['Qxyz', 'Qyz', 'Qxz', 'Qxy']},
        'outer_dims': ['dataset_num'],
        'outer_coords': {
            'dataset_num': 'Number of a dataset',
        },
        'inner_dims': dims,
        'inner_coords': {c: c + ' coordinate' for c in coord_names},
        'fpath_data_column': 'fpath_data',
    },
    'params': {}
}
 
# Create a DataFileGroup object
dfg_test = dfg.DataFileGroup()
dfg_test.create(root_proc_step)

# Create Dataset objects of different size and add them to the DataFileGroup
for n, (sz, fpath_data) in enumerate(zip(sz_values, fpath_data_list)):
    X = make_dataset(dims, coord_list, sz)
    outer_coords = {'dataset_num': n}
    dfg_test.add_entry(outer_coords, X, fpath_data)

# Load datasets, one by one, and print them into a file
fpath_txt = r'H:\WORK\Camilo\TEST\dfg_test\dfg_test_1.txt'
test.print_dfg(dfg_test, fpath_txt)
                
# Save the DataFileGroup
fpath_out = r'H:\WORK\Camilo\TEST\dfg_test\dfg_root'
dfg_test.save(fpath_out)

