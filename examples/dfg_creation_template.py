# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 14:59:59 2024

@author: Nikita
"""


import os
from pathlib import Path
from pprint import pprint
import sys

import numpy as np
#import pandas as pd
import xarray as xr

# Temporarily add the parent directory of `base` to sys.path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from data_file_group_2 import DataFileGroup


# Describe the creation step here
root_step = {
    'name': 'Data creation example',  # your description here
    'function': '',                   # refer to creating function if you want
    'data_desc_out':  {
        'variables': {'var1': 'Variable 1'},  # variable (e.g. 'LFP')
        'outer_dims': ['outer1'],  # Outer dimensions (of the external table)
        'outer_coords': {  # Coordinates associated with outer dimensions
            'outer1': 'Outer dimension 1 (main coord.)',
        },
        'inner_dims': ['inner1', 'inner2'],  # Inner dimensions (of stored xarrays)
        'inner_coords': {  # Coordinates associated with inner dimensions (>=1)
            'inner1': 'Inner dimension 1 (main coord.)',
            'inner2': 'Inner dimension 2 (main coord.)',
            'inner2_extra': 'Inner dimension 2 (extra coord.)',
        },
        'fpath_data_column': 'fpath_data',  # columns for paths to xarray files
    },
    'params': {  # parameters (arbitrary)
        'param1': {'desc': 'Parameter 1', 'value': 1},
        }
    }

# Working folders
dirpath_work = Path('..') / 'data' / 'dfg_creation_example'
dirpath_inner = dirpath_work / 'inner_data'
os.makedirs(dirpath_inner, exist_ok=True)

# Create dfg object 
dfg = DataFileGroup()
dfg.create(root_step)

# Loop over combinations of outer coordinate values
for outer1_coord_val in ['o1', 'o2', 'o3']:
    
    # Prepare the inner data
    inner1_sz, inner2_sz = (10, 20)
    X_ = np.random.randn(inner1_sz, inner2_sz)
    coords = {
        'inner1': ('inner1', np.arange(inner1_sz)),
        'inner2': ('inner2', np.arange(inner2_sz)),
        'inner2_extra': ('inner2', np.arange(inner2_sz) * 10),
        }
    X = xr.DataArray(X_, dims=['inner1', 'inner2'], coords=coords)
    X = xr.Dataset({'var1': X})
    
    # Add new entry: row in the outer table + file with inner data
    outer_coords = {'outer1': outer1_coord_val}  # identify outer table entry
    fname_inner = f'example_outer1={outer1_coord_val}.pkl'  # inner data filename
    fpath_inner = str(dirpath_inner / fname_inner)
    dfg.add_entry(outer_coords, X, fpath_inner)  # X -> fpath_inner

# Save dfg object
fpath_dfg = dirpath_work / 'dfg_example.pkl'
dfg.save(fpath_dfg)

# Load dfg object
dfg = DataFileGroup()
dfg.load(fpath_dfg)

# Print outer table
print('OUTER TABLE:')
print(dfg.outer_table)
print('FPATH_DATA:')
print(dfg.outer_table.fpath_data.values)

# Print information about the processing step
print('PROCESSING STEPS:')
dfg.print_proc_tree()

# Load inner data
entry = dfg.get_table_entry_by_coords({'outer1': 'o1'})  # integer index
X = dfg.load_inner_data(entry)

# Print inner data
print('OUTER COORDINATE:')
print('outer1=' + X.attrs['outer_coord_vals.outer1'])
print('INNER DATA:')
print(X)
#print('INNER DATA ATTRIBUTES:')
#pprint(X.attrs)
