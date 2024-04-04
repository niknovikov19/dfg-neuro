# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:11:00 2024

@author: Nikita

"""

import os
from pathlib import Path
from pprint import pprint
import sys

import numpy as np
import pandas as pd
import xarray as xr

# Temporarily add the parent directory of `base` to sys.path
parent_dir = Path(__file__).resolve().parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from data_file_group_2 import DataFileGroup


# Number of sessions
nsess = 4

# Epoching parameters
# Column with time markers that will become zero in the trials
event_time_column = 'stim_time'
# Columns with event metadata (each will be a coord by 'trial_num' dimension)
event_meta_columns = ['stim_id', 'stim_name']
# Beginning and end of an epoch relative to its central event
trial_win = (-20, 50)

# Inner data parameters
nchans = 5
dt = 0.5
T = 2000
chan_depths = np.linspace(0, 1000, nchans)
t = np.arange(0, T, dt)
nsamples = len(t)

# Number of trials to generate in each session
ntrials = 10

def get_xarray_coords_dict(X):
    return {name: (c.dims[0], c.values) for name, c in X.coords.items()}


# Working folders
dirpath_work = Path('..') / 'data' / 'dfg_epoch_example'
os.makedirs(dirpath_work, exist_ok=True)

# Description of the -1-st processing step (epoched LFP creation)
root_step = {
    'name': 'Create epoched LFP data',
    'function': '',
    'data_desc_out':  {
        'variables': {'LFP': 'LFP amplitude'},
        'outer_dims': ['session'],
        'outer_coords': {'session': 'Session name'},
        'inner_dims': ['trial', 'chan', 'time'],
        'inner_coords': {'trial': 'Trial number',
                         'chan': 'Channel number',
                         'depth': 'Channel depth',
                         'time': 'Sample time, s',
                         'sample': 'Sample number'}, 
        'fpath_data_column': 'fpath_lfp_epoched',
    },
    'params': {
        'ev_time_col': {'desc': 'Column with event times',
                        'value': event_time_column},
        'ev_meta_col': {'desc': 'Columns with event metadata',
                        'value': event_meta_columns},
        'trial_win': {'desc': 'Epoch tim window',
                      'value': trial_win},
        }
    }

# Create dfg object 
dfg = DataFileGroup()
dfg.create(root_step)

# Create event table and epoched random data for each session
for sess in range(nsess):
    
    # Session folder
    dirpath_sess = dirpath_work / f'Session{sess}'
    os.makedirs(dirpath_sess, exist_ok=True)

    # Create and save test event table
    ev_data_ = {'stim_id': np.arange(ntrials),
                'stim_time': (np.arange(ntrials) + 1) * 100 + sess,
                'stim_name': [f'S{n}' for n in range(ntrials)]}
    ev_data = pd.DataFrame(ev_data_)
    ntrials = len(ev_data['stim_time'])
    ev_data.to_csv(dirpath_sess / 'events.csv')
    
    # Create test data
    #X_ = np.random.randn(nchans, nsamples)
    X_ = (np.linspace(0, T - dt, nsamples).reshape((1, -1)) +
          np.linspace(0, nchans - 1, nchans).reshape((-1, 1)) * 0.1)
    
    # Associate an xarray with the data
    coords = {
        'chan': ('chan', np.arange(nchans)),
        'depth': ('chan', chan_depths),
        'time': ('time', t),
        'sample': ('time', np.arange(nsamples))
        }
    X = xr.DataArray(X_, dims=['chan', 'time'], coords=coords)
    
    # Calculate epoch time samples
    dt = t[1] - t[0]
    t_trial = np.arange(trial_win[0], trial_win[1], dt)
    
    # Allocate array for epoched data
    dims_ep = ('trial', *X.dims)
    coords_ep = {'trial': np.arange(ntrials)} | get_xarray_coords_dict(X)
    coords_ep['time'] = ('time', t_trial)
    coords_ep['sample'] = ('time', np.arange(len(t_trial)))
    Xep = xr.DataArray(np.nan, dims=dims_ep, coords=coords_ep)

    # Epoch the data
    for n, ev in ev_data.iterrows():
        t0 = ev[event_time_column]
        t_win = (t0 + trial_win[0], t0 + trial_win[1] - dt)
        Xep.loc[dict(trial=n)].values[...] = X.sel(time=slice(*t_win)).values
    
    # Add metadata for each trial from event table
    for col in event_meta_columns:
        Xep.coords[col] = ('trial', ev_data[col])
        
    # Put created DataArray into 'LFP' variable of Dataset
    Xep = xr.Dataset({'LFP': Xep})
    
    # Add new entry: row in the outer table + file with inner data
    outer_coords = {'session': sess}  # identify outer table entry
    fpath_inner = str((dirpath_sess / 'lfp_epoched.pkl').resolve())
    dfg.add_entry(outer_coords, Xep, fpath_inner)  # X -> fpath_inner
    
# Save dfg object
fpath_dfg = dirpath_work / 'dfg_lfp_epoched.pkl'
dfg.save(fpath_dfg)

# Load dfg object
dfg = DataFileGroup()
dfg.load(fpath_dfg)

# Print outer table
print('OUTER TABLE:')
print(dfg.outer_table)
print('FPATH_DATA:')
print(dfg.outer_table.fpath_lfp_epoched.values)

# Print information about the processing step
print('PROCESSING STEPS:')
dfg.print_proc_tree()

# Load inner data
entry = dfg.get_table_entry_by_coords({'session': 2})
X = dfg.load_inner_data(entry)

# Print inner data
print('OUTER COORDINATE:')
print('session=' + str(X.attrs['outer_coord_vals.session']))
print('INNER DATA:')
print(X)
print('LFP:')
print(X.LFP)
