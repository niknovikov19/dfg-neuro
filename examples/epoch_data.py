# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:11:00 2024

@author: Nikita

"""

import numpy as np
import pandas as pd
import xarray as xr


# Create test event table
ev_data_ = {'stim_id': [1, 3, 4], 'stim_time': [100, 200, 300], 'stim_name': ['S1', 'S2', 'S1']}
ev_data = pd.DataFrame(ev_data_)
ntrials = len(ev_data['stim_time'])

# Create test data
nchans = 2
dt = 0.5
chan_depths = np.linspace(0, 1000, nchans)
t = np.arange(0, 1000, dt)
nsamples = len(t)
X_ = np.random.randn(nchans, nsamples)

# Associate an xarray with the data
coords = {
    'chan': ('chan', np.arange(nchans)),
    'depth': ('chan', chan_depths),
    'time': ('time', t),
    'sample': ('time', np.arange(nsamples))
    }
X = xr.DataArray(X_, dims=['chan', 'time'], coords=coords)

# Column with time markers that will become zero in the trials
event_time_column = 'stim_time'

# Columns with event metadata (each will be a coord by 'trial_num' dimension)
event_meta_columns = ['stim_id', 'stim_name']

# Beginning and end of an epoch relative to its central event
trial_win = (-20, 50)

# Calculate epoch time samples
dt = t[1] - t[0]
t_trial = np.arange(trial_win[0], trial_win[1], dt)

def get_xarray_coords_dict(X):
    return {name: (c.dims[0], c.values) for name, c in X.coords.items()}

# Allocate array for epoched data
dims_ep = ('trial', *X.dims)
coords_ep = {'trial': np.arange(ntrials)} | get_xarray_coords_dict(X)
coords_ep['time'] = ('time', t_trial)
coords_ep['sample'] = ('time', np.arange(len(t_trial)))
Xep = xr.DataArray(np.nan, dims=dims_ep, coords=coords_ep)

trials = []

for n, ev in ev_data.iterrows():
    
    # Slice the DataArray for each time interval
    t0 = ev[event_time_column]
    t_win = (t0 + trial_win[0], t0 + trial_win[1] - dt)
    Xep.loc[dict(trial=n)].values[...] = X.sel(time=slice(*t_win)).values
    #print(X.sel(time=slice(*t_win)).shape)
    
# =============================================================================
#     # Assign a new coordinate for the trial dimension (no data copying)
#     # This temporarily creates an extra dimension for concatenation
#     coords = {'trial_num': n}
#     for ev_meta in event_meta_colums:
#         coords[ev_meta] = ev[ev_meta]
#     x = x.expand_dims('trial_num').assign_coords(coords)
# =============================================================================

# Add metadata for each trial from event table
trial_coords = {col: ('trial', ev[col]) for col in event_meta_columns}
Xep = Xep.ssign_coords(trial_coords)
