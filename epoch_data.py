from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import data_file_group_2 as dfg


def _get_xarray_coords_dict(X):
    return {name: (c.dims[0], c.values) for name, c in X.coords.items()}


def _epoch_data(X, fs, ev_times, trial_win, ev_meta=None, time_dim=1):
    
    dt = 1 / fs
    nsamples = X.shape[time_dim]
    t = np.arange(nsamples) * dt
    
    # Calculate epoch time samples
    t_trial = np.arange(trial_win[0], trial_win[1] + dt, dt)
    
    # Allocate array for epoched data
    dims_in = X.dims
    coords_in = _get_xarray_coords_dict(X)
    ntrials = len(ev_times)
    dims_ep = ('trial', *dims_in)
    coords_ep = {'trial': np.arange(ntrials)} | coords_in
    time_dim = coords_in['time'][0]    # time or sample
    coords_ep['time'] = (time_dim, t_trial)
    coords_ep['sample'] = (time_dim, np.arange(len(t_trial)))
    Xep = xr.DataArray(np.nan, dims=dims_ep, coords=coords_ep)

    # Epoch the data
    for n, t0 in enumerate(ev_times):
        print(f'{n} / {len(ev_times)}')
        t0 = t[np.argmin(np.abs(t - t0))]
        t_win = np.round((t0 + trial_win) / dt)
        Xep.loc[dict(trial=n)].values[...] = X.sel(sample=slice(*t_win)).values
    
    # Add metadata for each trial from event table
    if ev_meta is not None:
        for col in ev_meta.columns:
            Xep.coords[col] = ('trial', ev_meta[col])
    return Xep


def _dfg_epoch_data(X_in, ev_name, ev_time_col, ev_meta_col,
                    trial_win, ntrials_skip=0, fs=None,
                    ev_file_name='trial_events.csv'):
    
    # TODO: time limits
    
    # Sampling rate
    if fs is None:
        fs = X_in.attrs['proc_steps.(1).0.params.fs.value']
        
    # Load event table
    dirpath_sess = Path(X_in.attrs['fpath_parent']).parent
    fpath_ev = dirpath_sess / ev_file_name
    ev_data = pd.read_csv(fpath_ev)
    
    # Discard trials that don't fit into the data
    # and several first trials (if needed)
    t = X_in.time.values
    ev_times = ev_data[ev_time_col]
    mask = ((ev_times + trial_win[0]) > t[0]) & ((ev_times + trial_win[1]) < t[-1])
    mask &= (np.arange(len(ev_times)) >= ntrials_skip)
    ev_data = ev_data.loc[mask.values, :]

    ev_times = ev_data[ev_time_col]
    ev_meta = ev_data[ev_meta_col]
    
    # Epoch each variable
    X_out = {}    
    for var_name, X in X_in.data_vars.items():
        X_out[var_name] = _epoch_data(X, fs, ev_times, trial_win, ev_meta)
    
    return xr.Dataset(X_out)


def dfg_epoch_data(dfg_in, ev_name, ev_time_column, ev_meta_columns,
                trial_win, ntrials_skip=0, need_recalc=True,
                ev_file_name='trial_events.csv'):
    
    proc_step_desc = ('epoch', 'Epoch the data')
    params = {'ev_file_name': {'val': ev_file_name, 'short': None,
                      'desc': 'Event file name'},
              'ev_name': {'val': ev_name, 'short': 'ev',
                          'desc': 'Central event name'},
              'ev_time_col': {'val': ev_time_column, 'short': None,
                              'desc': 'Column with central event times'},
              'ev_meta_col': {'val': ev_meta_columns, 'short': None,
                              'desc': 'Columns with event metadata'},
              'trial_win': {'val': trial_win, 'short': 't',
                            'desc': 'Epoch time window'},
              'ntrials_skip': {'val': ntrials_skip, 'short': 'nskip',
                               'desc': 'Num. of trials in the beginning to discard'},
              }
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _dfg_epoch_data, proc_step_desc, params, need_recalc)    
    return dfg_out
