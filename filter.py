import numpy as np
import xarray as xr

from . import data_file_group_2 as dfg
from ._filter import bandpass, highpass, lowpass


def _filter(X, filt_type, freq, fs, order):
    if filt_type == 'bandpass':
        return bandpass(X, freq[0], freq[1], fs, order)
    elif filt_type == 'highpass':
        return highpass(X, freq, fs, order)
    elif filt_type == 'lowpass':
        return lowpass(X, freq, fs, order)
    else:
        raise ValueError('Unknown filtering type')
        

def _dfg_filt_inner(X_in, filt_type, freq, order, fs=None, var_names_filt=None):
    
    var_names_in = list(X_in.data_vars)    
    if var_names_filt is None:
        var_names_filt = var_names_in
        
    if fs is None:
        fs = X_in.attrs['proc_steps.(1).0.params.fs.value']
    
    X_out = {}    
    for var_name in var_names_in:            
        if var_name in var_names_filt:            
            X_out[var_name] = xr.full_like(X_in[var_name], np.nan)
            X_out[var_name].values = _filter(
                X_in[var_name].values, filt_type, freq, fs, order)
        else:
            X_out[var_name] = X_in[var_name]
            
    return xr.Dataset(X_out)


def dfg_filt(dfg_in, filt_type, freq, order, var_names_filt, need_recalc=True):
    
    proc_step_desc = ('filter', 'Filter data')
    params = {'var_names_filt': {'val': var_names_filt, 'short': None,
                                 'desc': 'Variables to process'},
              'filt_type': {'val': filt_type, 'short': None,
                            'desc': 'Filter type'},
              'freq': {'val': freq, 'short': 'f',
                            'desc': 'Filter pass band or cutoff frequency'},
              'order': {'val': order, 'short': None,
                        'desc': 'Filter order'}
              }
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _dfg_filt_inner, proc_step_desc, params, need_recalc)    
    return dfg_out
