import numpy as np
import xarray as xr

from . import data_file_group_2 as dfg


def _run_avg(X, w):
    w2 = int(w / 2)
    C = X.cumsum(dim='sample')
    Y = (C.roll(sample=-w2) - C.roll(sample=w2)) / w
    Y[:, :w2] = Y[:, w : w2 : -1].data
    Y[:, -w2:] = Y[:, -w2 - 1 : -w - 1 : -1].data
    return Y

def _dfg_normalize(X_in, T_win, var_name_norm='LFP'):
   
    # Length of averaging window in samples
    tt = X_in.time.values
    dt = tt[1] - tt[0]
    w = int(T_win / dt)
   
    X_out = {}    
    for var_name, X in X_in.data_vars.items():            
        if var_name == var_name_norm:
            Xavg = _run_avg(X, w)
            Xstd = np.sqrt(_run_avg((X - Xavg)**2, w))
            X_out[var_name] = X / Xstd
        else:
            X_out[var_name] = X
            
    return xr.Dataset(X_out)


def dfg_normalize(dfg_in, T_win, var_name_norm='LFP', need_recalc=True):
    
    proc_step_desc = ('norm', 'Normalize data by running std.')
    params = {'var_name_norm': {'val': var_name_norm, 'short': None,
                                 'desc': 'Variable to process'},
              'T_win': {'val': T_win, 'short': 'T',
                        'desc': 'Time window for running std. calclulation'}
              }
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _dfg_normalize, proc_step_desc, params, need_recalc)    
    return dfg_out
