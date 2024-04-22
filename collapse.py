# -*- coding: utf-8 -*-

import xarray as xr

from . import data_file_group_2 as dfg


def _mean_inner(X_in, dim):
    X_out = {}
    for var_name, var in X_in.items():
        if dim in var.dims:
            X_out[var_name] = var.mean(dim=dim)
        else:
            X_out[var_name] = var
    return xr.Dataset(X_out)


def mean(dfg_in, dim, need_recalc=True):    
    proc_step_desc = ('mean', 'Average data over given dimension(s)')
    params = {'dim': {'val': dim, 'short': 'dim',
                              'desc': 'Dimension(s) to average over'}}
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _mean_inner, proc_step_desc, params, need_recalc
            )    
    return dfg_out

