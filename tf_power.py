# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:20:45 2024

@author: Nikita
"""

import xarray as xr

from . import data_file_group_2 as dfg


def _calc_dfg_tfpow_inner(X_in, var_name_in='TF', var_name_out=None):
    
    if var_name_out is None:
        var_name_out = var_name_in + 'pow'
    
    # Select the variable to process
    X = X_in[var_name_in]
    
    # Total power (by trials)
    Xpow = (X * X.conj()).real
    
    # Phase-locked power (average over trials)
    Xpl = X.mean(dim='trial')
    Xpow_pl = (Xpl * Xpl.conj()).real
    
    # Collect the output
    X_out = {var_name_out: Xpow,
             var_name_out + '_pl': Xpow_pl}    
    return xr.Dataset(X_out)


def calc_dfg_tfpow(dfg_in, var_name='TF', need_recalc=True):
    
    proc_step_desc = ('TFpow', 'Complex amplitudes -> Power')
    params = {'var_name_in': {'val': var_name, 'short': None,
                              'desc': 'Variable to process'}}
    vars_new_desc = {
        var_name + 'pow': 'Total power (by trials): |X|^2',
        var_name + 'pow_pl': 'Phase-locked power (trial-averaged): |<X>|^2'
        }
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _calc_dfg_tfpow_inner, proc_step_desc, params,
            need_recalc, vars_new_desc
            )    
    return dfg_out
