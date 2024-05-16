import xarray as xr

from . import data_file_group_2 as dfg


def _dfg_avg_ref_inner(X_in, var_name_in='TF'):
    
    # Select the variable to process
    X = X_in[var_name_in]
    
    # Subtract average over channels
    X = X - X.mean(dim='chan')
    
    # Collect the output
    X_out = {var_name_in: X}    
    return xr.Dataset(X_out)


def dfg_avg_ref(dfg_in, var_name='TF', need_recalc=True):
    
    proc_step_desc = ('avgref', 'Subtract average over channels')
    params = {'var_name_in': {'val': var_name, 'short': None,
                              'desc': 'Variable to process'}}
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _dfg_avg_ref_inner, proc_step_desc, params, need_recalc)    
    return dfg_out
