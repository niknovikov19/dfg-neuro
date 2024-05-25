import xarray as xr

from . import data_file_group_2 as dfg


def _dfg_avg_ref_inner(X_in, var_name_ref='LFP', ref_type='avg'):
    
    X_out = {}    
    for var_name, X in X_in.data_vars.items():
        if var_name == var_name_ref: 
            # Subtract average / median over channels
            if ref_type == 'avg':
                Y = X - X.mean(dim='chan')
            elif ref_type == 'med':
                Y = X - X.median(dim='chan')
            else:
                raise ValueError('Unknown reference type')
        else:
            Y = X
        X_out[var_name] = Y
    
        return xr.Dataset(X_out)


def dfg_avg_ref(dfg_in, var_name_ref='LFP', ref_type='avg', need_recalc=True):
    
    proc_step_desc = ('ref', 'Subtract average/median over channels')
    params = {'var_name_ref': {'val': var_name_ref, 'short': None,
                               'desc': 'Variable to process'},
              'ref_type': {'val': ref_type, 'short': 'type',
                           'desc': 'Type of reference (average / median)'}
              }
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _dfg_avg_ref_inner, proc_step_desc, params, need_recalc)    
    return dfg_out
