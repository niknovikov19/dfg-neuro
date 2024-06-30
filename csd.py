import numpy as np
import xarray as xr

from . import data_file_group_2 as dfg


def _csd(X, axis, d):
    #axis = -1
    #print(f'Shape: {X.shape}, axis={axis}')
    Xfirst = np.expand_dims(np.take(X, 0, axis), axis)
    Xlast = np.expand_dims(np.take(X, -1, axis), axis)
    X = np.concatenate((Xfirst, X, Xlast), axis)
    Y = -np.diff(X, 2, axis) / d**2
    return Y
    
def _dfg_csd_inner(X_in):
    
    # Copy input variables to output
    X_out = {}    
    for var_name, X in X_in.data_vars.items():
        if var_name != 'LFP':
            X_out[var_name] = X
    
    # Change channel order: from superficial to deep
    # Take each 4-th channel, so they are all in the same column
    X = X_in['LFP']
    X = X.isel(chan=slice(None, None, -4))
    #X = X.compute()
    
    if (X.dims[-1] != 'sample') or (X.dims[-2] != 'chan'):
        raise ValueError('Last two dimensions should be chan and sample')
    
    chan_axis = X.dims.index('chan')
    d = float(X.depth[1] - X.depth[0]) * 1e-3
    
    Y = xr.apply_ufunc(
        _csd, X,
        kwargs={'axis': chan_axis, 'd': d},
        input_core_dims=[['chan', 'sample']],
        output_core_dims=[['chan', 'sample']],
        vectorize=False,
        dask='parallelized',
        #dask='forbidden',
        output_dtypes=[np.float64]
    )
    X_out['CSD'] = Y
    
    return xr.Dataset(X_out)


def dfg_csd(dfg_in, need_recalc=True):
    
    proc_step_desc = ('CSD', 'CSD')
    params = {}
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _dfg_csd_inner, proc_step_desc, params, need_recalc)    
    return dfg_out


# =============================================================================
# if __name__ == '__main__':
#     
#     fpath_in = r"E:\M1_exp\Proc\230726_3744_1622VAL\lfp_raw_filter_(type=highpass_f=0.5).nc"
#     X = xr.open_dataset(fpath_in)
#     X = X.chunk({'sample': 200000, 'chan': -1})
#     Y = _dfg_csd_inner(X)
#     Y.compute()
# =============================================================================
