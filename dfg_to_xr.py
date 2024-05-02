import os
from pathlib import Path
from time import time
import sys
import warnings

import xarray as xr

# Add dfg folder to path
parent_dir = str(Path('..').resolve())
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import dfg

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def dfg_to_xr(dfg_in):

    # Get outer dimensions and coordinate values
    outer_dims = dfg_in.get_data_desc()['outer_dims']
    outer_coords = {}
    for outer_dim in outer_dims:
        outer_coords[outer_dim] = (
            outer_dim, dfg_in.outer_table[outer_dim].unique())
    
    X_out = None
    
    for outer_coords_cur, X in dfg_in.items():
        
        print(outer_coords_cur)
        
        # Allocate output
        if X_out is None:
            X_out = {}
            for var_name, Xvar in X.items():
                coords = outer_coords | dfg.usf.get_xarray_coords_dict(Xvar)
                dims = tuple(outer_dims) + Xvar.dims
                X_out[var_name] = xr.DataArray(coords=coords, dims=dims)
            X_out = xr.Dataset(X_out)
            
        for var_name, Xvar in X.items():
            X_out[var_name].loc[outer_coords_cur] = Xvar
            
    return X_out   
