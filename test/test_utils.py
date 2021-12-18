# -*- coding: utf-8 -*-

import importlib
import itertools
import os
import sys

import numpy as np
from pprint import pprint
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group
importlib.reload(data_file_group)


def reduce_fun_1d_strjoin(x):
    """Join all elements of a string numpy 1-d array.
    
    This function is used to test reducing of n-d data over ROI's. 
    In this case, the 'reduction' over a ROI is just a concatenation of
    the elements that belong to this ROI.
    """
    return  '_'.join(x)

def reduce_fun_np(x: np.ndarray, axes_reduce, reduce_fun_1d):
    """Reduce a numpy ndarray over an unrolled combination of axes."""
    
    # Rearrage axes so the axes to reduce are the last ones
    axes_orig = np.arange(x.ndim)
    axes_new = np.concatenate((np.delete(axes_orig, axes_reduce), axes_reduce))
    y = np.moveaxis(x, axes_new, axes_orig)
    
    # Unroll to-be-reduced axes
    ndims_reduce = len(axes_reduce)
    ndims_keep = y.ndim - ndims_reduce
    shape_keep = np.array(y.shape)[:ndims_keep]
    shape_new = np.append(shape_keep, -1)
    y = y.reshape(shape_new)
    
    # Allocate output
    z = np.ndarray(shape_keep, dtype=x.dtype)
    
    # Walk trough the combinations of non-reduced coordinates and
    # apply a reduce function
    coords_keep = [np.arange(l) for l in shape_keep]
    len_reduce = y.shape[-1]
    idx = itertools.product(*coords_keep)
    for id in idx:
        ys = y[(*id,range(len_reduce))]    # to-be-reduced 1-d slice
        z[id] = reduce_fun_1d(ys)
        
    return z

def reduce_fun_xarray(X: xr.DataArray, dims_reduce, reduce_fun_1d):
    """Reduce an xr DataArray over an unrolled combination of dims."""
    reduce_fun = (lambda X, axis: reduce_fun_np(X, axis, reduce_fun_1d))
    return X.reduce(reduce_fun, dims_reduce)

# For the string array, combine ROI elements by string concatenation
def reduce_fun_strjoin(Z, dims):
    return reduce_fun_xarray(Z, dims, reduce_fun_1d_strjoin)


# Create test 3-d DataArray's that contains in a cell with the
# coordinates (i,j,k) a number equal to (100*k + 10*j + i) or
# its string represresentation

def lst_encode(lst, k=1):
    """[i, j, k] -> (100k + 10*j + i) """
    res = 0
    for n, x in enumerate(lst):
        res += (x * (10**k)**n)
    return res


def print_dfg(dfg, fpath_txt):
    """Load DataFileGroup's datasets, and print them into a file. """
    table_entries = dfg.get_table_entries()
    with open(fpath_txt, 'w') as fid:
        for table_entry in table_entries:
            print(f'\n\n{table_entry}\n===================\n', file=fid)
            X = dfg.load_inner_data(table_entry)
            for var_name, var_data in X.data_vars.items():
                print(f'\n==  VAR: {var_name}  ==', file=fid)
                print(var_data, file=fid)
                print('Coordinates:', file=fid)
                for coord_name, coord_vals in var_data.coords.items():
                    print(coord_name, file=fid)
                    print(coord_vals.values, file=fid)
                    
                    