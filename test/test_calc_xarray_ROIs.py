# -*- coding: utf-8 -*-
"""Tests for ROI reducing functions.

"""

import itertools
import os
import sys

import numpy as np
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

#import useful as usf
import data_proc_common as dpc


# Reducing functions used for testing data_proc_common::calc_xarray_ROIs()

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


# Create test 3-d DataArray's that contains in a cell with the
# coordinates (i,j,k) a number equal to (100*k + 10*j + i) or
# its string represresentation

def lst_encode(lst, k=1):
    """[i, j, k] -> (100k + 10*j + i) """
    res = 0
    for n, x in enumerate(lst):
        res += (x * (10**k)**n)
    return res

dims = ['dim0', 'dim1', 'dim2']
sz = [3, 2, 4]
coords = {
        'dim0':     np.arange(sz[0]),    # 0, 1, 2
        'dim1':     np.arange(sz[1]),    # 0, 1
        'dim2':     np.arange(sz[2]),    # 0, 1, 2, 3
        'dim2a':    ('dim2', 10*np.arange(sz[2]))
        }

x = np.ndarray(sz, dtype=np.int)
dim_coords = [coord_val for coord_name, coord_val in coords.items()
              if coord_name in dims]
idx = itertools.product(*dim_coords)
for id in idx:
    x[id] = lst_encode(list(id))
    
xs = x.astype(np.str)
xs = xs.astype(np.object)

X = xr.DataArray(x, coords=coords, dims=dims)    
Xs = xr.DataArray(xs, coords=coords, dims=dims)


ROI_coords = {}
ROI_descs = {}

# Test 1: reduce (dim0, dim2)
ROI_coords[0] = ['dim0', 'dim2a']
ROI_descs[0] = [
        {'name': 'ROI0', 'limits': {'dim0': (0.5, 2), 'dim2a': (10, 29)}},
        {'name': 'ROI1', 'limits': {'dim0': (0, 0), 'dim2a': (25, 30)}},
        {'name': 'ROI2', 'limits': {'dim0': (-1, 3), 'dim2a': (10, 20)}},
        ]

# Test 2: reduce (dim1, dim2)
ROI_coords[1] = ['dim1', 'dim2']
ROI_descs[1] = [
        {'name': 'ROI0', 'limits': {'dim1': (0, 1), 'dim2': (1, 2.9)}},
        {'name': 'ROI1', 'limits': {'dim1': (1, 1), 'dim2': (2.5, 3)}},
        ]

# Test 3: reduce dim0
ROI_coords[2] = ['dim0']
ROI_descs[2] = [
        {'name': 'ROI0', 'limits': {'dim0': (0, 1)}},
        {'name': 'ROI1', 'limits': {'dim0': (1, 2)}},
        {'name': 'ROI2', 'limits': {'dim0': (0, 2)}},
        {'name': 'ROI3', 'limits': {'dim0': (2, 2)}},
        ]

# Test 4: reduce all dims
ROI_coords[3] = ['dim0', 'dim1', 'dim2']
ROI_descs[3] = [
        {'name': 'ROI0', 
         'limits': {'dim0': (1, 2), 'dim1': (0, 1), 'dim2': (2, 3)}},
        {'name': 'ROI1', 
         'limits': {'dim0': (0, 2), 'dim1': (0, 0), 'dim2': (0, 1)}},
        ]

# TODO test different order of ROI dimensions and coords

# Perform tests
for test_num in range(4):
    
    # For the string array, combine ROI elements by string concatenation
    reduce_fun = lambda Z, dims: reduce_fun_xarray(Z, dims,
                                                   reduce_fun_1d_strjoin)
    Ys = dpc.calc_xarray_ROIs(Xs, 'ROI', ROI_coords[test_num],
                              ROI_descs[test_num], reduce_fun)
    
    # For the integer array, combine ROI elements by summation
    reduce_fun = lambda Z, dims: Z.sum(dim=dims)
    Y = dpc.calc_xarray_ROIs(X, 'ROI', ROI_coords[test_num],
                             ROI_descs[test_num], reduce_fun)

    print(f'======= TEST {test_num} =======\n\n')
    print(Xs)
    print('\n')
    print(Ys)
    print('\n')
    print(Y)
    print('\n')


