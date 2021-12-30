# -*- coding: utf-8 -*-
"""Tests for ROI reducing functions.

"""

import importlib
import itertools
import os
import sys

import numpy as np
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

#import useful as usf
import roi_utils as roi
importlib.reload(roi)


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

dims = ['x', 'y', 'z']
sz = [3, 2, 4]
coords = {
        'x':     np.arange(sz[0]),    # 0, 1, 2
        'y':     np.arange(sz[1]),    # 0, 1
        'z':     np.arange(sz[2]),    # 0, 1, 2, 3
        'ya':    ('y', 10*np.arange(sz[1]))
        }

x = np.ndarray(sz, dtype=np.int)
dim_coords = [coord_val for coord_name, coord_val in coords.items()
              if coord_name in dims]
idx = itertools.product(*dim_coords)
for id in idx:
    x[id] = lst_encode(list(id))
    
xs = x.astype(np.str)
xs = xs.astype(np.object)

Q = xr.DataArray(xs, coords=coords, dims=dims)


ROI_descs = {}

# ROI descriptions
ROI_descs['x'] = [
        {'name': 'xROI0', 'limits': {'x': (0, 1)}},
        {'name': 'xROI1', 'limits': {'x': (1, 1)}},
        {'name': 'xROI2', 'limits': {'x': (0, 2)}},
        ]
ROI_descs['y'] = [
        {'name': 'yROI0', 'limits': {'ya': (0, 10)}},
        {'name': 'yROI1', 'limits': {'ya': (1, 10)}},
        ]
ROI_descs['z'] = [
        {'name': 'zROI0', 'limits': {'z': (0, 2)}},
        {'name': 'zROI1', 'limits': {'z': (1, 3)}},
        {'name': 'zROI2', 'limits': {'z': (0, 3)}},
        {'name': 'zROI3', 'limits': {'z': (2, 3)}},
        ]

# TODO test different order of ROI dimensions and coords

# For the string array, combine ROI elements by string concatenation
reduce_fun = lambda Z, dims: reduce_fun_xarray(Z, dims, reduce_fun_1d_strjoin)
    
QROI = {}
QROI['x'], _ = roi.calc_xarray_ROIs(Q, ['x'], ROI_descs['x'], reduce_fun)
QROI['y'], _ = roi.calc_xarray_ROIs(Q, ['ya'], ROI_descs['y'], reduce_fun)
QROI['z'], _ = roi.calc_xarray_ROIs(Q, ['z'], ROI_descs['z'], reduce_fun)

QROI2 = {}
QROI2['xy'], _ = roi.calc_xarray_ROIs(QROI['y'], ['x'], ROI_descs['x'],
                                   reduce_fun, ROIset_dim_to_combine='yROI')
QROI2['xz'], _ = roi.calc_xarray_ROIs(QROI['z'], ['x'], ROI_descs['x'],
                                   reduce_fun, ROIset_dim_to_combine='zROI')
QROI2['yx'], _ = roi.calc_xarray_ROIs(QROI['x'], ['ya'], ROI_descs['y'],
                                   reduce_fun, ROIset_dim_to_combine='xROI')
QROI2['yz'], _ = roi.calc_xarray_ROIs(QROI['z'], ['ya'], ROI_descs['y'],
                                   reduce_fun, ROIset_dim_to_combine='zROI')
QROI2['zx'], _ = roi.calc_xarray_ROIs(QROI['x'], ['z'], ROI_descs['z'],
                                   reduce_fun, ROIset_dim_to_combine='xROI')
QROI2['zy'], _ = roi.calc_xarray_ROIs(QROI['y'], ['z'], ROI_descs['z'],
                                   reduce_fun, ROIset_dim_to_combine='yROI')

QROI3 = {}
QROI3['xyz'], _ = roi.calc_xarray_ROIs(QROI2['yz'], ['x'], ROI_descs['x'],
                                    reduce_fun, ROIset_dim_to_combine='yzROI')
QROI3['zxy'], _ = roi.calc_xarray_ROIs(QROI2['xy'], ['z'], ROI_descs['z'],
                                    reduce_fun, ROIset_dim_to_combine='xyROI')

print('\n====  Original  ====')
print(Q)

for ROI_name, qROI in QROI.items():
    print(f'\n====  ROI: {ROI_name}  ====')
    print(qROI)
    
print('\n====================')
for ROI_name, qROI in QROI2.items():
    print(f'\n====  ROI: {ROI_name}  ====')
    print(qROI)
    
print('\n====================')
for ROI_name, qROI in QROI3.items():
    print(f'\n====  ROI: {ROI_name}  ====')
    print(qROI)
    print('Coordinates:')
    for coord_name, coord_vals in qROI.coords.items():
        print(coord_name)
        print(coord_vals.values)

