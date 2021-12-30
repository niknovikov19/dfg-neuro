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

Qxyz = xr.DataArray(xs, coords=coords, dims=dims)
Qyz = Qxyz.sel(x=0, drop=True).copy()
Qxz = Qxyz.sel(y=0, drop=True).copy()
Qxy = Qxyz.sel(z=0, drop=True).copy()

Q = xr.Dataset({'Qxyz': Qxyz, 'Qyz': Qyz, 'Qxz': Qxz, 'Qxy': Qxy})


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
ROI_descs['xy'] = [
        {'name': 'xyROI0', 'limits': {'x': (0, 1), 'ya': (0,10)}},
        {'name': 'xyROI1', 'limits': {'x': (1, 1), 'ya': (0,10)}},
        {'name': 'xyROI2', 'limits': {'x': (0, 2), 'ya': (0,10)}},
        {'name': 'xyROI3', 'limits': {'x': (0, 1), 'ya': (1,10)}},
        {'name': 'xyROI4', 'limits': {'x': (1, 1), 'ya': (1,10)}},
        {'name': 'xyROI5', 'limits': {'x': (0, 2), 'ya': (1,10)}},
        ]
ROI_descs['zy'] = [
        {'name': 'zyROI0', 'limits': {'z': (0, 2), 'ya': (0,10)}},
        {'name': 'zyROI1', 'limits': {'z': (1, 3), 'ya': (0,10)}},
        {'name': 'zyROI2', 'limits': {'z': (0, 3), 'ya': (0,10)}},
        {'name': 'zyROI3', 'limits': {'z': (2, 3), 'ya': (0,10)}},
        {'name': 'zyROI0', 'limits': {'z': (0, 2), 'ya': (1,10)}},
        {'name': 'zyROI1', 'limits': {'z': (1, 3), 'ya': (1,10)}},
        {'name': 'zyROI2', 'limits': {'z': (0, 3), 'ya': (1,10)}},
        {'name': 'zyROI3', 'limits': {'z': (2, 3), 'ya': (1,10)}},
        ]

# TODO test different order of ROI dimensions and coords

# For the string array, combine ROI elements by string concatenation
reduce_fun = lambda Z, dims: reduce_fun_xarray(Z, dims, reduce_fun_1d_strjoin)

QROI = {}
QROI['xy'] = roi.calc_dataset_ROIs(Q, ['x', 'ya'], ROI_descs['xy'], reduce_fun)
QROI['zy'] = roi.calc_dataset_ROIs(Q, ['z', 'ya'], ROI_descs['zy'], reduce_fun)
QROI['x'] = roi.calc_dataset_ROIs(Q, ['x'], ROI_descs['x'], reduce_fun)
QROI['y'] = roi.calc_dataset_ROIs(Q, ['ya'], ROI_descs['y'], reduce_fun)
QROI['z'] = roi.calc_dataset_ROIs(Q, ['z'], ROI_descs['z'], reduce_fun)
#QROI['xz'] = roi.calc_dataset_ROIs(Q, ['x', 'z'], ROI_descs['xz'], reduce_fun)

QROI['y->x'] = roi.calc_dataset_ROIs(QROI['y'], ['x'], ROI_descs['x'],
                                   reduce_fun, ROIset_dim_to_combine='yROI')
QROI['z->x'] = roi.calc_dataset_ROIs(QROI['z'], ['x'], ROI_descs['x'],
                                   reduce_fun, ROIset_dim_to_combine='zROI')
QROI['x->y'] = roi.calc_dataset_ROIs(QROI['x'], ['ya'], ROI_descs['y'],
                                   reduce_fun, ROIset_dim_to_combine='xROI')
QROI['z->y'] = roi.calc_dataset_ROIs(QROI['z'], ['ya'], ROI_descs['y'],
                                   reduce_fun, ROIset_dim_to_combine='zROI')
QROI['x->z'] = roi.calc_dataset_ROIs(QROI['x'], ['z'], ROI_descs['z'],
                                   reduce_fun, ROIset_dim_to_combine='xROI')
QROI['y->z'] = roi.calc_dataset_ROIs(QROI['y'], ['z'], ROI_descs['z'],
                                   reduce_fun, ROIset_dim_to_combine='yROI')

QROI['x->y->z'] = roi.calc_dataset_ROIs(QROI['x->y'], ['z'], ROI_descs['z'],
                                   reduce_fun,
                                   ROIset_dim_to_combine=['y', 'x'])

# Save the results
fpath_out = r'H:\WORK\Camilo\TEST\ROI_test\dataset_ROI_test.txt'
with open(fpath_out, 'w') as fid:
    print('\n====  Original  ====', file=fid)
    print(Q, file=fid)
    
    for ROI_name, qROI in QROI.items():
        print(f'\n====  ROI: {ROI_name}  ====', file=fid)
        print(qROI, file=fid)
    
    ROI_names = ['xy', 'y->x', 'x->y->z']
    for ROI_name in ROI_names:
        print(f'\n====  ROI: {ROI_name}  ====', file=fid)
        for var_name, var_data in QROI[ROI_name].data_vars.items():
            print(f'\n==  VAR: {var_name}  ==', file=fid)
            print(var_data, file=fid)
            print('Coordinates:', file=fid)
            for coord_name, coord_vals in var_data.coords.items():
                print(coord_name, file=fid)
                print(coord_vals.values, file=fid)
        
