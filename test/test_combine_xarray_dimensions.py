# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:02:16 2021

@author: Nikita
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


# Create test 3-d DataArray's that contains in a cell with the
# coordinates (i,j,k) a number equal to (100*k + 10*j + i) or
# its string represresentation

def lst_encode(lst, k=1):
    """[i, j, k] -> (100k + 10*j + i) """
    res = 0
    for n, x in enumerate(lst):
        res += (x * (10**k)**n)
    return res

dims = ['xROI_num', 'yROI_num', 'zROI_num']
sz = [3, 2, 4]
coords = {
        'xROI_num': [0, 2, 4],
        'yROI_num': [1, 3],
        'zROI_num': [5, 6, 7, 8],
        'xROI_name': ('xROI_num',
                      ['xROI0', 'xROI1', 'xROI2']),
        'yROI_name': ('yROI_num',
                      ['yROI0', 'yROI1']),
        'zROI_name': ('zROI_num',
                      ['zROI0', 'zROI1', 'zROI2', 'zROI3']),
        'xROI_name2': ('xROI_num',
                       ['xROI_(0-0)', 'xROI_(2-20)', 'xROI_(4-40)']),
        'yROI_name2': ('yROI_num',
                       ['yROI_(1-10)', 'yROI_(3-30)']),
        'zROI_name2': ('zROI_num',
                       ['zROI_(5-50)', 'zROI_(6-60)', 'zROI_(7-70)', 'zROI_(8-80)'])
        }

x = np.ndarray(sz, dtype=np.int)
dim_coords = [list(range(dim_len)) for dim_len in sz]
idx = itertools.product(*dim_coords)
for id in idx:
    x[id] = lst_encode(list(id))
    
X = xr.DataArray(x, coords=coords, dims=dims)    


# =============================================================================
# # Test 1
# 
# dims_to_combine = ['xROI_num', 'yROI_num']
# dim_name_new = 'xyROI_num'
# coord_names_new = ['xyROI_num']
# 
# Y = roi.combine_xarray_dimensions(X, dims_to_combine=dims_to_combine,
#                                   dim_name_new=dim_name_new,
#                                   coord_names_new=coord_names_new,
#                                   coord_val_generator=None)
# 
# print('======= TEST 1 =======\n\n')
# print(f'{X}\n\n{Y}\n\n')
# =============================================================================

# Test coord_val_generator()
# =============================================================================
# coord_vals_comb = {
#     'xdim': {
#         'xROI_num': 2,
#         'xROI_name': 'xROI2',
#         'xROI_name2': 'xROI_(2-20)'                
#     },
#     'ydim': {
#         'yROI_num': 3,
#         'yROI_name': 'yROI3',
#         'yROI_name2': 'yROI_(3-30)'                
#     }
# }
# comb_num = 10
# coord_names_out = ['xyROI_num', 'xyROI_name', 'xyROI_name2']
# res = coord_val_generator(coord_vals_comb, comb_num, coord_names_out)
# from pprint import pprint
# pprint(res)
# =============================================================================


# Test 2

test_data_lst = [
    {
        'dims_to_combine': ['xROI_num', 'yROI_num'],
        'dim_name_new': 'xyROI_num',
        'coord_names_new': ['xyROI_num', 'xyROI_name', 'xyROI_name2']
    },
    {
        'dims_to_combine': ['xROI_num', 'zROI_num'],
        'dim_name_new': 'xzROI_num',
        'coord_names_new': ['xzROI_num', 'xzROI_name', 'xzROI_name2']
    },
    {
        'dims_to_combine': ['yROI_num', 'zROI_num'],
        'dim_name_new': 'yzROI_num',
        'coord_names_new': ['yzROI_num', 'yzROI_name', 'yzROI_name2']
    },
    {
        'dims_to_combine': ['xROI_num', 'yROI_num', 'zROI_num'],
        'dim_name_new': 'xyzROI_num',
        'coord_names_new': ['xyzROI_num', 'xyzROI_name', 'xyzROI_name2']
    },
    {
        'dims_to_combine': ['xROI_num'],
        'dim_name_new': 'xROI_num',
        'coord_names_new': ['xROI_num', 'xROI_name', 'xROI_name2']
    },
    {
        'dims_to_combine': ['yROI_num'],
        'dim_name_new': 'yROI_num',
        'coord_names_new': ['yROI_num', 'yROI_name', 'yROI_name2']
    }
]

# TODO: Test reverse dim order

for test_num, test_data in enumerate(test_data_lst):

    Y = roi.combine_xarray_dimensions(
            X, dims_to_combine=test_data['dims_to_combine'],
            dim_name_new=test_data['dim_name_new'],
            coord_names_new=test_data['coord_names_new'],
            coord_val_generator=roi.coord_val_generator_ROI)

    print(f'======= TEST 2.{test_num+1} =======\n\n')
    print(f'{X}\n\n{Y}\n\n')
    print(Y.coords[test_data['coord_names_new'][1]].values)  # name
    print('')
    print(Y.coords[test_data['coord_names_new'][2]].values)  # name2
    print('\n')






