# -*- coding: utf-8 -*-
"""Tests for ROI reducing functions on DataFileGroup.

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
import data_file_group_2 as dfg
import test_utils as test


ROI_descs = {}

# ROI descriptions (1-st generation)
ROI_descs['y'] = [
        {'name': 'yROI0', 'limits': {'ya': (0, 10)}},
        {'name': 'yROI1', 'limits': {'ya': (1, 10)}},
        ]
ROI_descs['x'] = [
        {'name': 'xROI0', 'limits': {'x': (0, 1)}},
        {'name': 'xROI1', 'limits': {'x': (1, 1)}},
        {'name': 'xROI2', 'limits': {'x': (0, 2)}},
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

# Load root DatFileGroup
fpath_dfg_root = r'D:\WORK\Camilo\TEST\dfg_test\dfg_root_5'
dfg_root = dfg.DataFileGroup()
dfg_root.load(fpath_dfg_root)

dfg_roi = {}
dirpath_dfg_txt = r'D:\WORK\Camilo\TEST\dfg_ROI_test\ROI_test_5'


ROIset_dim_to_combine = None
ROIset_dim_name = None

#ROIset_name = 'xy'
#ROIset_desc = ROI_descs[ROIset_name]

for ROIset_name, ROIset_desc in ROI_descs.items():

    ROI_coords = list(ROIset_desc[0]['limits'].keys())
    fpath_data_column = 'fpath_data_' + ROIset_name + 'ROI'
    fpath_data_postfix = ROIset_name + 'ROI'
    
    # Calculate ROI's
    dfg_roi[ROIset_name] = roi.calc_data_file_group_ROIs(
            dfg_root, ROI_coords, ROIset_desc, test.reduce_fun_strjoin,
            ROIset_dim_to_combine, ROIset_dim_name, fpath_data_column,
            fpath_data_postfix)
    
    # Load datasets, one by one, and print them into a file
    fname_txt = 'dfg_' + ROIset_name + 'ROI.txt'
    fpath_txt = os.path.join(dirpath_dfg_txt, fname_txt)
    test.print_dfg(dfg_roi[ROIset_name], fpath_txt)


# ROI descriptions (2-nd generation)
ROI_descs_2 = {
        'y-x': {'ROIset_old': 'y', 'ROIset_new': 'x', 'ROIdims_old': ['y']},
        'z-x': {'ROIset_old': 'z', 'ROIset_new': 'x', 'ROIdims_old': ['z']},
        'x-y': {'ROIset_old': 'x', 'ROIset_new': 'y', 'ROIdims_old': ['x']},
        'z-y': {'ROIset_old': 'z', 'ROIset_new': 'y', 'ROIdims_old': ['z']},
        'x-z': {'ROIset_old': 'x', 'ROIset_new': 'z', 'ROIdims_old': ['x']},
        'y-z': {'ROIset_old': 'y', 'ROIset_new': 'z', 'ROIdims_old': ['y']},
        'x-y-z': {'ROIset_old': 'x-y', 'ROIset_new': 'z',
                  'ROIdims_old': ['x', 'y']},
        'xy-z': {'ROIset_old': 'xy', 'ROIset_new': 'z',
                 'ROIdims_old': ['x', 'y']}
}

for ROIset_name, ROIset_desc in ROI_descs_2.items():
    
    # Info about the new ROI dimension
    ROIset_name_new = ROIset_desc['ROIset_new']
    ROIset_desc_new = ROI_descs[ROIset_name_new]
    ROI_coords_new = list(ROIset_desc_new[0]['limits'].keys())
    fpath_data_column = f'fpath_data_({ROIset_name})_ROI'
    fpath_data_postfix = f'({ROIset_name})_ROI'
    
    # Info about the old ROI dimension to combine with.
    # The name of the old ROI dimension will be searched by trying all
    # combinations of the given dim names
    ROIset_name_old = ROIset_desc['ROIset_old']
    ROIset_dim_to_combine = ROIset_desc['ROIdims_old']
    
    # Input dataset
    dfg_in = dfg_roi[ROIset_name_old]
    
    # Calculate ROI's
    dfg_roi[ROIset_name] = roi.calc_data_file_group_ROIs(
            dfg_in, ROI_coords_new, ROIset_desc_new, test.reduce_fun_strjoin,
            ROIset_dim_to_combine, ROIset_dim_name, fpath_data_column,
            fpath_data_postfix)
    
    # Load datasets, one by one, and print them into a file
    fname_txt = 'dfg_' + ROIset_name + 'ROI.txt'
    fpath_txt = os.path.join(dirpath_dfg_txt, fname_txt)
    test.print_dfg(dfg_roi[ROIset_name], fpath_txt)
