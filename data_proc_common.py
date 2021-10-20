# -*- coding: utf-8 -*-
"""Provides common functions for processing various types of data.

"""


import itertools
#import os
import re

import numpy as np
#import pandas as pd
import xarray as xr

#import useful as usf


def get_xarrray_dim_by_coord(X, coord):
    return X.coords[coord].dims[0]

def xarray_coords_to_dict(X, coords):
    """Convert coords into a dict, as used in Dataset constructor."""
    d = dict()
    for coord_name, coord_vals in coords.items():
        dim_name = get_xarrray_dim_by_coord(X, coord_name)
        if coord_name == dim_name:
            d[coord_name] = coord_vals
        else:
            d[coord_name] = (dim_name, coord_vals)
    return d

def generate_ROI_names(ROI_coords, ROI_descs):
    """Generate ROI names from coord names and limits
    
    Warning: ROI name generation works fine only for integer coordinates.
    """
    ROI_names, ROI_names2 = [], []
    for ROI in ROI_descs:
        ROI_name_2 = 'ROI'
        for coord in ROI_coords:
            coord_val_first = ROI['limits'][coord][0]
            coord_val_last = ROI['limits'][coord][1]
            ROI_name_2 += f'_({coord}={coord_val_first}-{coord_val_last})'
        ROI_names.append(ROI['name'])
        ROI_names2.append(ROI_name_2)
    return ROI_names, ROI_names2


def _calc_xarray_ROIs(X_in, ROIset_dim_name, ROI_coords, ROI_descs,
                     reduce_fun=(lambda X, dims: X.mean(dim=dims))):
    """Groups xarray::DataArray content into ROIs over given dimensions.
    
    ROI_coords = ('coord1', 'coord2', ...)
    
    ROI_descs = [
        {
            'name': 'ROIname1',
            'limits': {
                'coord1': (x1first, x1last),
                'coord2': (x2first, x2last),
                ...
            }
        }
        ...
    ]
    
    """

    # Dimensions over which the ROIs are grouped
    ROI_dims = [get_xarrray_dim_by_coord(X_in, coord) for coord in ROI_coords]
    
    # Select dimensions not covered by ROI
    # dims_out - names of the selected dimension
    # dim_nums_out - positions of the selected dimensions in the list
    # of all input dimensions
    # shape_out - shape list without elements corresponding to ROI dimensions
    dim_data_out = [(dim_num, dim_name)
                for dim_num, dim_name in enumerate(X_in.dims)
                if dim_name not in ROI_dims]
    if dim_data_out:
        dim_nums_out, dims_out = zip(*dim_data_out)
    else:
        dim_nums_out, dims_out = [], []
    shape_out = [X_in.shape[n] for n in dim_nums_out]
    
    # Select coordinates not covered by ROI
    coords_out = {coord_name: coord_val.values
                  for coord_name, coord_val in X_in.coords.items()
                  if get_xarrray_dim_by_coord(X_in, coord_name) not in ROI_dims}
    coords_out = xarray_coords_to_dict(X_in, coords_out)

    # Add a new first dimension that corresponds to ROI numbers
    dim_new = ROIset_dim_name + '_num'
    dims_out = list(dims_out)
    dims_out.insert(0, dim_new)
    shape_out = [len(ROI_descs)] + shape_out
    
    # Add new coordinates corresponding to the new 'ROI-number' dimension
    # 1. Contains ROI numbers (0, 1, 2, ...)
    # 2. Contains ROI names given from the outside
    # 3. Contains ROI names that are generated from the coordinate names
    # and limits that define a ROI
    ROI_names, ROI_names2 = generate_ROI_names(ROI_coords, ROI_descs)
    coords_new = dict()
    coords_new[dim_new] = np.arange(len(ROI_descs))
    coords_new[ROIset_dim_name + '_name']  = (dim_new, ROI_names)
    coords_new[ROIset_dim_name + '_name2'] = (dim_new, ROI_names2)
    coords_out = {**coords_new, **coords_out}
    
    # Create output xarray
    x_out = np.ndarray(shape=shape_out, dtype=X_in.dtype)
    X_out = xr.DataArray(x_out, coords=coords_out, dims=dims_out)
    
    # Calculate ROI data
    for n, ROI in enumerate(ROI_descs):
        
        # Build an index
        index = dict()
        for m, coord in enumerate(ROI_coords):
            dim = get_xarrray_dim_by_coord(X_in, coord)
            coord_vals = X_in.coords[coord].values
            index[dim] = (
                (coord_vals >= ROI_descs[n]['limits'][coord][0]) &
                (coord_vals <= ROI_descs[n]['limits'][coord][1]))
            
        # Select input slice by the index, process it, and store to the output
        X_out[{dim_new: n}] = reduce_fun(X_in[index], ROI_dims)
        
    # Copy attributes and add info about the operation
    X_out.attrs = X_in.attrs.copy()
    # TODO: attributes
    
    return X_out


def combine_xarray_dimensions(X_in, dims_to_combine, dim_name_new,
                              coord_names_new=None, coord_val_generator=None):
    
    # Mapping between the input dimensions and coordinates
    dim_to_coord_map = {}
    for dim_name in X_in.dims:
        dim_to_coord_map[dim_name] = [
            coord_name for coord_name, coord_data in X_in.coords.items()
            if coord_data.dims[0] == dim_name]

    # Spared dimensions: those that should not be combined
    # dims_spared - names of the spared dimension
    # dim_nums_spared - positions of the spared dimensions in the list
    # of all input dimensions
    # shape_spared - shape list corresponding to the spared dimensions
    dim_data_spared = [(dim_num, dim_name)
                for dim_num, dim_name in enumerate(X_in.dims)
                if dim_name not in dims_to_combine]
    if dim_data_spared:
        dim_nums_spared, dims_spared = zip(*dim_data_spared)
    else:
        dim_nums_spared, dims_spared = [], []
    shape_spared = [X_in.shape[n] for n in dim_nums_spared]
    
    # Spared coordinates: those that correspond to the spared dimensions
    coords_names_spared = [
        coord_names
        for dim_name, coord_names in dim_to_coord_map.items()
        if dim_name not in dims_to_combine
    ]
    # Unroll the coord names into a single list
    coord_names_spared = list(itertools.chain(*coords_names_spared))
    coords_spared = {coord_name: X_in.coords[coord_name].values
                     for coord_name in coord_names_spared}
    # Prepare the coords dict in the form ised in the DataArray constructor
    coords_spared = xarray_coords_to_dict(X_in, coords_spared)
    
    # Combinations of the coordinate values along the combinated dimensions
    # Coordinates within a combination are in the same order as the
    # dimensions in 'dim_name_new' argument
    # Coordinate values are 0, 1, 2, ...
    combined_shape = []
    for dim_name in dims_to_combine:
        dim_num = X_in.dims.index(dim_name)
        combined_shape.append(X_in.shape[dim_num])
    combined_ranges = [np.arange(n) for n in combined_shape]
    coord_combinations = list(itertools.product(*combined_ranges))

    # Size by the new (combined) dimension
    dim_new_sz = len(coord_combinations)

    # New (combined) coordinates
    if coord_names_new is None:
        coord_names_new = [dim_name_new]
    if dim_name_new not in coord_names_new:
        raise ValueError('One of the new coordinate names must be the same '
                         'as the new dimension name')
    coords_new = {coord_name: [] for coord_name in coord_names_new}
    if coord_val_generator is None:
        if len(coord_names_new) != 1:
            raise ValueError('There should be exactly one new coordinate '
                             'if no coord_val_generator() is provided')
        coords_new[coord_names_new[0]] = np.arange(dim_new_sz)
    else:
        for comb_num, coord_combination in enumerate(coord_combinations):
            coord_vals_comb = {}
            for dim_num, coord_val_num in enumerate(coord_combination):
                dim_name = dims_to_combine[dim_num]
                coord_vals_comb[dim_name] = {}
                for coord_name in dim_to_coord_map[dim_name]:
                    coord_vals_comb[dim_name][coord_name] = (
                            X_in.coords[coord_name].values[coord_val_num])
            coord_vals_new = coord_val_generator(coord_vals_comb,
                                                 comb_num,
                                                 coord_names_new)
            for coord_name in coord_names_new:
                coords_new[coord_name].append(coord_vals_new[coord_name])
        for coord_name, coord_vals in coords_new.items():
            if coord_name != dim_name_new:
                coords_new[coord_name] = (dim_name_new, coord_vals)
                
    # Output dimension and coordinates
    dims_out = [dim_name_new] + list(dims_spared)
    coords_out = {**coords_new, **coords_spared}
    
    # Allocate the output            
    shape_out = [dim_new_sz] + shape_spared
    x_out = np.ndarray(shape_out, dtype=X_in.dtype)
    X_out = xr.DataArray(x_out, coords=coords_out, dims=dims_out)
    
    # Fill the output
    for n, coord_combination in enumerate(coord_combinations):
        index_in = {dim_name: coord_combination[m]
                    for m, dim_name in enumerate(dims_to_combine)}
        X_out[{dim_name_new: n}] = X_in[index_in]
        
    # Copy attributes
    X_out.attrs = X_in.attrs.copy()
    # TODO: attributes
    
    return X_out


def coord_val_generator_ROI(coord_vals_comb, comb_num, coord_names_out):
    """ Combine ROI names, to use in combine_xarray_dimensions().
    
    Example:        
        coord_vals_comb = {
            'xdim': {
                'xROI_num': 2,
                'xROI_name': 'xROI2',
                'xROI_name2': 'xROI_(2-20)'                
            },
            'ydim': {
                'yROI_num': 3,
                'yROI_name': 'yROI3',
                'yROI_name2': 'yROI_(3-30)'                
            }
        }
            
        comb_num = 10
        
        coord_names_out = ['xyROI_num', 'xyROI_name', 'xyROI_name2']
        
        output = {
            'xyROI_num': 10,
            'xyROI_name': 'xROI2_yROI3',
            'xyROI_name2': 'xyROI_(2-20)_(3-30)'
        }
    """

    # Keep together input and output coordinates for similar processing
    dim_names_in = list(coord_vals_comb.keys())
    dim_names = dim_names_in + ['OUTPUT']
    
    # Types of the coordinates. Each type corresponds to a postfix of 
    # coordinate name. Each type is processed in its own way.
    coord_types = ['num', 'name', 'name2']
    
    # For each dimension (including the 'OUTPUT' dimension),
    # find the coordinate name for each coodinate type
    coords_by_type = {}
    for dim_name in dim_names:
        # Initialize the info for a given dimension
        coords_by_type[dim_name] = {
            coord_type: None for coord_type in coord_types
        }
        # Will search either among the output or input coordinates
        if dim_name == 'OUTPUT':
            coord_names = coord_names_out
        else:
            coord_names = coord_vals_comb[dim_name]
        # Search for each coordinate type among given coordinates
        for coord_name in coord_names:
            for coord_type in coord_types:
                # If a match is found - store the full coordinate name,
                # and a name without the type-denoting postfix
                if coord_name.endswith(coord_type):
                    postfix = '_' + coord_type
                    coord_name_prefix = coord_name[:-len(postfix)]
                    coords_by_type[dim_name][coord_type]= (
                            coord_name, coord_name_prefix)
    
    name_coord_vals_in = []
    name2_coord_vals_in = []
    for dim_name in dim_names_in:
        # Get 'name' coordinate value
        coord_name = coords_by_type[dim_name]['name'][0]
        coord_val = coord_vals_comb[dim_name][coord_name]
        name_coord_vals_in.append(coord_val)
        # Get 'name2' coordinate, without a prefix
        coord_name = coords_by_type[dim_name]['name2'][0]
        coord_prefix = coords_by_type[dim_name]['name2'][1]
        coord_val = coord_vals_comb[dim_name][coord_name]
        coord_val = coord_val.rpartition(coord_prefix + '_')[-1]
        name2_coord_vals_in.append(coord_val)
    
    coord_vals_out = {}       
    # Set output 'num' coordinate value
    coord_name = coords_by_type['OUTPUT']['num'][0]
    coord_vals_out[coord_name] = comb_num    
    # Set output 'name' coordinate value
    coord_name = coords_by_type['OUTPUT']['name'][0]
    coord_val = '_'.join(name_coord_vals_in)
    coord_vals_out[coord_name] = coord_val
    # Set output 'name2' coordinate value
    coord_name = coords_by_type['OUTPUT']['name2'][0]
    coord_prefix = coords_by_type['OUTPUT']['name2'][1]
    coord_val = coord_prefix + '_' + '_'.join(name2_coord_vals_in)
    coord_vals_out[coord_name] = coord_val
    
    return coord_vals_out


def coord_val_generator_ROI_2(coord_vals_comb, comb_num, coord_names_out):
    """ Combine ROI names, to use in combine_xarray_dimensions().
    
    Example:        
        coord_vals_comb = {
            'xdim': {
                'xROI_num': 2,
                'xROI_name': 'xROI2',
                'xROI_name2': 'ROI_(x=2-20)'                
            },
            'ydim': {
                'yROI_num': 3,
                'yROI_name': 'yROI3',
                'yROI_name2': 'ROI_(y=3-30)'                
            }
        }
            
        comb_num = 10
        
        coord_names_out = ['xyROI_num', 'xyROI_name', 'xyROI_name2']
        
        output = {
            'xyROI_num': 10,
            'xyROI_name': 'xROI2_yROI3',
            'xyROI_name2': 'ROI_(x=2-20)_(y=3-30)'
        }
    """

    # Keep together input and output coordinates for similar processing
    dim_names_in = list(coord_vals_comb.keys())
    dim_names = dim_names_in + ['OUTPUT']
    
    # Types of the coordinates. Each type corresponds to a postfix of 
    # coordinate name. Each type is processed in its own way.
    coord_types = ['num', 'name', 'name2']
    
    # For each dimension (including the 'OUTPUT' dimension),
    # find the coordinate name for each coodinate type
    coords_by_type = {}
    for dim_name in dim_names:
        # Initialize the info for a given dimension
        coords_by_type[dim_name] = {
            coord_type: None for coord_type in coord_types
        }
        # Will search either among the output or input coordinates
        if dim_name == 'OUTPUT':
            coord_names = coord_names_out
        else:
            coord_names = coord_vals_comb[dim_name]
        # Search for each coordinate type among given coordinates
        for coord_name in coord_names:
            for coord_type in coord_types:
                # If a match is found - store the full coordinate name,
                # and a name without the type-denoting postfix
                if coord_name.endswith(coord_type):
                    postfix = '_' + coord_type
                    coord_name_prefix = coord_name[:-len(postfix)]
                    coords_by_type[dim_name][coord_type]= (
                            coord_name, coord_name_prefix)
    
    name_coord_vals_in = []
    name2_coord_vals_in = []
    for dim_name in dim_names_in:
        # Get 'name' coordinate value
        coord_name = coords_by_type[dim_name]['name'][0]
        coord_val = coord_vals_comb[dim_name][coord_name]
        name_coord_vals_in.append(coord_val)
        # Get 'name2' coordinate, without a prefix
        coord_name = coords_by_type[dim_name]['name2'][0]
        #coord_prefix = coords_by_type[dim_name]['name2'][1]
        coord_val = coord_vals_comb[dim_name][coord_name]
        #coord_val = coord_val.rpartition(coord_prefix + '_')[-1]
        coord_val = re.compile('ROI_(.+)').findall(coord_val)[0]
        name2_coord_vals_in.append(coord_val)
    
    coord_vals_out = {}       
    # Set output 'num' coordinate value
    coord_name = coords_by_type['OUTPUT']['num'][0]
    coord_vals_out[coord_name] = comb_num    
    # Set output 'name' coordinate value
    coord_name = coords_by_type['OUTPUT']['name'][0]
    coord_val = '_'.join(name_coord_vals_in)
    coord_vals_out[coord_name] = coord_val
    # Set output 'name2' coordinate value
    coord_name = coords_by_type['OUTPUT']['name2'][0]
    #coord_prefix = coords_by_type['OUTPUT']['name2'][1]
    coord_val = 'ROI_' + '_'.join(name2_coord_vals_in)
    coord_vals_out[coord_name] = coord_val
    
    return coord_vals_out


def calc_xarray_ROIs(X_in, ROIset_dim_name, ROI_coords, ROI_descs,
                     reduce_fun=(lambda X, dims: X.mean(dim=dims)),
                     ROIset_dim_to_combine=None):
    """Perform _calc_xarray_ROIs() + combine_xarray_dimensions()."""
    
    X_out = _calc_xarray_ROIs(X_in, ROIset_dim_name, ROI_coords, ROI_descs,
                              reduce_fun)
    
    if ROIset_dim_to_combine is not None:
        r = re.compile('(.+)ROI')
        dim_name_base_1 = r.findall(ROIset_dim_name)[0]
        dim_name_base_2 = r.findall(ROIset_dim_to_combine)[0]
        dim_name_base_new = dim_name_base_1 + dim_name_base_2
        dims_to_combine = [dim_name_base_1 + 'ROI_num',
                           dim_name_base_2 + 'ROI_num']
        dim_name_new = dim_name_base_new + 'ROI_num'
        coord_names_new = [dim_name_base_new + 'ROI_num',
                           dim_name_base_new + 'ROI_name',
                           dim_name_base_new + 'ROI_name2']
        X_out = combine_xarray_dimensions(X_out, dims_to_combine, dim_name_new,
                              coord_names_new, coord_val_generator_ROI_2)
        
    return X_out
    
        
        
        
        

    
    
    
    