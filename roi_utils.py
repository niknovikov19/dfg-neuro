# -*- coding: utf-8 -*-
"""Functions for creating ROI's of xarray.DataArray.

"""

import itertools
import os
import re

from more_itertools import unique_everseen
import numpy as np
#import pandas as pd
from tqdm import tqdm
import xarray as xr

import useful as usf
import data_file_group as dfg


def xarray_coords_to_dict(X, coords):
    """Convert coords into a dict, as used in Dataset constructor."""
    d = dict()
    for coord_name, coord_vals in coords.items():
        dim_name = usf.get_xarrray_dim_by_coord(X, coord_name)
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


def _calc_xarray_ROIs(X_in: xr.DataArray, ROI_coords, ROI_descs,
                     reduce_fun=(lambda X, dims: X.mean(dim=dims)),
                     ROIset_dim_name=None) -> '(xr.DataArray, str)':
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
    
    Returns new DataArray and the name of the created ROI dimenion
    
    """
    
    # Dimensions over which the ROIs are grouped
    ROI_dims = [usf.get_xarrray_dim_by_coord(X_in, coord)
                for coord in ROI_coords]
    
    # Name of the to-be-created ROI dimension
    if ROIset_dim_name is None:
        ROIset_dim_name = ''.join(ROI_dims) + 'ROI'
    
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
                  if usf.get_xarrray_dim_by_coord(X_in, coord_name) not in ROI_dims}
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
    
    # Size of each ROI
    ROI_sz = np.zeros(len(ROI_descs))
    
    # Calculate ROI data
    for n, ROI in enumerate(ROI_descs):
        
        # Build an index
        index = dict()
        for m, coord in enumerate(ROI_coords):
            dim = usf.get_xarrray_dim_by_coord(X_in, coord)
            coord_vals = X_in.coords[coord].values
            index[dim] = (
                (coord_vals >= ROI_descs[n]['limits'][coord][0]) &
                (coord_vals <= ROI_descs[n]['limits'][coord][1]))
            
        # Select input slice by the index, process it, and store to the output
        # TODO: Treat the case if X_in[index] is empty
        sz = X_in[index].size
        if sz == 0:
            raise ValueError('Empty ROI')
        # ROI size: number of collapsing bins
        ROI_sz_vec = np.array([sum(mask) for mask in index.values()])
        ROI_sz[n] = np.prod(ROI_sz_vec)
        X_out[{dim_new: n}] = reduce_fun(X_in[index], ROI_dims)
        
    # Copy attributes and add info about the operation
    X_out.attrs = X_in.attrs.copy()
    # TODO: attributes
    
    return X_out, ROIset_dim_name, ROI_sz


def combine_xarray_dimensions(X_in: xr.DataArray, dims_to_combine,
                              dim_name_new, coord_names_new=None,
                              coord_val_generator=None) -> xr.DataArray:
    
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


def calc_xarray_ROIs(X_in: xr.DataArray, ROI_coords, ROI_descs,
                     reduce_fun=(lambda X, dims: X.mean(dim=dims)),
                     ROIset_dim_to_combine=None,
                     ROIset_dim_name=None) -> xr.DataArray:
    """Perform _calc_xarray_ROIs() + combine_xarray_dimensions()."""
    
    X_out, _ROIset_dim_name, ROI_sz = _calc_xarray_ROIs(
            X_in, ROI_coords, ROI_descs, reduce_fun, ROIset_dim_name)
    
    if ROIset_dim_to_combine is not None:
        r = re.compile('(.+)ROI')
        dim_name_base_1 = r.findall(_ROIset_dim_name)[0]
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
        
    return X_out, {'ROIset_dim_name': _ROIset_dim_name, 'ROI_sz': ROI_sz}


def get_combinations(lst: list):
    comb_lst = []
    for n in range(1, len(lst)+1):
        comb_lst += list(itertools.combinations(lst, n))
    comb_lst_all = []
    for comb in comb_lst:
        comb_lst_all += list(itertools.permutations(comb))
    return comb_lst_all
    
        
def calc_dataset_ROIs(X_in: xr.Dataset, ROI_coords, ROI_descs,
                     reduce_fun=(lambda X, dims: X.mean(dim=dims)),
                     ROIset_dim_to_combine=None, ROIset_dim_name=None,
                     var_renamings=None, preproc_fun=None,
                     add_ROIsz_vars=False) -> xr.Dataset:
    """Calculate ROIs, optionally merge new ROI dimension with an old one."""
    # Preprocess the dataset
    if preproc_fun is not None:
        X_in = preproc_fun(X_in)
        
    ROI_sz_info = {}
    
    X_out_vars = {}
    for var_name, X_in_var in X_in.data_vars.items():        
        # Keep in the ROI desc only those coordinates that the current
        # variable does have
        coords_var = list(X_in_var.coords.keys())
        ROI_coords_var = [ROI_coord for ROI_coord in ROI_coords
                          if ROI_coord in coords_var]
        ROI_descs_var = []
        ROI_limits_list = []
        for ROI_desc in ROI_descs:
            ROI_limits = {
                    coord_name: coord_limits
                    for coord_name, coord_limits in ROI_desc['limits'].items()
                    if coord_name in coords_var}
            ROI_desc_var = {'name': ROI_desc['name'], 'limits': ROI_limits}
            ROI_descs_var.append(ROI_desc_var)
            ROI_limits_list.append(tuple(ROI_limits.values()))
            
        # Keep unique ROIs
        ROI_limits_unique = list(unique_everseen(ROI_limits_list))
        ROI_idx_unique = [ROI_limits_list.index(x) for x in ROI_limits_unique]
        ROI_descs_var = [ROI_descs_var[i] for i in ROI_idx_unique]
        
        # - If an old ROI dimension to combinate is given by its name - check
        # whether the current variable has this dimension, if not - do nothing
        # - If the old ROI dimension is given by the list of its constituent
        # dimensions, produce all possible ROI names from their combinations
        # and search for the one that belongs to the current variable dims
        dims_var = list(X_in_var.dims)
        ROIset_dim_to_combine_var = ROIset_dim_to_combine
        if ROIset_dim_to_combine_var is not None:
            # Old ROI dimension is given by its name
            if type(ROIset_dim_to_combine_var) == str:
                if (ROIset_dim_to_combine_var + '_num') not in coords_var:
                    ROIset_dim_to_combine_var = None
            # Old ROI dimension is given by its costituent dimensions
            else:
                old_ROIset_coords = list(ROIset_dim_to_combine_var)
                old_ROIset_coord_combinations = (
                        get_combinations(old_ROIset_coords))
                old_ROIset_name_variants = [
                        ''.join(coord_comb) + 'ROI_num' 
                        for coord_comb in old_ROIset_coord_combinations]
                old_ROIset_names = [ROI_name
                                   for ROI_name in old_ROIset_name_variants
                                   if ROI_name in dims_var]
                if len(old_ROIset_names) == 1:
                    ROIset_dim_to_combine_var = old_ROIset_names[0]
                elif len(old_ROIset_names) == 0:
                    ROIset_dim_to_combine_var = None
                else:
                    raise ValueError('More than one name is suitable for the'
                                     'to-be-combinated ROI dimension')
                    
        # New variable name
        var_name_new = var_name
        if var_renamings is not None:
            if var_name in var_renamings:
                var_name_new = var_renamings[var_name]
                
        # ROI-reducing function
        if isinstance(reduce_fun, dict):
            reduce_fun_cur = reduce_fun[var_name]
        else:
            reduce_fun_cur = reduce_fun            
                
        # Calculate ROIs and combinate with an old ROI dimension (if needed)
        # If no dimensions from the ROI desc are present in the current
        # variable - just copy the variable to the output
        if len(ROI_coords_var) != 0:
            X_out_vars[var_name_new], _ROI_sz_info = calc_xarray_ROIs(
                    X_in_var, ROI_coords_var, ROI_descs_var,
                    reduce_fun_cur, ROIset_dim_to_combine_var, ROIset_dim_name)
            # Store ROI size vector
            dim_name = _ROI_sz_info['ROIset_dim_name']
            if dim_name in ROI_sz_info:
                if np.any(ROI_sz_info[dim_name] != _ROI_sz_info['ROI_sz']):
                    raise ValueError('ROI dimensions with the same name but'
                                     'different ROI size vectors')
            ROI_sz_info[dim_name] = _ROI_sz_info['ROI_sz']
        else:
            X_out_vars[var_name_new] = X_in_var.copy()
            
    # Add variables for ROI sizes
    if add_ROIsz_vars:
        for dim_name, ROI_sz in ROI_sz_info.items():
            var_name = 'ROI_sz_' + dim_name
            dim_out = dim_name + '_num'
            coords_out = {dim_out: np.arange(len(ROI_sz))}
            Y = xr.DataArray(ROI_sz.astype(np.int), coords=coords_out,
                             dims=[dim_out])            
            X_out_vars[var_name] = Y
    
    # TODO: Modify ROI dim names if not all to-be-collapsed dimensions
    # were present in the input Dataset, so all output variables have
    # compatible dimensions (i.e. dims with the same name have the same
    # coord values). Now only ROIset_dim_name=None works??
    X_out = xr.Dataset(X_out_vars)
    return X_out
        
        
# TODO: modify ROI naming in such way that an individual name is given for
# each interval by each coordinate
# Old:
#ROI_descs['xy'] = [
#        {'name': 'xyROI0', 'limits': {'x': (0, 1), 'ya': (0,10)}},
#        {'name': 'xyROI1', 'limits': {'x': (1, 1), 'ya': (0,10)}},
#        {'name': 'xyROI3', 'limits': {'x': (0, 1), 'ya': (1,10)}},
#        {'name': 'xyROI4', 'limits': {'x': (1, 1), 'ya': (1,10)}}]
# New:
#ROI_descs['xy'] = [
#        [{'x':  {'name': 'xROI0', 'limits': (0, 1)},
#         {'ya': {'name': 'yROI0', 'limits': (0,10)}],
#        [{'x':  {'name': 'xROI1', 'limits': (1, 1)},
#         {'ya': {'name': 'yROI0', 'limits': (0,10)}],    
#        [{'x':  {'name': 'xROI0', 'limits': (0, 1)},
#         {'ya': {'name': 'yROI1', 'limits': (1,10)}],
#        [{'x':  {'name': 'xROI1', 'limits': (1, 1)},
#         {'ya': {'name': 'yROI1', 'limits': (1,10)}]]
# This would allow correct ROI naming for dataset variables that do not have
# all the dimensions given in the ROI description

# TODO: Raise error if ROI coord is not present


def reduce_fun_mean(X, dims):
    return X.mean(dim=dims)


def calc_data_file_group_ROIs(dfg_in: dfg.DataFileGroup,
                              ROI_coords, ROI_descs,
                              reduce_fun=reduce_fun_mean,
                              ROIset_dim_to_combine=None,
                              ROIset_dim_name=None,
                              fpath_data_column=None,
                              fpath_data_postfix=None,
                              var_renamings=None,
                              coords_new_descs=None,
                              preproc_fun=None,
                              add_ROIsz_vars=False) -> dfg.DataFileGroup:

    proc_step_name = 'ROI calculation (%s)' % ', '.join(ROI_coords)
    
    # Dictionary of parameters
    param_names = ['ROI_coords', 'ROI_descs', 'reduce_fun', 'preproc_fun',
                   'ROIset_dim_to_combine', 'ROIset_dim_name', 'add_ROIsz_vars']
    local_vars = locals()
    params = {par_name: local_vars[par_name] for par_name in param_names}
    
    # Variable renamings (old name -> new name) for calc_dataset_ROIs()
    # and descriptions of the renamed variables for dfg.apply_dfg_inner_proc()
    vars_new_descs = None
    if var_renamings is not None:
        params['var_renamings'] = {var_name_old: var_new['name']
                               for var_name_old, var_new in var_renamings.items()}
        vars_new_descs = {var_new['name']: var_new['desc']
                          for var_name_old, var_new in var_renamings.items()}
    
    # Name of the dfg's outer table column for the paths to Dataset files
    if fpath_data_column is None:
        ROI_str = ''.join(ROI_coords) + 'ROI'
        fpath_data_column = dfg_in.get_data_desc()['fpath_data_column']
        fpath_data_column += ROI_str
        
    # Postfix added to the output inner data files
    if fpath_data_postfix is None:
        fpath_data_postfix = ''.join(ROI_coords) + 'ROI'

    # Function that converts the parameters dict to the form suitable
    # for storing into a processing step description
    def gen_proc_step_params(par):
        # Reducing function
        if isinstance(reduce_fun, dict):
            reduce_fun_str = ''
            for var_name, fun in reduce_fun.items():
                reduce_fun_str += f'{var_name}: {fun.__name__}, '
        else:
            reduce_fun_str = par['reduce_fun'].__name__
        # Preprocessing function
        if par['preproc_fun'] is not None:
            preproc_fun_str = par['preproc_fun'].__name__
        else:
            preproc_fun_str = str(None)
        # Params
        par_out = {
            'ROI_coords': {
                'desc': 'Coordinates to collapse into ROIs',
                'value': par['ROI_coords']},
            'ROI_descs': {
                'desc': 'Names and coordinate ranges of the ROIs',
                'value': [str(d) for d in par['ROI_descs']]},
            'reduce_fun': {
                'desc': ('Function(s) for converting input values that belong '
                         'to a ROI into a sinle output value'),
                'value': reduce_fun_str},
            'preproc_fun': {
                'desc': ('Preprocessing function'),
                'value': preproc_fun_str},
            'ROIset_dim_to_combine': {
                'desc': 'Old ROI dimension to combine the result with',
                'value': str(par['ROIset_dim_to_combine'])},
            'ROIset_dim_name': {
                'desc': 'Name of the new ROI dimension',
                'value': str(par['ROIset_dim_name'])}
        }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath(fpath_in, params):
        fpath_noext, ext  = os.path.splitext(fpath_in)
        return fpath_noext + '_' + fpath_data_postfix + ext
    
    # Call calc_dataset_ROIs() for each inner dataset of the DataFileGroup
    dfg_out = dfg.apply_dfg_inner_proc(
            dfg_in, calc_dataset_ROIs, params, proc_step_name,
            gen_proc_step_params, fpath_data_column, gen_fpath,
            vars_new_descs, coords_new_descs)
    
    return dfg_out
    

    