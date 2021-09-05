# -*- coding: utf-8 -*-
"""Provides common functions for processing various types of data.

"""


#import os

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
    

def calc_xarray_ROIs(X_in, ROIset_dim_name, ROI_coords, ROI_descs,
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
                         
