# -*- coding: utf-8 -*-
"""Provides common functions for processing various types of data

"""


import os

#import pandas as pd
import xarray as xr

import useful as usf


def get_xarrray_dim_by_coord(X, c):
    return X.coords[c].dims[0]

def xarray_coords_to_dict(coords):
    """Convert xarray coords into a dict, as used in Dataset constructor
    
    """
    d = dict()
    for coord_name in coords.keys():
        dim_name = get_xarrray_dim_by_coord(coord_name)
        if coord_name == dim_name:
            d[coord_name] = coords[coord_name].values
        else:
            d[coord_name] = (dim_name, coords[coord_name].values)
    return d

def generate_ROI_names(ROI_coords, ROI_descs):
    """Generate ROI names from coord names and limits
    
    Warning: ROI name generation works fine only for integer coordinates
    """
    ROI_names, ROI_names2 = [], []
    for ROI in ROI_descs:
        ROI_name_2 = 'ROI'
        for coord in ROI_coords:
            ROI_name_2 += f'_({coord}={ROI[coord][0]}-{ROI[coord][1]})'
        ROI_names.append(ROI['name'])
        ROI_names2.append(ROI_name_2)
    return ROI_names, ROI_names2
    

def calc_xarray_ROIs(X_in, ROIset_dim_name, ROI_coords, ROI_descs,
                     reduce_fun = (lambda X,dims: X.mean(dim=dims))):
    """Groups xarray::DataArray content into ROIs over given dimensions
    
    ROI_coords = ('coord1', 'coord2', ...)
    
    ROI_descs = [
        {'name': 'ROIname1', 'coord1': (x1first, x1last), 'coord2': (x2first, x2last), ...} 
        ...
        ]
    
    """

    # Dimensions over which the ROIs are grouped
    ROI_dims = [get_xarrray_dim_by_coord(X_in, coord) for coord in ROI_coords]
    
    # Initialize output dims and coords
    dims_out = list(X_in.dims.keys()).copy()
    coords_out = xarray_coords_to_dict(X_in.coords)
    
    # Exclude output dims and coords that are collapsed into ROIs
    for m, coord in enumerate(ROI_coords):
        usf.list_remove(dims_out, ROI_dims[m])
        usf.dict_remove(coords_out, coord)
        
    # Add output ROI-related dim and coords (num, name, name2)
    dim_new = ROIset_dim_name + '_num'
    dims_out.insert(0, dim_new)
    ROI_names, ROI_names2 = generate_ROI_names(ROI_coords, ROI_descs)
    coords_new = dict()
    coords_new[dim_new] = range(len(ROI_descs))
    coords_new[ROIset_dim_name + '_name']  = (dim_new, ROI_names)
    coords_new[ROIset_dim_name + '_name2'] = (dim_new, ROI_names2)
    coords_out = {**coords_new, **coords_out}
    
    # Create output xarray
    X_out = xr.DataArray(coords=coords_out, dims=dims_out)
    
    # Calculate ROI data
    for n, ROI in enumerate(ROI_descs):
        
        # Build an index
        index = dict()
        for m, coord in enumerate(ROI_coords):
            dim = get_xarrray_dim_by_coord(X_in, coord)
            index[dim] = ((X_in.coords[coord] >= ROI_descs[coord][0]) &
                          (X_in.coords[coord] <= ROI_descs[coord][1]))
            
        # Select input slice by the index, process it, and store to the output
        X_out[dict(dim_new=n)] = reduce_fun(X_in[index], ROI_dims)
        
    # Copy attributes and add info about the operation
    X_out.attrs = X_in.attrs.copy()
    # TODO: attributes
    
    return X_out
                         
