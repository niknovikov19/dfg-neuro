# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:36:11 2021

@author: Nikita
"""

import inspect
import itertools
import os
import re
import sys

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr


def generate_fpath_out(fpath_in, postfix, ext='nc', dirpath_out=None,
                       add_fname_in=True):
    
    if dirpath_out==None:
        dirpath_out = os.path.split(fpath_in)[0]
    fname_in = os.path.split(fpath_in)[1]
    fname_in_noext = os.path.splitext(fname_in)[0]
    if add_fname_in:
        fname_out = '%s_%s.%s' % (fname_in_noext, postfix, ext)
    else:
        fname_out = '%s.%s' % (postfix, ext)
    fpath_out = os.path.join(dirpath_out, fname_out)
    return fpath_out

def parse_chan_name(s):
    
    if isinstance(s, (list, tuple, np.ndarray)):        
        
        d = {'subj_name': [], 'sess_id': [], 'chan_id': []}
        
        for n in range(len(s)):
            lst = s[n].split('_')
            d['subj_name'].append(lst[0])
            d['sess_id'].append(lst[1] + '_' + lst[2])
            d['chan_id'].append(int(lst[3][2:]))
            
    else:

        d = dict()
        
        lst = s.split('_')
        d['subj_name'] = lst[0]
        d['sess_id'] = lst[1] + '_' + lst[2]
        d['chan_id'] = int(lst[3][2:])
        
    return d

# =============================================================================
# def get_chan_id_by_cell(cell_name, cell_info_tbl):
#     ci = cell_info_tbl[cell_info_tbl.cell_name == cell_name]
#     chan_id = ci.chan_id.item()
#     return chan_id
# =============================================================================
    
def get_chan_by_cell(cell_name, cell_info_tbl, chan_info_tbl):
    ci = cell_info_tbl[cell_info_tbl.cell_name == cell_name]
    chan_id = ci.chan_id.item()
    sess_id = ci.sess_id.item()
    subj_name = ci.subj_name.item()
    mask = (chan_info_tbl.chan_id == chan_id) &\
        (chan_info_tbl.sess_id == sess_id) &\
        (chan_info_tbl.subj_name == subj_name)
    chan_name = chan_info_tbl[mask].chan_name.item()
    return chan_name

def get_chan_idx_by_cell_names(cell_names, chan_names, cell_info_tbl,
                               chan_info_tbl):
    Ncell = len(cell_names)
    chan_idx = np.zeros(Ncell, dtype=np.int64)
    for n in range(Ncell):
        cell_name = cell_names[n]
        chan_name = get_chan_by_cell(cell_name, cell_info_tbl, chan_info_tbl)
        chan_idx[n] = np.where(chan_names == chan_name)[0]
    return chan_idx

def load_xarray(fpath_in, unpack=True):
    try:
        X = xr.load_dataset(fpath_in, engine='h5netcdf')
        if unpack:
            X = X['__xarray_dataarray_variable__']
        return X
    except:
        return None

def get_trial_info_by_sess(trial_info, subj_name, sess_id):
    for n in range(len(trial_info)):
        if ((trial_info[n]['subj_name'] == subj_name) and \
            (trial_info[n]['sess_id'] == sess_id)):
            return trial_info[n]
    return None

# Get list of all channels
def get_all_channels(dirpath_root):
    
    col_names =['chan_name', 'subj_name', 'sess_id', 'chan_id', 'fpath_lfp']
    chan_tbl = pd.DataFrame(columns=col_names)
    
    chan_id = 0
    
    for dirpath, dirnames, filenames in os.walk(dirpath_root):
        for filename in filenames:
            if filename == 'lowpass.mat':
                
                fpath_lfp = os.path.join(dirpath, 'lowpass.mat')
    
                dirpath_base, chan_dirname = os.path.split(dirpath)
                dirpath_base, array_dirname = os.path.split(dirpath_base)
                dirpath_base, sess_dirname = os.path.split(dirpath_base)
                dirpath_base, date_dirname = os.path.split(dirpath_base)
                dirpath_base, subj_dirname = os.path.split(dirpath_base)
                
                chan_id = int(re.match('channel([0-9]+)', chan_dirname).groups()[0])
                sess_id_local = int(re.match('session([0-9]+)', sess_dirname).groups()[0])
                date_str = date_dirname
                subj_name = subj_dirname
                
                sess_id = date_str + '_' + str(sess_id_local)
                chan_name = '%s_%s_ch%i' % (subj_name, sess_id, chan_id)
                
                entry = pd.DataFrame([[chan_name, subj_name, sess_id, chan_id, fpath_lfp]],
                                     columns=col_names)
                chan_tbl = chan_tbl.append(entry)
                
                chan_id = chan_id + 1
                
    return chan_tbl

# Get list of channels associated with cells
def get_cell_channels(cell_info):
    
    col_names = ['chan_name', 'subj_name', 'sess_id', 'chan_id', 'fpath_lfp']
    chan_info = pd.DataFrame(columns=col_names)
    
    for n in range(len(cell_info)):
        
        cell = cell_info.iloc[n]
        
        subj_name = cell.subj_name
        sess_id = cell.sess_id
        chan_id = cell.chan_id
        
        chan_name = '%s_%s_ch%i' % (subj_name, sess_id, chan_id)
        
        fpath_spikes = cell.fpath_spikes
        dirpath_lfp = os.path.split(fpath_spikes)[0]
        dirpath_lfp = os.path.split(dirpath_lfp)[0]
        fpath_lfp = os.path.join(dirpath_lfp, 'lowpass.mat')
        
        chan_info_cur = pd.DataFrame([[chan_name, subj_name, sess_id, chan_id, fpath_lfp]],
                                     columns=col_names)
        chan_info = chan_info.append(chan_info_cur)
        
    chan_info = chan_info.drop_duplicates()
        
    return chan_info


def list_remove(lst, elem):
    """ Remove an element from a list without exception if it is missing
    
    """
    if elem in lst:
        lst.remove(elem)
        
def dict_remove(d, key):
    """ Remove a key from a dict without exception if it is missing
    
    """
    if key in d:
        d.pop(key)


def flatten_dict(dictionary, level = []):
    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten_dict(val, level + [key]))
        else:
            tmp_dict['.'.join(level + [key])] = val
    return tmp_dict

def unflatten_dict(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split('.')
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict    


def get_kwargs():
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

def is_scalar(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return False
    else:
        return True

def get_xarrray_dim_by_coord(X, coord_name, coord_vals=None):
    dim_name = X.coords[coord_name].dims[0]
    if coord_vals is None:
        return dim_name
    else:
        if dim_name == coord_name:
            dim_vals = coord_vals
        else:
            sc = is_scalar(coord_vals)
            if sc:
                coord_vals = [coord_vals]
            ind = np.array([np.argwhere(x == X[coord_name].values)[0][0]
                            for x in coord_vals])
            dim_vals = X[dim_name][ind].values
            #mask = [c in coord_vals for c in X[coord_name].values]
            #dim_vals = X[dim_name][mask].data
            if sc:
                dim_vals = dim_vals[0]
        return (dim_name, dim_vals)
    
def get_xarray_coords_dict(X):
    return {name: (c.dims[0], c.values) for name, c in X.coords.items()}


def xarray_select_xr(X, coords):
    index = {}
    for coord_name, coord_vals in coords.items():
        dim_name, dim_vals = get_xarrray_dim_by_coord(X, coord_name, coord_vals)
        index[dim_name] = dim_vals
    return X.loc[index]

def xarray_select(X, coords):
    return xarray_select_xr(X, coords).values


class XrCoordManager:
    
    def __init__(self, X=None):
        self.dims = None
        self.coords = None
        self.coord_dim_map = None
        self.dim_num_map = None
        if X:
            self.set_xr(X)
            
    def set_xr(self, X: xr.Dataset):
        self.dims = list(X.dims)
        self.coords = {}
        self.coord_dim_map = {}
        for coord_name in list(X.coords):
            self.coords[coord_name] = X.coords[coord_name].values
            self.coord_dim_map[coord_name] = (
                     get_xarrray_dim_by_coord(X, coord_name))
        self.dim_num_map = {}
        for n, dim in enumerate(self.dims):
            self.dim_num_map[dim] = n
            
    def get_dim_names(self):
        return self.dims
    
    def get_coord_names(self):
        return list(self.coords.keys())
    
    def get_dim_ranges(self):
        return [range(len(self.coords[dim])) for dim in self.dims]
            
    def get_all_positions(self):
        return itertools.product(*self.get_dim_ranges())
    
    def dims_by_pos(self, pos):
        x = {}
        for n, pos_n in enumerate(pos):
            dim = self.dims[n]
            coord_val = self.coords[dim][pos_n]
            x[dim] = coord_val
        return x
    
    def coords_by_pos(self, pos):
        x = {}
        for coord in self.coords:
            dim = self.coord_dim_map[coord]
            n = self.dim_num_map[dim]
            x[coord] = self.coords[coord][pos[n]]
        return x


def list_wrap(x):
    if isinstance(x, list):
        return x
    else:
        return [x] 
    

# =============================================================================
# def mkdir_rec(dirpath):
#     dirpath = os.path.normpath(dirpath)
#     dir_names = dirpath.split(os.sep)
#     dirpath_cur = ''
#     for dir_name in dir_names:
#         dirpath_cur = os.path.join(dirpath_cur, dir_name)
#         if not os.path.exists(dirpath_cur):
#             sys.m
# =============================================================================

import shutil
from tqdm import tqdm

def copy_files(fpath_list, dirpath_base_old, dirpath_base_new):
    pbar = tqdm(total=len(fpath_list))
    for fpath_in in fpath_list:
        fpath_out = fpath_in.replace(dirpath_base_old, dirpath_base_new)
        dirpath_out, fname_out = os.path.split(fpath_out)
        os.makedirs(dirpath_out)
        shutil.copy(fpath_in, fpath_out)
        pbar.update()
    pbar.close()


def get_dataset_chunks(X: xr.Dataset()):
    chunks = {}
    for var in X.data_vars:
        var_chunks = X[var].chunks
        if var_chunks is None:
            var_chunks_sz = None
        else:
            var_chunks_sz = tuple([chunk[0] for chunk in var_chunks])
        chunks[var] = {'chunksizes': var_chunks_sz}
    return chunks


def create_compatible_xarray(X, dims_excl=None, dims_new=None, dims_rep=None,
                             coords_new=None):
    """ Create array similar to X, with excluded, replaced and added dims."""
    if dims_excl is None:  dims_excl = []
    if dims_new is None:  dims_new = []
    if dims_rep is None:  dims_rep = []
    # Dimensions
    dims_out = [dim for dim in X.dims if dim not in dims_excl]
    for dim_old, dim_new in dims_rep:
        dims_out[dims_out.index(dim_old)] = dim_new
    dims_out += dims_new
    # Coordinates    
    coords_out = get_xarray_coords_dict(X)
    coords_out = {name: coord for name, coord in coords_out.items()
                  if coord[0] not in dims_excl}
    if coords_new is not None:
        coords_out.update(coords_new)
    # Shape
    shape_out = []
    for dim in dims_out:
        coord_vals = [c[1] for c in coords_out.values() if c[0] == dim][0]
        shape_out.append(len(coord_vals))
    # Chunks
    if X.chunks is None:
        chunks_out = None
    else:
        chunks_out = {dim: sz[0] for dim, sz in zip(X.dims, X.chunks)
                      if dim in dims_out}
        for dim in dims_new:
            chunks_out[dim] = -1
        chunks_out = [chunks_out[dim] for dim in dims_out]
    # Create dask-backed DataArray
    Y_ = da.full(shape_out, np.nan, chunks=chunks_out)    
    Y = xr.DataArray(Y_, dims=dims_out, coords=coords_out)
    return Y


def slice_ndarray(x, ind, axis):
    idx = [slice(None)] * x.ndim
    idx[axis] = ind
    return x[tuple(idx)]