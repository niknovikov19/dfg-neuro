# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 18:36:11 2021

@author: Nikita
"""

import inspect
import os
import re

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


def get_xarrray_dim_by_coord(X, coord):
    return X.coords[coord].dims[0]


def xarray_select(X, coords):
    for coord_name, coord_val in coords.items():
        X = X[X[coord_name] == coord_val]
        dim = get_xarrray_dim_by_coord(X, coord_name)
        X = X.squeeze(dim=dim, drop=True)
    return X
