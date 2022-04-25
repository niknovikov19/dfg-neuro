# -*- coding: utf-8 -*-


import os

#import pandas as pd
import pickle
#from pprint import pprint
from tqdm import tqdm
#import xarray as xr

from data_file_group_2 import DataFileGroup
import useful as usf


branch_id = '1'

proc_steps = {
    '0': {
        'name': 'LFP data',
        'function': '',
        'data_desc_out':  {
            'variables': {'LFP': 'LFP amplitude'},
            'outer_dims': ['chan_name'],
            'outer_coords': {
                'chan_name': 'Subject + session + channel',
            },
            'inner_dims': ['sample_num'],
            'inner_coords': {
                'sample_num': 'Sample number',
                'time': 'Time, s'
            },
            'fpath_data_column': 'fpath_lfp',
        },
        'params': {
            'fs': {
                'desc': 'Sampling rate',
                'value': 1000},
        }
    },
    '1': {
        'name': 'Epoching',
        'function': 'epoch_lfp_data_batch()',
        'data_desc_out':  {
            'variables': {'LFP': 'LFP amplitude'},
            'outer_dims': ['chan_name'],
            'outer_coords': {
                'chan_name': 'Subject + session + channel',
            },
            'inner_dims': ['sample_num', 'trial_num'],
            'inner_coords': {
                'sample_num': 'Sample number',
                'time': 'Time, s',
                'trial_num': 'Trial number (sequential)',
                'trial_id': 'Trial number in the experiment'
            },
            'fpath_data_column': 'fpath_epoched',
        },
        'params': {
            'lock_event': {
                'desc': 'Type of the central epoch event',
                'value': None,
                'name_old': 'epoching_lock_event'},
            'time_win': {
                'desc': 'Epoch limits',
                'value': None,
                'name_old': 'epoching_time_win'},
        }
    }
}
            
fpath_dfg_in = (r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan'
                '\chan_all_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)')
fpath_dfg_out = (r'D:\WORK\Camilo\Processing_Pancake_2sess_allchan' 
                  '\dfg_LFP_(ev=stim1_t)_(t=-1.00-3.00)')

dfg = DataFileGroup()

# Load description of a data file group
with open(fpath_dfg_in, 'rb') as fid:
    dfg.outer_table = pickle.load(fid)
dfg._init_outer_indices()

# Get parameter values from the table attributes and store them into
# the processing steps list 'proc_steps'
for proc_step in proc_steps.values():
    for param in proc_step['params'].values():
        if 'name_old' in param:
            val = dfg.outer_table.attrs[param['name_old']]
            param['value'] = val
            param.pop('name_old')

# Create data processing tree form 'proc_steps'
for proc_step in proc_steps.values():
    dfg.data_proc_tree.add_process_step(
            proc_step['name'],
            proc_step['function'],
            proc_step['params'],
            proc_step['data_desc_out'])

# Change root folder
dfg.change_root('H:', 'D:')
    
data_desc = dfg.get_data_desc()

pbar = tqdm(total=len(dfg.outer_table))

for entry in dfg.get_table_entries():
    
    # Load dataset
    X = dfg.load_inner_data(entry, h5=False)
    
    # Give a name to the dataset variable
    assert(len(data_desc['variables']) == 1)
    var_name = list(data_desc['variables'].keys())[0]
    X = X.rename_vars({'__xarray_dataarray_variable__': var_name})
    
    # Set dataset attributes
    dfg.set_inner_data_attrs(entry, X)
        
    # Resave dataset
    fpath_data = dfg.get_inner_data_path(entry)
    dirpath_data = os.path.split(fpath_data)[0]
    fname_out = 'LFP_(ev=stim1_t)_(t=-1.00-3.00).nc'
    fpath_out = os.path.join(dirpath_data, fname_out)
    dfg.save_inner_data(entry, X, fpath_out)

    pbar.update()
   
pbar.close()

#fpath = fpath_out
#Q = xr.load_dataset(fpath, engine='h5netcdf')

# Fill the outer table attributes
dfg.outer_table.attrs = usf.flatten_dict(dfg.make_data_attrs())
dfg.save(fpath_dfg_out)

dfg = None
dfg = DataFileGroup()
dfg.load(fpath_dfg_out)

# TODO: does this work correctly, given that get_table_entries() returns
# copies, not references to the rows of pandas table?
