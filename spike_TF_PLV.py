# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 22:03:10 2021

@author: Nikita
"""

import os
import numpy as np
#import h5py
#import scipy as sc
#import scipy.signal as sig
#import matplotlib.pyplot as plt
#import sys
#import pandas as pd
import xarray as xr
import pickle as pk
#from timeit import default_timer as timer
#import pycorrelate as pycor
import time

import data_file_group as dfg
#import firing_rate as fr
import roi_utils as roi
#import spiketrain_manager as spk
import useful as usf


def _calc_dfg_spike_TF_PLV_inner(X_in, cell_epoched_info, tROI_descs,
                          tROIset_name, ncell_max=None):

    # Channel, subject, session
    # TODO: instead of parsing chan name to obtain subj + session,
    # make them additional outer coords
    attrs = usf.unflatten_dict(X_in.attrs)
    chan_name = attrs['outer_coord_vals']['chan_name']
    chan_name_parts = chan_name.split('_')
    subj_name = chan_name_parts[0]
    sess_id = '_'.join(chan_name_parts[1:3])
    
    # Cells with the same subject + session as the current channel
    cell_mask = (cell_epoched_info.subj_name == subj_name) & \
           (cell_epoched_info.sess_id == sess_id) 
    cells = cell_epoched_info[cell_mask]
    
    # Limit the number of cells (for testing purposes)
    if ncell_max is not None:
        cells = cells[:ncell_max]
    
    NROI = len(tROI_descs)
    Nfreq = len(X_in.freq)
    Ncell = len(cells)
    Ntrials = len(X_in.trial_num)
    
    # Generate ROI names    
    ROI_nums = np.arange(NROI)
    ROI_names, ROI_names2 = roi.generate_ROI_names(['time'], tROI_descs)

    # Initialize PLV DataArray
    coords_PLV = {
        'freq':         X_in.freq,
        'tROI_num':     ROI_nums,
        'tROI_name':    ('tROI_num', ROI_names),
        'tROI_name2':   ('tROI_num', ROI_names2),
        'trial_num':    X_in.trial_num,
        'trial_id':     ('trial_num', X_in.trial_id),
        'cell_id':      cells.cell_id,
        'cell_name':    ('cell_id', cells.cell_name)
        }
    Xnan = np.nan * np.ones((Nfreq, NROI, Ntrials, Ncell), dtype='complex128')    
    data_PLV = xr.DataArray(Xnan, coords=coords_PLV,
                            dims=['freq', 'tROI_num', 'trial_num', 'cell_id'])
    
    # Initialize Nspikes DataArray
    coords_Nspikes = coords_PLV.copy()
    coords_Nspikes.pop('freq')
    Xnan = np.nan * np.ones((NROI, Ntrials, Ncell))    
    data_Nspikes = xr.DataArray(Xnan, coords=coords_Nspikes,
                          dims=['tROI_num', 'trial_num', 'cell_id'])
    
    for cell_num in range(Ncell):

        print(f'Cell: {cell_num} / {Ncell}')
        t0 = time.time()
        
        # Load spiketrain
        cell = cells.iloc[cell_num]
        with open(cell.fpath_epoched, 'rb') as fid:
            spikes = pk.load(fid).values
        
        for trial_num, trial_spikes in enumerate(spikes):            
            for tROI_num, tROI in enumerate(tROI_descs):
                
                # Select spikes within the given time ROI
                t_range = tROI['limits']['time']
                spike_mask = ((trial_spikes >= t_range[0]) &
                              (trial_spikes < t_range[1]))
                ROI_spikes = trial_spikes[spike_mask]
                Nspikes = len(ROI_spikes)
                if Nspikes == 0:
                    continue
                
                # Spike-triggerred LFP phase
                PLV = X_in.TF.isel(trial_num=trial_num).interp(time=ROI_spikes)
                PLV /= np.abs(PLV)
                PLV = PLV.mean(dim='time').data
                
                # Store into output arrays
                data_PLV[:, tROI_num, trial_num, cell_num] = PLV
                data_Nspikes[tROI_num, trial_num, cell_num] = Nspikes
                
        dt = time.time() - t0
        print(f'dt = {dt}')
        
    # Collect the output dataset
    data_vars = {'PLV': data_PLV, 'Nspikes': data_Nspikes}
    X_out = xr.Dataset(data_vars)
    return X_out


def calc_dfg_spike_TF_PLV(dfg_in, cell_epoched_info, tROI_descs,
                          tROIset_name, ncell_max=None):
    """ Calculate spike-field coherence for each channel-cell pair.
    
    The result is calculated for each frequency, in the time intervals
    given by tROI_vec
    
    Output variables:  PLV, Nspikes
    Output dimensions: freq, tROI, trial, cell
    """
    
    print('calc_dfg_spike_TF_PLV')
    
    # Name of the processing step
    proc_step_name = 'Spike-LFP PLV calculation'
    
    # Dictionary of parameters
    param_names = ['tROI_descs', 'tROIset_name', 'cell_epoched_info',
                   'ncell_max']
    local_vars = locals()
    params = {par_name: local_vars[par_name] for par_name in param_names}
    
    # Name of the dfg's outer table column for the paths to Dataset files
    fpath_data_column = 'fpath_spPLV_tROI'
        
    # Function that converts the parameters dict to the form suitable
    # for storing into a processing step description
    def gen_proc_step_params(par):
        par_out = {
            'ROI_descs': {
                'desc': 'Names and coordinate ranges of the ROIs',
                'value': [str(d) for d in par['tROI_descs']]},
            'ROIset_name': {
                'desc': 'Name of the ROI set',
                'value': str(par['tROIset_name'])},
            'cells': {
                'desc': 'Cells that provide spike trains',
                'value': 'All from the same session'}
        }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath(fpath_in, params):
        fpath_data_postfix = 'spPLV_tROI'
        fpath_noext, ext  = os.path.splitext(fpath_in)
        return fpath_noext + '_' + fpath_data_postfix + ext
    
    # Description of the new variables
    vars_new_descs = {
            'PLV': 'Spike-LFP phase-locking value in a time ROI',
            'Nspikes': 'Number of spikes in a time ROI'
    }
    
    # Description of the new coordinates
    coords_new_descs = {
            'tROI_num': 'Time ROI (number)',
            'tROI_name': 'Time ROI (name)',
            'tROI_name2': 'Time ROI (limits)',
            'cell_id': 'Cell number',
            'cell_name': 'Cell name (subject + session + channel)',
    }
    
    # Call calc_dataset_ROIs() for each inner dataset of the DataFileGroup
    dfg_out = dfg.apply_dfg_inner_proc(
            dfg_in, _calc_dfg_spike_TF_PLV_inner, params, proc_step_name,
            gen_proc_step_params, fpath_data_column, gen_fpath,
            vars_new_descs, coords_new_descs)
    
    return dfg_out



