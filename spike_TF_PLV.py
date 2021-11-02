# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 22:03:10 2021

@author: Nikita
"""

import os
import numpy as np
import h5py
import scipy as sc
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys
import pandas as pd
import xarray as xr
import pickle as pk
from timeit import default_timer as timer
import pycorrelate as pycor


import spiketrain_manager as spk
import firing_rate as fr
import useful as usf



def calc_spike_TF_PLV_by_trial_batch(cell_epoched_info, chan_tf_info,
                                     tROI_vec, tROIset_name):
# Calculate spike-field coherence for each channel-cell pair in each trial
# Result is calculated for each frequency, in the time intervals given by tROI_vec
# ROI = [{'name': ..., 'tlim': ...}, ...]  
# Output variables:  PLV, Nspikes
# Output dimensions: freq, tROI, trial, cell
    
    print('calc_spike_TF_PLV_by_trial_batch')
    
    # Create output table
    spPLVf_info_out = chan_tf_info.copy()
    spPLVf_info_out.insert(len(spPLVf_info_out.columns), 'fpath_spPLVf', '')
    
    NROI = len(tROI_vec)
    Nchan = len(chan_tf_info)
    
    # Generate ROI names
    ROI_nums, ROI_names, ROI_names2 = [], [], []
    for ROI_num, ROI in enumerate(tROI_vec):
        ROI_nums.append(ROI_num)
        ROI_names.append(ROI['name'])
        ROI_name2 = f'ROI_(t={ROI["tlim"][0]}-{ROI["tlim"][1]})'
        ROI_names2.append(ROI_name2)
    
    for chan_num in range(Nchan):
        
        print('Channel: %i / %i' % (chan_num, Nchan))
        
        chan = chan_tf_info.iloc[chan_num]
        
        # Load epoched TF
        fpath_tf = chan.fpath_tf
        W = xr.load_dataset(fpath_tf, engine='h5netcdf')
        W = W['__xarray_dataarray_variable__']
        
        # Cells with the same subject + session as the current channel
        mask = (cell_epoched_info.subj_name == chan.subj_name) & \
               (cell_epoched_info.sess_id == chan.sess_id) 
        cell_vec_cur = cell_epoched_info[mask]
        
        Nfreq = len(W.freq)
        Ncell = len(cell_vec_cur)
        Ntrials = len(W.trial_num)
    
        coords_w = {
            'freq':         W.freq,
            'tROI_num':     ROI_nums,
            'tROI_name':    ('tROI_num', ROI_names),
            'tROI_name2':   ('tROI_num', ROI_names2),
            'trial_num':    W.trial_num,
            'trial_id':     ('trial_num', W.trial_id),
            'cell_id':      cell_vec_cur.cell_id,
            'cell_name':    ('cell_id', cell_vec_cur.cell_name)
            }
        Xnan = np.nan * np.ones((Nfreq, NROI, Ntrials, Ncell),
                                dtype='complex128')    
        data_out_w = xr.DataArray(
                Xnan, coords=coords_w,
                dims=['freq', 'ROI_num', 'trial_num', 'cell_id'])
        
        coords_N = {
            'tROI_num':     ROI_nums,
            'tROI_name':    ('tROI_num', ROI_names),
            'tROI_name2':   ('tROI_num', ROI_names2),
            'trial_num':    W.trial_num,
            'trial_id':     ('trial_num', W.trial_id),
            'cell_id':      cell_vec_cur.cell_id,
            'cell_name':    ('cell_id', cell_vec_cur.cell_name)
            }
        Xnan = np.nan * np.ones((NROI, Ntrials, Ncell))    
        data_out_N = xr.DataArray(Xnan, coords=coords_N,
                                  dims=['tROI_num', 'trial_num', 'cell_id'])
        
        for cell_num in range(Ncell):

            print('Cell: %i / %i' % (cell_num, Ncell))
        
            cell = cell_vec_cur.iloc[cell_num]
            
            # Load spiketrain
            with open(cell.fpath_epoched, 'rb') as fid:
                st_epoched = pk.load(fid).values
            
            for trial_num in range(Ntrials):
                
                # Spike times for the current trial
                st_trial = st_epoched[trial_num]
                
                for ROI_num in range(NROI):
                    
                    t_range = tROI_vec[ROI_num]['tlim']
                
                    # Select spikes from the given time range
                    mask = (st_trial >= t_range[0]) & (st_trial < t_range[1])
                    st_ROI = st_trial[mask]
                    Nspikes = len(st_ROI)
                    if Nspikes==0:
                        continue
                    
                    # Spike-triggerred LFP phase
                    w = W.isel(trial_num=trial_num).interp(time=st_ROI)
                    w /= np.abs(w)
                    w = w.mean(dim='time').data
                    
                    # Store into output arrays
                    data_out_w[:, ROI_num, trial_num, cell_num] = w
                    data_out_N[ROI_num, trial_num, cell_num] = Nspikes
            
        # Collect the output dataset
        data_vars = {
            'PLV':      data_out_w,
            'Nspikes':  data_out_N
            }
        data_out = xr.Dataset(data_vars=data_vars, coords=coords_w)
        
        # Add info about the operation
        data_out.attrs = W.attrs.copy()
        W.attrs['tROIset_name'] = tROIset_name
        W.attrs['tROI_vec'] = tROI_vec
        data_out.attrs['spPLV_fpath_source'] = fpath_tf
            
        # Save the transformed data
        fpath_in = chan.fpath_tf
        postfix = 'spPLV_trials_(%s)' % tROIset_name
        fpath_out = usf.generate_fpath_out(fpath_in, postfix)
        data_out.to_netcdf(fpath_out, engine='h5netcdf')
        
        # Update output table
        spPLVf_info_out.fpath_spPLVf.iloc[chan_num] = fpath_out
    
    # Add info about the epoching operation to the output table
    spPLVf_info_out.attrs['tROIset_name'] = tROIset_name
    spPLVf_info_out.attrs['tROI_vec'] = tROI_vec
    
    return spPLVf_info_out# -*- coding: utf-8 -*-

