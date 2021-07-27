# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 02:56:47 2021

@author: Nikita
"""

import pandas as pd
import os
import re
import numpy as np
import h5py
import xarray as xr
import pickle as pk


def create_spiketrain_info(dirpath_root):
    
    col_names =['cell_id', 'cell_name', 'subj_name', 'sess_id', 'chan_id', 'cell_id_local', 'fpath_spikes']
    cell_tbl = pd.DataFrame(columns=col_names)
    
    cell_id = 0
    
    for dirpath, dirnames, filenames in os.walk(dirpath_root):
        for filename in filenames:
            if filename == 'unit.mat':
                
                fpath_spikes = os.path.join(dirpath, 'unit.mat')
    
                dirpath_base, cell_dirname = os.path.split(dirpath)
                dirpath_base, chan_dirname = os.path.split(dirpath_base)
                dirpath_base, array_dirname = os.path.split(dirpath_base)
                dirpath_base, sess_dirname = os.path.split(dirpath_base)
                dirpath_base, date_dirname = os.path.split(dirpath_base)
                dirpath_base, subj_dirname = os.path.split(dirpath_base)
                
                cell_id_local = int(re.match('cell([0-9]+)', cell_dirname).groups()[0])
                chan_id = int(re.match('channel([0-9]+)', chan_dirname).groups()[0])
                sess_id_local = int(re.match('session([0-9]+)', sess_dirname).groups()[0])
                date_str = date_dirname
                subj_name = subj_dirname
                
                sess_id = date_str + '_' + str(sess_id_local)
                cell_name = '%s_%s_ch%i_c%i' % (subj_name, sess_id, chan_id, cell_id_local)
                
                entry = pd.DataFrame([[cell_id, cell_name, subj_name, sess_id, chan_id, cell_id_local, fpath_spikes]], columns=col_names)
                cell_tbl = cell_tbl.append(entry)
                
                cell_id = cell_id + 1
                
    return cell_tbl

# Epoch spike data
def epoch_spike_data_batch(cell_info, trial_info, lock_event, time_win):
    
    print('epoch_spike_data_batch')
    
    # Create output table
    cell_info_out = cell_info.copy()
    cell_info_out.insert(len(cell_info_out.columns), 'fpath_epoched', '')
    
    Ncell = len(cell_info)
    
    for n in range(Ncell):
        
        print('%i / %i' % (n, Ncell))
        
        cell = cell_info.iloc[n]

        # Trial info entry for the same subject + session as the current cell
        id_ = np.where([x['subj_name']==cell.subj_name and x['sess_id']==cell.sess_id for x in trial_info])[0][0]
        ti = trial_info[id_]
        
        # Load spiketrain
        with h5py.File(cell.fpath_spikes, 'r') as f:
            spike_times = np.array(f['timestamps']) / 1000            

        # Times of the events to lock trials on
        lock_ev_times = np.array(ti['trial_tbl'][lock_event])
        Ntrials = len(lock_ev_times)
        
        # Allocate matrix for the epoched spike data
        X = np.empty(Ntrials, dtype=np.object)
        
        for m in range(Ntrials):
            
            t0 = lock_ev_times[m]
            
            if np.isnan(t0):
                X[m] = []
                continue
            
            # Select spikes that belong to the current trial
            mask = (spike_times >= (t0 + time_win[0])) & (spike_times < (t0 + time_win[1]))
            X[m] = np.array(spike_times[mask] - t0)
        
        # Create output xarray
        coords = {
            'trial_num':    range(Ntrials),
            'trial_id':     ('trial_num', ti['trial_tbl'].trial_id),
            }
        X_ep = xr.DataArray(X, coords=coords, dims=['trial_num'])
            
        # Add info about the epoching operation to the output file
        X_ep.attrs['epoching_lock_event'] = lock_event
        X_ep.attrs['epoching_time_win'] = '%.02f - %.02f' % (time_win[0], time_win[1])
        X_ep.attrs['epoching_fpath_source'] = cell.fpath_spikes
            
        # Save the epoched data
        dirpath_out = os.path.split(cell.fpath_spikes)[0]
        fname_out = 'epochs_(ev=%s)_(t=%.02f-%.02f).pickle' % (lock_event, time_win[0], time_win[1])
        fpath_out = os.path.join(dirpath_out, fname_out)
        with open(fpath_out, 'wb') as fid:
            pk.dump(X_ep, fid)

        
        # Update output table
        cell_info_out.fpath_epoched.iloc[n] = fpath_out
    
    # Add info about the epoching operation to the output table
    cell_info_out.attrs['epoching_lock_event'] = lock_event
    cell_info_out.attrs['epoching_time_win'] = '%.02f - %.02f' % (time_win[0], time_win[1])
    
    return cell_info_out
        
        
        