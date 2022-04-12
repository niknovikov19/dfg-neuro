# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 22:03:10 2021

@author: Nikita
"""

import copy
import os
#import sys

#import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import pickle as pk
#import scipy as sc
#import scipy.signal as sig
from tqdm import tqdm
import xarray as xr

#import pycorrelate as pycor

import data_file_group_2 as dfg
#import spiketrain_manager as spk
import firing_rate as fr
import useful as usf

    
def _calc_rvec_cov_nojit(rvec1, rvec2, nbins_jit, niter_jit, lag_range=None):
    """Cross-correlogram of two firing rate signals, excluding jittered cov.
    
    """

    # Non-jittered covariance
    c = np.correlate(rvec1, rvec2, mode='same')

    rvec1_jit = rvec1.copy()
    rvec2_jit = rvec2.copy()

    C_jit = np.nan * np.ones((niter_jit, len(c)))

    # Jittered covariances
    for n in range(niter_jit):

        fr.jitter_frate_vec(rvec1_jit, nbins_jit)
        fr.jitter_frate_vec(rvec2_jit, nbins_jit)

        C_jit[n,:] = np.correlate(rvec1_jit, rvec2_jit, mode='same')

    # Mean jittered cov, cov without jittered part
    c_jit = C_jit.mean(axis=0)
    c_nojit = c - c_jit

    N = len(c)
    lag0 = int(N/2)

    if lag_range is not None:        
        lag1 = max(0, lag0+lag_range[0])
        lag2 = min(N, lag0+lag_range[1]+1)
        c = c[lag1:lag2]
        c_jit = c_jit[lag1:lag2]
        c_nojit = c_nojit[lag1:lag2]    

    if lag_range is None:
        lag1 = -lag0
        lag2 = lag0
    else:
        lag1 = lag_range[0]
        lag2 = lag_range[1]+1        
    lags = np.arange(lag1, lag2)
    
    return c, c_jit, c_nojit, lags


def calc_dfg_rvec_cov_nojit(dfg_in, nbins_jit, niter_jit, time_range, 
                            lag_range=None, test_mode=False):
    """Calculate cross-correlograms for all cell pairs within a session. 
    
    """
 
    # Initialize output DataFileGroup
    proc_step_new =  {
        'name': 'Rate covariance',
        'function': 'calc_dfg_rvec_cov_nojit()',
        'params': {
            'nbins_jit': nbins_jit,
            'niter_jit': niter_jit,
            'lag_range': lag_range,
            'time_range': time_range
            },
        'data_desc_out': {
            'variables': 
                {'rcov': 'Rate covariance'},
            'outer_dims':
                ['subj_name', 'sess_id'],
            'outer_coords': {
                'subj_name': 'Subject name',
                'sess_id': 'Session id'
                },
            'fpath_data_column':
                'fpath_rvec_cov',
            'inner_dims':
                ['cell1_num', 'cell2_num', 'lag_num', 'trial_num'],
            'inner_coords': {
                'cell1_name': 'Cell 1 name: subject + session + channel + cell',
                'cell1_num': 'Cell 1 number',
                'cell2_name': 'Cell 2 name: subject + session + channel + cell',
                'cell2_num': 'Cell 2 number',
                'lag_num': 'Covariance lag number, sequential',
                'lag': 'Covariance lag, s',
                'trial_num': 'Trial number (sequential)',
                'trial_id': 'Trial number in the experiment'
                }
            }
        }
    dpc_out = copy.deepcopy(dfg_in.data_proc_tree)
    dpc_out.add_process_step(proc_step_new['name'],
                             proc_step_new['function'],
                             proc_step_new['params'],
                             proc_step_new['data_desc_out'])
    dfg_out = dfg.DataFileGroup()
    dfg_out.create2(dpc_out.proc_steps, used_proc_steps='last')
    
    
    # Unique subject names and session identifiers
    tbl_in = dfg_in.outer_table
    subj_names_uni = tbl_in.subj_name.unique()
    sess_idx_uni = tbl_in.sess_id.unique()
    
    for subj_name in subj_names_uni:
        for sess_id in sess_idx_uni:
        
            # Get cell info for the current session
            mask = (tbl_in.subj_name == subj_name) & (tbl_in.sess_id == sess_id)
            dfg_sess = dfg_in.subset(mask)
    
            print('Session: {subj_name}_{sess_id}')
            Ncells = dfg_sess.get_num_table_entries()
            sz = Ncells * Ncells
            pbar = tqdm(total=sz)
            C0 = None
            
            for entry1 in dfg_sess.get_table_entries():
                
                # Load rate signal Select time interval (cell 1)
                R1 = dfg_sess.load_inner_data(entry1).r 
                tvec = R1.time.values
                mask = (tvec >= time_range[0]) & (tvec < time_range[1])
                tvec_ROI = tvec[mask]
                R1 = usf.xarray_select_xr(R1, {'time': tvec_ROI})
                
                for entry2 in dfg_sess.get_table_entries():
                    
                    # Test
                    if test_mode:
                        if ((sess_id != sess_idx_uni[0])
                                or (entry1 != 49) or (entry2 != 6)):
                            pbar.update() 
                            continue
                
                    # Load rate signal Select time interval (cell 2)
                    R2 = dfg_sess.load_inner_data(entry2).r
                    R2 = usf.xarray_select_xr(R2, {'time': tvec_ROI})
                    
                    Ntrials = len(R1.trial_num)
                    for trial_num in range(Ntrials):
                        
                        # Firing rate vectors of two cells in the current trial
                        rvec1 = R1.data[trial_num,:]
                        rvec2 = R2.data[trial_num,:]
                        
                        # Skip empty spiketrains
                        #if ~np.any(rvec1) and ~np.any(rvec2):
                        #    continue
                        
                        # Cross-correlation
                        c, c_jit, c_nojit, lags = _calc_rvec_cov_nojit(
                            rvec1, rvec2, nbins_jit, niter_jit, lag_range)
                        
                        '''
                        plt.clf()
                        plt.plot(c)
                        plt.plot(c_nojit)
                        plt.plot(c_jit)
                        plt.draw()
                        b = plt.waitforbuttonpress()
                        if b==False:
                            sys.exit()
                        '''
                        
                        # Allocate the output ndarray for a session
                        if C0 is None:
                            Nlags = len(lags)
                            sz = (Ncells, Ncells, Nlags, Ntrials)
                            C0 = np.nan * np.ones(sz)
                        
                        # Update the output
                        C0[entry1, entry2, :, trial_num] = c_nojit
                        
                    pbar.update()                        
            
            # Build xarray DataSet from ndarray for the session
            dt = tvec[1] - tvec[0]
            dims = ['cell1_num', 'cell2_num', 'lag_num', 'trial_num']
            coords = {
                'cell1_num': dfg_sess.get_table_entries(),
                'cell1_name': ('cell1_num',
                               dfg_sess.outer_table.cell_name.values),
                'cell2_num': dfg_sess.get_table_entries(),
                'cell2_name': ('cell2_num',
                               dfg_sess.outer_table.cell_name.values),
                'lag_num': np.arange(len(lags)),
                'lag': ('lag_num', lags * dt),
                'trial_num': R1.trial_num.data,
                'trial_id': ('trial_num', R1.trial_id.data)
                }
            C = xr.DataArray(C0, coords=coords, dims=dims)
            X = xr.Dataset({'rcov': C})
            
            # Output path for the DataSet
            fpath_rvec = dfg_sess.outer_table.fpath_rvec[0]
            dirpath_sess = fpath_rvec
            for n in range(4):
                dirpath_sess = os.path.split(dirpath_sess)[0]
            fname_out = os.path.split(fpath_rvec)[1]
            fname_out_noext = os.path.splitext(fname_out)[0]
            fname_out_noext = fname_out_noext.replace('rvec', 'rcov')
            fname_out = f'{fname_out_noext}_(bins={nbins_jit}_iter={niter_jit}'
            fname_out += f'_nlags={Nlags}_t={time_range[0]}-{time_range[1]}).nc'
            fpath_out = os.path.join(dirpath_sess, fname_out)
            
            # Add DataSet to DataFileGroup
            outer_coords = {'subj_name': subj_name, 'sess_id': sess_id}
            dfg_out.add_entry(outer_coords, X, fpath_out)
            
            pbar.close()
    
    return dfg_out

    


