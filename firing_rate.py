# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 05:45:01 2021

@author: Nikita
"""

import pandas as pd
import numpy as np
import trial_manager as trl
import h5py
import xarray as xr
import matplotlib.pyplot as plt
import os
import pickle as pk
import scipy.stats as stat

import useful as usf


def calc_firing_rates_by_stim_types(cell_info, trial_info, subj_used=None, sess_used=None):
    
    subj_used = 'Pancake'
    sess_used = ['20130923_1']
    #sess_used = ['20130923_1', '20130927_1']
    
    lock_event = 'stim1_t'
    time_win = np.array([0.3, 1])
    
    # Indices in trial_info with appropriate subject + session    
    subj_sess_mask = [
        (x['subj_name'] in subj_used) and (x['sess_id'] in sess_used)
        for x in trial_info]
    subj_sess_idx = np.where(subj_sess_mask)[0]
    
    # Cycle over subject + sesssion combinations 
    for subj_sess_id in subj_sess_idx:
        
        ti = trial_info[subj_sess_id]
        
        # Table of correct trials
        trial_tbl = trl.select_correct_trials(ti['trial_tbl'])
        Ntrial0 = len(trial_tbl)
        
        # Stimulus codes
        stim_codes_uni = np.unique(trial_tbl.stim1_code)
        Nstim = len(stim_codes_uni)
        
        # Cells recorded in this session
        cell_mask = ((cell_info.subj_name == ti['subj_name']) &
                     (cell_info.sess_id == ti['sess_id']))
        cell_tbl = cell_info[cell_mask]
        Ncell = len(cell_tbl)
        
        # Matrix of firing rates (cells, stim_codes), averaged over trials and samples
        coords = {'cell_id': cell_tbl.cell_id,
                  'cell_name': ('cell_id', cell_tbl.cell_name),
                  'stim_code': stim_codes_uni}
        R = xr.DataArray(np.zeros((Ncell,Nstim)), coords=coords,
                         dims=['cell_id', 'stim_code'])
        
        coords = {'cell_id': cell_tbl.cell_id,
                  'cell_name': ('cell_id', cell_tbl.cell_name),
                  'trial_id': range(Ntrial0)}
        Nspikes_mat = xr.DataArray(np.zeros((Ncell,Ntrial0)), coords=coords,
                                   dims=['cell_id', 'trial_id'])
        
        # Cycle over cells
        for n in range(Ncell):
            
            print('cell: %i / %i' % (n, Ncell))
            
            cell = cell_tbl.iloc[n]
            
            # Load spiketrain
            with h5py.File(cell.fpath_spikes, 'r') as f:
                spike_times = np.array(f['timestamps']) / 1000
                
            # Cycle over trials
            for m in range(Ntrial0):
                epoch = trial_tbl.iloc[m][lock_event] + time_win
                Nsp = np.sum((spike_times >= epoch[0]) & (spike_times <= epoch[1]))
                Nspikes_mat.loc[dict(cell_id=cell.cell_id, trial_id=m)] = Nsp
    
            '''            
            # Cycle over stimulus types
            for stim_code in stim_codes_uni:
                
                # Trials with the given stimulus presented
                trial_stim_tbl = trial_tbl[trial_tbl.stim1_code == stim_code]
                Ntrial = len(trial_stim_tbl)
                
                rvec = np.zeros(Ntrial)
                
                # Cycle over trials
                for m in range(Ntrial):
                    #print('%i / %i\n' % (m,Ntrial))
                    epoch = trial_stim_tbl.iloc[m][lock_event] + time_win
                    rvec[m] = np.sum((spike_times >= epoch[0]) & (spike_times <= epoch[1]))
                    
                # Firing rate
                r = rvec.mean() / (epoch[1] - epoch[0])
                R.loc[dict(cell_id=cell.cell_id, stim_code=stim_code)] = r
            '''
        
def calc_frate_vec(spikes, time_range, dt):
    
    t1 = time_range[0]
    t2 = time_range[1]
    
    Nbins = int((t2 - t1) / dt)
    rvec = np.zeros(Nbins)
    
    bin_idx = np.round((spikes - t1) / dt)
    bin_idx = bin_idx[(bin_idx >=0) & (bin_idx < Nbins)]
    bin_idx = bin_idx.astype(np.int64)
    
    rvec = np.bincount(bin_idx, minlength=Nbins)
    
    return rvec


#@profile    
def jitter_frate_vec(rvec, nbins_jit):
    rmat = rvec.reshape((nbins_jit,-1), order='F')
    np.random.shuffle(rmat)
    
    
def calc_frate_vec_batch(cell_epoched_info, time_range, dt, need_recalc=False):
        
    print('calc_frate_vec_batch')

    # Create output table
    cell_info_out = cell_epoched_info.copy()
    cell_info_out.insert(len(cell_info_out.columns), 'fpath_rvec', '')
    
    Ncell = len(cell_epoched_info)
    
    for n in range(Ncell):
        
        print(f'{n} / {Ncell}')
        
        cell = cell_epoched_info.iloc[n]
        
        # Input and output paths
        fpath_in = cell.fpath_epoched
        postfix = f'rvec_(t={time_range[0]*1000:.0f}-{time_range[1]*1000:.0f}_dt={dt*1000:.0f})'
        fpath_out = usf.generate_fpath_out(fpath_in, postfix)
        
        # Check whether it is already calculated
        if os.path.exists(fpath_out) and (need_recalc==False):
            print('Already calculated')
            cell_info_out.fpath_rvec.iloc[n] = fpath_out
            continue
        
        # Load epoched spikes
        with open(fpath_in, 'rb') as fid:
            S = pk.load(fid)
        
        Ntrials = S.trial_num.size
        
        R = None
        
        for m in range(Ntrials):
            
            # Spikes for the current trial
            spikes = S[m].item()
            
            # Calculate firing rate vector
            rvec = calc_frate_vec(spikes, time_range, dt)
            
            # Allocate output
            if R is None:
                
                Nsamples = len(rvec)
                coords = {
                    'trial_num':    range(Ntrials),
                    'trial_id':     ('trial_num', S.trial_id),
                    'sample_num':   range(Nsamples),
                    'time':         ('sample_num', time_range[0] + dt * np.arange(Nsamples))
                    }
                Rnan = np.nan * np.ones((Ntrials, Nsamples), dtype='float64')
                R = xr.DataArray(Rnan, coords=coords, dims=['trial_num', 'sample_num'])
                
            # Store TF of the currenttrial
            R[m,:] = rvec
        
        # Add info about the operation
        R.attrs = S.attrs.copy()
        R.attrs['rvec_time_range'] = time_range
        R.attrs['rvec_dt'] = dt
        R.attrs['rvec_fpath_source'] = fpath_in
            
        # Save the transformed data
        R.to_netcdf(fpath_out, engine='h5netcdf')
        
        # Update output table
        cell_info_out.fpath_rvec.iloc[n] = fpath_out
        
    # Add info about the epoching operation to the output table
    cell_info_out.attrs['rvec_time_range'] =  time_range
    cell_info_out.attrs['rvec_dt'] = dt
    
    return cell_info_out
    

def calc_rvec_trial_avg(rvec_info):

    RR = None
    
    Ncell = len(rvec_info)
    
    for n in range(Ncell):
        
        print(f'{n} / {Ncell}')
        
        cell = rvec_info.iloc[n]
        
        # Load epoched TF
        fpath_in = cell.fpath_rvec
        R = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']
        
        # Allocate output
        if n==0:
            coords = {
                'sample_num':   R.sample_num,
                'time':         ('sample_num', R.time),
                'cell_num':     range(Ncell),
                'cell_name':    ('cell_num', rvec_info.cell_name)
                }
            Nsamples = len(R.sample_num)
            RR = xr.DataArray(np.zeros((Nsamples, Ncell)), coords=coords, dims=['sample_num', 'cell_num'])
            
        # Update output
        RR[:,n] = R.mean(dim='trial_num')
        
    return RR  

'''
qvec = np.zeros(Ncell)
            
for n in range(Ncell):
    rvec = np.array(R[n,:])
    id_max = np.argmax(rvec)
    rmax = rvec[id_max]
    rvec[id_max] = np.nan
    ravg = np.nanmean(rvec)
    qvec[n] = rmax / ravg
'''

'''    
epoch = trial_tbl.iloc[0][lock_event] + time_win
mask = (spike_times >= epoch[0]) & (spike_times <= epoch[1])
st = spike_times[mask]
plt.figure()
plt.plot(st, np.zeros(len(st)), '.')
'''

def calc_firing_rates_by_stim_types_2(rvec_info, trial_info, time_win,
                                      use_log=False, outlier_thresh=None,
                                      dirpath_out_img=None):
    
    if dirpath_out_img is not None:
        plt.figure()
    
    Nsess = len(trial_info)
    
    # Get all stimulus codes
    stim_codes_all = []
    for sess_num in range(Nsess):
        stim_codes_all += trial_info[sess_num]['trial_tbl'].stim1_code.tolist()
    stim_codes_all_uni = np.unique(stim_codes_all)
    
    # Initialize output table
    tbl_out = rvec_info.copy()
    col_names_new = [f'r_stim{x}' for x in stim_codes_all_uni]
    col_names_new += ['stim_code_rmax', 'stim_code_rmin',
                      'pF_anova', 'pT_minmax']
    tbl_out[col_names_new] = np.nan    
    
    for sess_num in range(Nsess):
        
        sess = trial_info[sess_num]
        
        trial_tbl = sess['trial_tbl']
        stim_codes_uni = np.unique(trial_tbl.stim1_code)
        Ncodes = len(stim_codes_uni)
        
        cell_mask = (rvec_info.sess_id == sess['sess_id'])
        rvec_info_sess = rvec_info[cell_mask]
    
        Ncells = len(rvec_info_sess)
        
        for cell_num in range(Ncells):
            
            print(f'sess: {sess_num} / {Nsess}  cell: {cell_num} / {Ncells}')
            
            cell = rvec_info_sess.iloc[cell_num]
            
            # Load firing rate data
            fpath_rvec = cell.fpath_rvec
            R  = usf.load_xarray(fpath_rvec)
            
            # Here we will store firing rates for each stimulus code
            RR = []
            RR_avg = np.zeros(Ncodes)
            #RR_isnorm = np.zeros(Ncodes)
            
            for code_num in range(Ncodes):
                
                # Trials with the current stimulus code
                trial_mask_tbl = (trial_tbl.stim1_code == stim_codes_uni[code_num])
                trial_idx = trial_tbl[trial_mask_tbl].trial_id
                
                # Get firing rates: average over the window and select trials 
                trial_mask_R = R.trial_id.isin(trial_idx)
                sample_mask_R = (R.time >= time_win[0]) & (R.time <= time_win[1])
                rvec = R[trial_mask_R, sample_mask_R].mean(dim='sample_num').data
                
                # Log
                if use_log:
                    rvec = np.log(rvec)
                
                # Remove outliers
                if outlier_thresh is not None:
                    out_mask = (np.abs(stat.zscore(rvec)) < 2.5)
                    rvec = rvec[out_mask]
                
                RR.append(rvec)
                RR_avg[code_num] = np.mean(rvec)
                #st, RR_isnorm[code_num] = stat.normaltest(rvec)
                
            # ANOVA on firing rates over the stimulus codes
            F,pF = stat.f_oneway(*RR)
            
            # T-test between the stimuli with the min and max firing rate
            id_min = np.argmin(RR_avg)
            id_max = np.argmax(RR_avg)
            T, pT = stat.ttest_ind(RR[id_min], RR[id_max])
            
            # Visualize the result
            if dirpath_out_img is not None:                
                plt.clf()                
                for code_num in range(Ncodes):
                    plt.plot([stim_codes_uni[code_num]] * len(RR[code_num]), RR[code_num], 'k.')
                plt.xlabel('Stimulus code')
                plt.ylabel('Firing rate')
                plt.title(f'{cell.cell_name}  t = ({time_win[0]} - {time_win[1]})  F = {F:.03f}, p = {pF:.08f}')
                
            # Add the results to the output table
            row_id = np.where(tbl_out.cell_name==cell.cell_name)[0][0]
            for code_num in range(Ncodes):
                col_name = f'r_stim{stim_codes_uni[code_num]}'
                tbl_out.iloc[row_id, tbl_out.columns.get_loc(col_name)] = RR_avg[code_num]
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('stim_code_rmax')] = stim_codes_all_uni[np.nanargmax(RR_avg)]
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('stim_code_rmin')] = stim_codes_all_uni[np.nanargmin(RR_avg)]
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('pF_anova')] = pF
            tbl_out.iloc[row_id, tbl_out.columns.get_loc('pT_minmax')] = pT
            
    return tbl_out
            
                
#tbl_out = calc_firing_rates_by_stim_types_2(rvec_info, trial_info, time_win=(0.5, 1.2), use_log=True, outlier_thresh=2.5)
#tbl_out = calc_firing_rates_by_stim_types_2(rvec_info, trial_info, time_win=(0.5, 1.2), use_log=False, outlier_thresh=None)
            
                
        
        
    
    
    
    
    

