# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 23:32:07 2021

@author: Nikita
"""

import pandas as pd
import os
import numpy as np
import xarray as xr
import h5py
import scipy as sc
import scipy.signal as sig
import matplotlib.pyplot as plt
import sys
import re
import pickle as pk

import data_file_group as dfg
import useful as usf

        
# Epoch data
def epoch_lfp_data_batch(chan_info, trial_info, lock_event, time_win, need_recalc=False):
    
    print('epoch_lfp_data_batch')
    
    # Create output table
    chan_info_out = chan_info.copy()
    chan_info_out.insert(len(chan_info_out.columns), 'fpath_epoched', '')
    
    Nchan = len(chan_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_info.iloc[n]

        # Trial info entry for the same subject + session as the current channel
        id_ = np.where([x['subj_name']==chan.subj_name and x['sess_id']==chan.sess_id for x in trial_info])[0][0]
        ti = trial_info[id_]
        
        # Input and output paths
        fpath_in = chan.fpath_lfp
        postfix = 'epochs_(ev=%s)_(t=%.02f-%.02f)' % (lock_event, time_win[0], time_win[1])
        fpath_out = usf.generate_fpath_out(fpath_in, postfix, add_fname_in=False)
        
        # Check whether it is already calculated
        if os.path.exists(fpath_out) and (need_recalc==False):
            print('Already calculated')
            chan_info_out.fpath_epoched.iloc[n] = fpath_out
            continue
       
        # Load LFP
        with h5py.File(fpath_in, 'r') as f:
        
            # LFP signal
            X = np.array(f['lowpassdata']['data']['data']).flatten()
            
            # Sampling rate
            fs = f['lowpassdata']['data']['sampling_rate'][0,0]
        
        # Epoch window
        epoch_sample_idx = np.arange(time_win[0]*fs, time_win[1]*fs).astype(int)
        Nsamples = len(epoch_sample_idx)
        
        # Times of the events to lock trials on
        lock_ev_times = np.array(ti['trial_tbl'][lock_event])
        Ntrials = len(lock_ev_times)
        
        # Allocate matrix for the epoched LFP data
        coords = {
            'trial_num':    range(Ntrials),
            'trial_id':     ('trial_num', ti['trial_tbl'].trial_id),
            'sample_num':   range(Nsamples),
            'time':         ('sample_num', epoch_sample_idx / fs)
            }
        Xnan = np.nan * np.ones((Ntrials, Nsamples))
        X_ep = xr.DataArray(Xnan, coords=coords, dims=['trial_num', 'sample_num'])
        
        for m in range(Ntrials):
            
            if np.isnan(lock_ev_times[m]):
                continue
        
            # Indices of the epoch samples in the unrolled data    
            id0 = int(np.round(lock_ev_times[m] * fs))
            idx = epoch_sample_idx + id0
            
            # Epoch data
            x = X[idx]
            
            # Store
            X_ep.loc[m,:] = x
            
        # Add info about the epoching operation to the output LFP file
        X_ep.attrs['fs'] = fs
        X_ep.attrs['epoching_lock_event'] = lock_event
        X_ep.attrs['epoching_time_win'] = '%.02f - %.02f' % (time_win[0], time_win[1])
        X_ep.attrs['epoching_fpath_source'] = chan.fpath_lfp
            
        # Save the epoched data
        X_ep.to_netcdf(fpath_out)
        
        # Update output table
        chan_info_out.fpath_epoched.iloc[n] = fpath_out
    
    # Add info about the epoching operation to the output table
    chan_info_out.attrs['epoching_lock_event'] = lock_event
    chan_info_out.attrs['epoching_time_win'] = '%.02f - %.02f' % (time_win[0], time_win[1])
    
    return chan_info_out
  
def calc_ERP_batch(chan_epoched_info):
    
    Nchan = len(chan_epoched_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_epoched_info.iloc[n]
        
        # Load epoched LFP
        fpath_lfp = chan.fpath_epoched
        X = xr.load_dataset(fpath_lfp)['__xarray_dataarray_variable__']
        
        # Average over trials
        x = X.mean(dim='trial_num')
        Nsamples = len(x)
        
        # Allocate output
        if n==0:
            coords = {
                'chan_num':     range(Nchan),
                'chan_name':    ('chan_num', chan_epoched_info.chan_name),
                'sample_num':   range(Nsamples),
                'time':         ('sample_num', X.time)
                }
            ERP = xr.DataArray(np.zeros((Nchan,Nsamples)), coords=coords, dims=['chan_num', 'sample_num'])
            
        # Update output
        ERP[n,:] = x
        
    return ERP

def filt_epoched_data_batch(chan_epoched_info, freq_band, filt_order=5):
    
    Nchan = len(chan_epoched_info)
    
    # Create output table
    chan_info_out = chan_epoched_info.copy()
    chan_info_out.insert(len(chan_info_out.columns), 'fpath_filtered', '')
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_epoched_info.iloc[n]
        
        # Load epoched LFP
        fpath_lfp = chan.fpath_epoched
        X = xr.load_dataset(fpath_lfp)['__xarray_dataarray_variable__']
        
        # Create filter
        fs = X.attrs['fs']
        sos = sig.butter(filt_order, freq_band, 'bandpass', output='sos', fs=fs)
        
        # Filter the trial data
        Xfilt_np = sig.sosfiltfilt(sos, X, axis=1)
        Xfilt = xr.zeros_like(X)
        Xfilt.data = Xfilt_np
        
        # Add info about the epoching operation to the output LFP file
        Xfilt.attrs['filt_band'] = '%.02f - %.02f Hz' % (freq_band[0], freq_band[1])
        Xfilt.attrs['filt_order'] = filt_order
        Xfilt.attrs['filt_fpath_source'] = fpath_lfp
            
        # Save the filtered data
        dirpath_lfp = os.path.split(fpath_lfp)[0]
        fname_lfp = os.path.split(fpath_lfp)[1]
        fname_lfp_noext = os.path.splitext(fname_lfp)[0]
        fname_out = '%s_(f=%.01f-%.01f).nc' % (fname_lfp_noext, freq_band[0], freq_band[1])
        fpath_out = os.path.join(dirpath_lfp, fname_out)
        Xfilt.to_netcdf(fpath_out)
        
        # Update output table
        chan_info_out.fpath_filtered.iloc[n] = fpath_out
    
    # Add info about the epoching operation to the output table
    chan_info_out.attrs['filt_band'] = Xfilt.attrs['filt_band']
    chan_info_out.attrs['filt_order'] = Xfilt.attrs['filt_order']
    
    return chan_info_out
   
def make_hilbert_batch(chan_epoched_info):
    
    Nchan = len(chan_epoched_info)
    
    # Create output table
    chan_info_out = chan_epoched_info.copy()
    chan_info_out.insert(len(chan_info_out.columns), 'fpath_hilbert', '')
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_epoched_info.iloc[n]
        
        # Load epoched LFP
        fpath_lfp = chan.fpath_epoched
        X = xr.load_dataset(fpath_lfp)['__xarray_dataarray_variable__']
        
        # Hilbert transform
        Xhilb_np = sig.hilbert(X, axis=1)
        Xhilb = xr.zeros_like(X)
        Xhilb.data = Xhilb_np
        
        # Add info about the operation
        Xhilb.attrs['Hilbert'] = ''
            
        # Save the transformed data
        dirpath_lfp = os.path.split(fpath_lfp)[0]
        fname_lfp = os.path.split(fpath_lfp)[1]
        fname_lfp_noext = os.path.splitext(fname_lfp)[0]
        fname_out = '%s_H.nc' % fname_lfp_noext
        fpath_out = os.path.join(dirpath_lfp, fname_out)
        Xhilb.to_netcdf(fpath_out, engine='h5netcdf')
        
        # Update output table
        chan_info_out.fpath_hilbert.iloc[n] = fpath_out
    
    # Add info about the epoching operation to the output table
    chan_info_out.attrs['Hilbert'] = ''
    
    return chan_info_out
 
           
# Perform time-frequency transformation of epoched LFP data
def calc_lfp_tf(chan_epoched_info, win_len=0.5, win_overlap=0.45, fmax=100, need_recalc=False):
    
    print('calc_lfp_tf')

    # Create output table
    chan_info_out = chan_epoched_info.copy()
    chan_info_out.insert(len(chan_info_out.columns), 'fpath_tf', '')
    
    W = None
    
    Nchan = len(chan_epoched_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_epoched_info.iloc[n]
        
        # Input and output paths
        fpath_in = chan.fpath_epoched
        postfix = 'TF_(wlen=%.03f_wover=%.03f_fmax=%.01f)' % (win_len, win_overlap, fmax)
        fpath_out = usf.generate_fpath_out(fpath_in, postfix)
        
        # Check whether it is already calculated
        if os.path.exists(fpath_out) and (need_recalc==False):
            print('Already calculated')
            chan_info_out.fpath_tf.iloc[n] = fpath_out
            continue
        
        # Load epoched LFP
        X = xr.load_dataset(fpath_in)['__xarray_dataarray_variable__']

        fs = X.attrs['fs']

        # Window and overlap in samples
        win_len_samp = round(win_len * fs)
        win_overlap_samp = round(win_overlap * fs)
        
        W = None
        
        Ntrials = X.trial_num.size

        for m in range(Ntrials):
            
            #if (m % 10) == 0:
            #    print('Trial: %i / %i' % (m, Ntrials))

            x = X.isel(trial_num=m)
            
            # TF of the current channel + trial
            (ff, tt, Wcur) = sig.spectrogram(x, fs, mode='complex', window=sig.windows.hamming(win_len_samp), noverlap=win_overlap_samp)
            
            idx = np.round(tt * fs).astype(int)
            tt = X.time.data[idx]
            
            # Frequencies of interest
            idx = (ff < fmax)
            ff = ff[idx]
            Wcur = Wcur[idx,:]
            
            # Allocate output
            if W is None:
                
                coords = {
                    'freq':         ff,
                    'time':         tt,
                    'trial_num':    X.trial_num,
                    'trial_id':     ('trial_num', X.trial_id)
                    }
                Wnan = np.nan * np.ones((len(ff), len(tt), Ntrials), dtype='complex128')
                W = xr.DataArray(Wnan, coords=coords, dims=['freq', 'time', 'trial_num'])
                
            # Store TF of the currenttrial
            W[:,:,m] = Wcur
            
            '''
            plt.plot(100)
            Q = abs(Wcur)
            Q -= Q.mean(axis=1, keepdims=1)
            plt.imshow(Q[7:,:], aspect='auto', origin='lower', extent=[tt[0],tt[-1],ff[7],ff[-1]])
            title_str = 'Trial = %i' % m
            plt.title(title_str)
            plt.draw()
            b = plt.waitforbuttonpress()
            if b==False:
                sys.exit()
            '''

        '''
        plt.figure(100)
        plt.clf()
        Qtotal = (abs(W)**2).mean(axis=2)
        Qpl = abs(W.mean(axis=2))**2
        Qnpl = Qtotal - Qpl
        QQ = [Qtotal, Qpl, Qnpl]
        ss = ['Total  (chan = %i)' % n, 'Evoked', 'Induced']
        f0_id = 10
        for m in range(3):
            plt.subplot(3,1,m+1)
            Q = QQ[m][f0_id:,:]
            Q -= Q.mean(axis=1)
            plt.imshow(Q, aspect='auto', origin='lower', extent=[tt[0],tt[-1],ff[f0_id],ff[-1]])
            plt.title(ss[m])
        idx = (ff >= 20) & (ff <= 35)
        q_beta = Qnpl[idx,:].mean(axis=0)
        idx = (ff >= 40) & (ff <= 60)
        q_gamma = Qnpl[idx,:].mean(axis=0)
        q_beta /= q_beta.mean()
        q_gamma /= q_gamma.mean()
        plt.plot(tt, q_gamma)
        plt.plot(tt, q_beta)
        plt.legend(['gamma', 'beta'])
        plt.title('Channel: %i' % n)
        plt.draw()
        b = plt.waitforbuttonpress()
        if b==False:
            sys.exit()
        '''
        
        # Add info about the operation
        W.attrs = X.attrs.copy()
        W.attrs['tf_win_len'] = win_len
        W.attrs['tf_win_overlap'] = win_overlap
        W.attrs['tf_fmax'] = fmax
        W.attrs['tf_fpath_source'] = fpath_in
            
        # Save the transformed data
        W.to_netcdf(fpath_out, engine='h5netcdf')
        
        # Update output table
        chan_info_out.fpath_tf.iloc[n] = fpath_out
        
    # Add info about the epoching operation to the output table
    chan_info_out.attrs['tf_win_len'] =  win_len
    chan_info_out.attrs['tf_win_overlap'] = win_overlap
    chan_info_out.attrs['tf_fmax'] = fmax
    
    return chan_info_out
        
def calc_TFpow_batch(chan_tf_info, mode='induced'):

    WW = None
    
    Nchan = len(chan_tf_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_tf_info.iloc[n]
        
        # Load epoched TF
        fpath_tf = chan.fpath_tf
        W = xr.load_dataset(fpath_tf, engine='h5netcdf')['__xarray_dataarray_variable__']
        
        # Total power
        w_total = (abs(W)**2).mean(dim='trial_num')
        
        # Evoked power
        w_evoked = abs(W.mean(dim='trial_num'))**2
        
        # Induced power
        w_induced = w_total - w_evoked
        
        if mode=='induced':
            w = w_induced
        elif mode=='evoked':
            w = w_evoked
        elif mode=='total':
            w = w_total
        
        # Allocate output
        if n==0:
            coords = {
                'freq':         W.freq,
                'time':         W.time,
                'chan_num':     range(Nchan),
                'chan_name':    ('chan_num', chan_tf_info.chan_name)                
                }
            Nf = len(W.freq)
            Nt = len(W.time)
            WW = xr.DataArray(np.zeros((Nf,Nt,Nchan)), coords=coords, dims=['freq', 'time', 'chan_num'])
            
        # Update output
        WW[:,:,n] = w
        
    return WW


def calc_TF_ROIs(chan_tf_info, ROI_vec, ROIset_name, TFpow_mode='induced', need_recalc=False):
    
    print('calc_TF_ROIs')

    # Create output table
    chan_info_out = chan_tf_info.copy()
    chan_info_out.insert(len(chan_info_out.columns), 'fpath_ROIs', '')
    
    WROI = None
    
    NROI = len(ROI_vec)
    
    # Generate ROI names
    ROI_names = [''] * NROI
    ROI_names2 = [''] * NROI
    for n in range(NROI):
        ROI_cur = (ROI_vec[n]['tlim'][0]*1000, ROI_vec[n]['tlim'][1]*1000, ROI_vec[n]['flim'][0], ROI_vec[n]['flim'][1])
        ROI_names2[n] = 'ROI_(t=%i-%i)_(f=%i-%i)' % ROI_cur
        ROI_names[n] = ROI_vec[n]['name']
    
    Nchan = len(chan_tf_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_tf_info.iloc[n]
        
        # Input and output paths
        fpath_in = chan.fpath_tf
        postfix = '(ROIset=%s)' % ROIset_name
        fpath_out = usf.generate_fpath_out(fpath_in, postfix)
        
        # Check whether it is already calculated
        if os.path.exists(fpath_out) and (need_recalc==False):
            print('Already calculated')
            chan_info_out.fpath_ROIs.iloc[n] = fpath_out
            continue
        
        # Load epoched TF
        W = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']
        
        # Total / evoked / induced power
        W_total = abs(W)**2
        W_evoked = abs(W.mean(dim='trial_num'))**2
        W_induced = W_total - W_evoked
        
        if TFpow_mode == 'total':
            W1 = W_total
        elif TFpow_mode == 'induced':
            W1 = W_induced
        
        # Allocate output
        coords = {
            'ROI_num':      range(NROI),
            'ROI_name':     ('ROI_num', ROI_names),
            'ROI_name2':    ('ROI_num', ROI_names2),
            'trial_num':    W.trial_num,             
            'trial_id':     ('trial_num', W.trial_id)
            }
        Ntrial = len(W.trial_num)
        WROI = xr.DataArray(np.zeros((NROI,Ntrial)), coords=coords, dims=['ROI_num', 'trial_num'])
        
        # Calculate ROIs
        for m in range(NROI):
            tmask = (W.time >= ROI_vec[m]['tlim'][0]) & (W.time <= ROI_vec[m]['tlim'][1])
            fmask = (W.freq >= ROI_vec[m]['flim'][0]) & (W.freq <= ROI_vec[m]['flim'][1])
            WROI[m,:] = W1.isel(time=tmask, freq=fmask).mean(dim=['time','freq'])
        
        # Add info about the operation
        WROI.attrs = W.attrs.copy()
        W.attrs['ROIset_name'] = ROIset_name
        W.attrs['ROI_vec'] = ROI_vec
        W.attrs['TFpow_mode'] = TFpow_mode
        W.attrs['ROI_fpath_source'] = fpath_in
            
        # Save the transformed data
        WROI.to_netcdf(fpath_out, engine='h5netcdf')
        
        # Update output table
        chan_info_out.fpath_ROIs.iloc[n] = fpath_out
        
    # Add info about the epoching operation to the output table
    chan_info_out.attrs['ROIset_name'] = ROIset_name
    chan_info_out.attrs['ROI_vec'] = ROI_vec
    chan_info_out.attrs['TFpow_mode'] = TFpow_mode
    
    return chan_info_out

def calc_TF_ROI_trial_avg(chan_tfROI_info):

    WW = None
    
    Nchan = len(chan_tfROI_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_tfROI_info.iloc[n]
        
        # Load epoched TF
        fpath_in = chan.fpath_ROIs
        W = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']
        
        # Allocate output
        if n==0:
            coords = {
                'ROI_num':      W.ROI_num,
                'ROI_name':     ('ROI_num', W.ROI_name),
                'ROI_name2':    ('ROI_num', W.ROI_name2),
                'chan_num':     range(Nchan),
                'chan_name':    ('chan_num', chan_tfROI_info.chan_name)
                }
            NROI = len(W.ROI_num)
            WW = xr.DataArray(np.zeros((NROI,Nchan)), coords=coords, dims=['ROI_num', 'chan_num'])
            
        # Update output
        WW[:,n] = W.mean(dim='trial_num')
        
    return WW    
    
def calc_stim_TFpow_batch(chan_tf_info, trial_info, mode='induced'):

    # Get all stimulus codes
    stim_codes_all = []
    for sess_num in range(Nsess):
        stim_codes_all += trial_info[sess_num]['trial_tbl'].stim1_code.tolist()
    stim_codes_all_uni = np.unique(stim_codes_all)
    Ncodes = len(stim_codes_all_uni)
    
    WW = None
    
    Nchan = len(chan_tf_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = chan_tf_info.iloc[n]
        
        # Load epoched TF
        fpath_tf = chan.fpath_tf
        W = xr.load_dataset(fpath_tf, engine='h5netcdf')['__xarray_dataarray_variable__']
        
        # Trial table associated with the same session as the channel
        trial_tbl = usf.get_trial_info_by_sess(trial_info, chan.subj_name, chan.sess_id)['trial_tbl']
        
        for code_num in range(Ncodes):
            
            # Tf matrix subset for the trials with the given stimulus code
            trial_mask = np.array(trial_tbl.stim1_code == stim_codes_all_uni[code_num])            
            W1 = W[:,:,trial_mask]
        
            # Total power
            w_total = (abs(W1)**2).mean(dim='trial_num')        
            # Evoked power
            w_evoked = abs(W1.mean(dim='trial_num'))**2        
            # Induced power
            w_induced = w_total - w_evoked
            
            if mode=='induced':
                w = w_induced
            elif mode=='evoked':
                w = w_evoked
            elif mode=='total':
                w = w_total
        
            # Allocate output
            if WW is None:
                coords = {
                    'freq':         W.freq,
                    'time':         W.time,
                    'stim_num':     range(Ncodes),
                    'stim_code':    ('stim_num', stim_codes_all_uni),
                    'chan_num':     range(Nchan),
                    'chan_name':    ('chan_num', chan_tf_info.chan_name)                
                    }
                Nf = len(W.freq)
                Nt = len(W.time)
                WW = xr.DataArray(np.zeros((Nf,Nt,Ncodes,Nchan)), coords=coords, dims=['freq', 'time', 'stim_num', 'chan_num'])
            
            # Update output
            WW[:,:,code_num,n] = w
        
    return WW  


def _calc_dfg_TFpow_inner(X_in, subtract_mean):
    TFpow = X_in['TF'].copy()
    if subtract_mean:
        TFpow -= TFpow.mean(dim='trial_num')
    TFpow = np.abs(TFpow)**2
    return xr.Dataset({'TFpow': TFpow})    

        
def calc_dfg_TFpow(dfg_in, subtract_mean=True):

    # Step name. params, vars, fpath column
    proc_step_name = 'Calculate TF power'
    params = {'subtract_mean': subtract_mean}
    vars_new_descs = {'TFpow': 'Time-frequency LFP power'}
    fpath_data_column = 'fpath_TFpow'

    # Function that converts the parameters dict to the form suitable
    # for storing into a processing step description
    def gen_proc_step_params(par):
        par_out = {
            'subtract_mean': {
                'desc': 'Subtract trial-averaged complex mean before the power calculation',
                'value': str(subtract_mean)},
        }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath(fpath_in, params):
        if params['subtract_mean']:
            return fpath_in.replace('TF', 'TFpow_noERP')
        else:
            return fpath_in.replace('TF', 'TFpow')
    
    # Call the inner procedure for each inner dataset of the DataFileGroup
    dfg_out = dfg.apply_dfg_inner_proc(
            dfg_in, _calc_dfg_TFpow_inner, params, proc_step_name,
            gen_proc_step_params, fpath_data_column, gen_fpath, vars_new_descs)
    
    return dfg_out
    
        
        
        
    
    
    
    
    
    
