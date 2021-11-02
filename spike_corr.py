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


def calc_spike_TF_PLV_batch(cell_epoched_info, chan_tf_info, t_range):
    
    print('calc_spike_TF_PLV')
    
    # Create output table
    spPLVf_info_out = chan_tf_info.copy()
    spPLVf_info_out.insert(len(spPLVf_info_out.columns), 'fpath_spPLVf', '')
        
    Nchan = len(chan_tf_info)
    
    for chan_num in range(Nchan):
        
        print('Channel: %i / %i' % (chan_num, Nchan))
        
        chan = chan_tf_info.iloc[chan_num]
        
        # Load epoched TF
        fpath_tf = chan.fpath_tf
        W = xr.load_dataset(fpath_tf, engine='h5netcdf')['__xarray_dataarray_variable__']
        
        # Cells with the same subject + session as the current channel
        mask = (cell_epoched_info.subj_name == chan.subj_name) & (cell_epoched_info.sess_id == chan.sess_id) 
        cell_vec_cur = cell_epoched_info[mask]
        
        Nfreq = len(W.freq)
        Ncell = len(cell_vec_cur)
        Ntrials = len(W.trial_num)
    
        coords = {
            'freq':         W.freq,
            'cell_id':      cell_vec_cur.cell_id,
            'cell_name':    ('cell_id', cell_vec_cur.cell_name)
            }
        Xnan = np.nan * np.ones((Nfreq, Ncell), dtype='complex128')    
        data_out_w = xr.DataArray(Xnan, coords=coords, dims=['freq', 'cell_id'])
        
        coords = {
            'cell_id':      cell_vec_cur.cell_id,
            'cell_name':    ('cell_id', cell_vec_cur.cell_name)
            }
        Xnan = np.nan * np.ones(Ncell)    
        data_out_N = xr.DataArray(Xnan, coords=coords, dims=['cell_id'])
        
        for cell_num in range(Ncell):
            
            print('Cell: %i / %i' % (cell_num, Ncell))
        
            cell = cell_vec_cur.iloc[cell_num]
            
            # Load spiketrain
            with open(cell.fpath_epoched, 'rb') as fid:
                st_epoched = pk.load(fid).values
            
            ww = np.zeros(Nfreq, dtype='complex128')
            Nspikes = 0
            
            for trial_num in range(Ntrials):
                
                # Spike times for the current trial
                st_cur = st_epoched[trial_num]
                
                # Select spikes from the given time range
                mask = (st_cur >= t_range[0]) & (st_cur < t_range[1])
                st_cur = st_cur[mask]
                if len(st_cur)==0:
                    continue
                
                # Accumulate spike-triggerred LFP phase
                w = W.isel(trial_num=trial_num).interp(time=st_cur)
                w /= np.abs(w)
                ww += w.sum(dim='time').data
                Nspikes += len(w.time)
                
            ww /= Nspikes
            
            # Store into output arrays
            data_out_w[:,cell_num] = ww
            data_out_N[cell_num] = Nspikes
            
        # Collect the output dataset
        data_vars = {
            'PLV':      data_out_w,
            'Nspikes':  data_out_N
            }
        coords = {
            'freq':         W.freq,
            'cell_id':      cell_vec_cur.cell_id,
            'cell_name':    ('cell_id', cell_vec_cur.cell_name)
            }
        data_out = xr.Dataset(data_vars=data_vars, coords=coords)
        
        # Add info about the operation
        data_out.attrs = W.attrs.copy()
        data_out.attrs['spPLV_t_range'] = t_range
        data_out.attrs['spPLV_fpath_source'] = fpath_tf
            
        # Save the transformed data
        dirpath_out = os.path.split(fpath_tf)[0]
        fname_out = os.path.split(fpath_tf)[1]
        fname_out_noext = os.path.splitext(fname_out)[0]
        fname_out = '%s_spPLV_(t=%.03f-%.03f).nc' % (fname_out_noext, t_range[0], t_range[1])
        fpath_out = os.path.join(dirpath_out, fname_out)
        data_out.to_netcdf(fpath_out, engine='h5netcdf')
        
        # Update output table
        spPLVf_info_out.fpath_spPLVf.iloc[chan_num] = fpath_out
    
    # Add info about the epoching operation to the output table
    spPLVf_info_out.attrs['spPLV_t_range'] = t_range
    
    return spPLVf_info_out


def calc_spike_TF_PLV_trial_avg_batch(spPLVf_info):
# Load results of calc_spike_TF_PLV_by_trial_batch and average them over trials
# Also calculate significance of inter-trial phase coherence
    
    WW = None
    
    Nchan = len(spPLVf_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = spPLVf_info.iloc[n]
        
        # Load PLV data
        fpath_in = chan.fpath_spPLVf
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
    

# Load precalculated spike-LFP PLV data for each channel and average it over cells
def calc_spPLVf_cell_avg(spPLVf_info, chans_used=None):
    
    if chans_used is not None:
        chan_mask = [x in chans_used for x in spPLVf_info.chan_id]
        spPLVf_info = spPLVf_info[chan_mask]
    
    Nchan = len(spPLVf_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = spPLVf_info.iloc[n]
        
        # Load spike-LPF PLV data
        fpath_in = chan.fpath_spPLVf
        data_in = xr.load_dataset(fpath_in, engine='h5netcdf')['PLV']
        
        if chans_used is not None:
            cell_chan_idx = spk.parse_chan_name(data_in.cell_name.data)['chan_id']
            cell_mask = [x in chans_used for x in cell_chan_idx]
            data_in = data_in.isel(cell_id=cell_mask)
        
        # Average over cells
        X = np.abs(data_in).mean(dim='cell_id')
        #X = np.abs(data_in.mean(dim='cell_id'))
        
        # Allocate output
        if n==0:
            Nfreq = len(X.freq)
            coords = {
                'chan_num':     range(Nchan),
                'chan_name':    ('chan_num', spPLVf_info.chan_name),
                'freq':         X.freq,
                }
            data_out = xr.DataArray(np.zeros((Nchan,Nfreq)), coords=coords, dims=['chan_num', 'freq'])
            
        # Update output
        data_out[n,:] = X
        
    return data_out


# Make table of chan-cell pairs with abs PLV values averaged over given freq bands
def calc_spPLV_fband_chan_unroll(spPLVf_info, fbands):
    
    # Create output table
    col_names =['subj_name', 'sess_id', 'chan_id', 'cell_id', 'chan_name', 'cell_name', 'Nspikes']
    for m in range(len(fbands)):
        col_name = 'spPLV_(%i-%i_Hz)' % (fbands[m][0], fbands[m][1])
        col_names.append(col_name)
    tbl_out = pd.DataFrame(columns=col_names)
    
    Nchan = len(spPLVf_info)
    
    chan_num = 0
    
    for chan_num in range(Nchan):
        
        print('%i / %i' % (chan_num, Nchan))
        
        chan = spPLVf_info.iloc[chan_num]
        
        # Load spike-LPF PLV data
        fpath_in = chan.fpath_spPLVf
        data_in = xr.load_dataset(fpath_in, engine='h5netcdf')
        PLV = data_in['PLV']
        Nspikes = data_in['Nspikes']
        
        # Calculate mean absolute PLV in the freq bands
        PLV_fbands = []
        for m in range(len(fbands)):
            mask = (PLV.freq >= fbands[m][0]) & (PLV.freq <= fbands[m][1])
            PLV_fbands_cur = np.abs(PLV).isel(freq=mask).mean(dim='freq')
            PLV_fbands.append(PLV_fbands_cur)
            
        Ncell = len(PLV.cell_id)
        
        for cell_num in range(Ncell):
            
            row_data = [chan.subj_name, chan.sess_id, chan.chan_id, PLV.cell_id.data[cell_num],
                        chan.chan_name, PLV.cell_name.data[cell_num], Nspikes.data[cell_num]]
            
            for m in range(len(fbands)):
                row_data.append(PLV_fbands[m].data[cell_num])
            
            row = pd.DataFrame(data=[row_data], columns=col_names)
            tbl_out = tbl_out.append(row)
            
    return tbl_out
    
# Calculare cross-correlogram of two firing rate signals, excluding jittered covariance
#@profile
def calc_rvec_cov_nojit(rvec1, rvec2, nbins_jit, niter_jit, lag_range=None):

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

#@profile
def calc_rvec_cov_nojit_batch(rvec_info, nbins_jit, niter_jit, lag_range=None):
    
    # Create output table
    col_names =['subj_name', 'sess_id', 'fpath_rvec_cov']
    tbl_out = pd.DataFrame(columns=col_names)
    
    # Unique session identifiers
    sess_idx_uni = rvec_info.sess_id.unique()
    Nsess = len(sess_idx_uni)
    
    #plt.figure()
    
    for sess_num in range(Nsess):
        
        print(f'Session: {sess_num} / {Nsess}')
        
        sess_id = sess_idx_uni[sess_num]
        
        # Get cell info for the current session
        rvec_info_sess = rvec_info[rvec_info.sess_id == sess_id]
        Ncells = len(rvec_info_sess)
        
        # Get session folder
        dirpath_sess = rvec_info_sess.iloc[0].fpath_spikes
        for n in range(4):
            dirpath_sess = os.path.split(dirpath_sess)[0]

        C0 = None
        
        for c1 in range(Ncells):
        #for c1 in range(6):
            
            # Load rvec for the 1-st cell in a pair
            fpath_in = rvec_info_sess.iloc[c1].fpath_rvec
            R1 = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']
            
            Ntrials = len(R1.trial_num)
            
            for c2 in range(c1,Ncells):
            #for c2 in range(c1,6):
                
                print(f'Cells: {c1} / {Ncells}  {c2} / {Ncells}')
                
                # Load rvec for the 2-nd cell in a pair
                fpath_in = rvec_info_sess.iloc[c2].fpath_rvec
                R2 = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']

                tt1 = timer()
                
                for trial_num in range(Ntrials):
                #for trial_num in range(100):
                    
                    #print(f'Trial: {trial_num} / {Ntrials} ')
                    
                    # Firing rate vectors of two cells in the current trial
                    rvec1 = R1.data[trial_num,:]
                    rvec2 = R2.data[trial_num,:]
                    
                    # Skip empty spiketrains
                    if ~np.any(rvec1) and ~np.any(rvec2):
                        continue
                    
                    # Cross-correlation
                    c, c_jit, c_nojit, lags = calc_rvec_cov_nojit(rvec1, rvec2, nbins_jit, niter_jit, lag_range)
                    Nsamples = len(c)
                    
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
                    
                    # Allocate the ndarray
                    if C0 is None:
                        C0 = np.nan * np.ones((Ncells, Ncells, Nsamples, Ntrials))
                    
                    # Update the output
                    C0[c1, c2, :, trial_num] = c_nojit
                    
                #print(f'dt = {timer()-tt1:.05f}')

        # Create xarray output
        coords = {
            'cell1_num':    range(Ncells),
            'cell1_name':   ('cell1_num', rvec_info_sess.cell_name),
            'cell2_num':    range(Ncells),
            'cell2_name':   ('cell2_num', rvec_info_sess.cell_name),
            'sample_num':   range(Nsamples),
            'lags':         ('sample_num', lags * R1.rvec_dt),
            'trial_num':    range(Ntrials),
            'trial_id':     ('trial_num', R1.trial_id)
            }
        dims = ['cell1_num', 'cell2_num', 'sample_num', 'trial_num']        
        C = xr.DataArray(C0, coords=coords, dims=dims)
        
        # Add info about the operation
        C.attrs = R1.attrs.copy()
        C.attrs.pop('epoching_fpath_source')
        C.attrs.pop('rvec_fpath_source')
        C.attrs['rcov_nbins_jit'] = nbins_jit
        C.attrs['rcov_niter_jit'] = niter_jit
        C.attrs['rcov_lag_range'] = lag_range

        # Save the result
        dirpath_out = dirpath_sess
        fname_out = os.path.split(fpath_in)[1]
        fname_out_noext = os.path.splitext(fname_out)[0]
        fname_out = f'{fname_out_noext}_cov_(bins={nbins_jit}_iter={niter_jit}_lags={Nsamples}).nc'
        fpath_out = os.path.join(dirpath_out, fname_out)
        C.to_netcdf(fpath_out, engine='h5netcdf')

        # Update output table
        subj_name = rvec_info_sess.iloc[0].subj_name
        entry = {'subj_name': subj_name, 'sess_id': sess_id, 'fpath_rvec_cov': fpath_out}
        tbl_out = tbl_out.append(entry, ignore_index=True)

    # Add info about the operation to the table
    tbl_out.attrs = rvec_info.attrs.copy()
    tbl_out.attrs['rcov_nbins_jit'] = nbins_jit
    tbl_out.attrs['rcov_niter_jit'] = niter_jit
    tbl_out.attrs['rcov_lag_range'] = lag_range
            
    return tbl_out

#rcov_info = calc_rvec_cov_nojit_batch(rvec_info, nbins_jit=5, niter_jit=1000, lag_range=(-15,15))

'''
N = 70
mode = 'full'
niter = 10000

lags = sig.correlation_lags(N,N,mode)
nlags = 15

t = np.linspace(0,1,70)
f = 2
a = 0.6

x = (1-a) * np.random.normal(size=N) + a * np.sin(2*np.pi*f*t)
y = (1-a) * np.random.normal(size=N) + a * np.cos(2*np.pi*f*t)

t0 = timer()    
for n in range(niter):        
    c1 = sig.correlate(y, x, mode=mode, method='direct')
    #c = sig.correlate(x, y, mode=mode)      
dt = timer() - t0
print(f't = {dt:.05f}')

t0 = timer()    
for n in range(niter):        
    c2 = pycor.ucorrelate(x, y, nlags)
dt = timer() - t0
print(f't = {dt:.05f}')

mask = (lags>=0)
lags_pos = lags[mask]

plt.figure()
plt.plot(lags_pos, c1[mask])
plt.plot(lags_pos[:nlags], c2)
'''

#def rcov_info_visualize(rcov_info):
    
    


