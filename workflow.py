# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import trial_manager as trl
import spiketrain_manager as spk
import lfp
import spike_corr as spcor
import pickle as pk
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np


# Root paths for the data and the processing results
dirpath_data_root = 'H:\\WORK\\Camilo\\data'
dirpath_res_root = 'H:\\WORK\\Camilo\\Processing'


# Run arbitrary function and save the result, or load the previously calculated result
def run_or_load(f, fname_cache, recalc=False):
    
    fpath_cache = os.path.join(dirpath_res_root, fname_cache)
    
    if recalc or not os.path.exists(fpath_cache):
        res = f()
        with open(fpath_cache, 'wb') as fid:
            pk.dump(res, fid)
    else:
        with open(fpath_cache, 'rb') as fid:
            res = pk.load(fid)
        
    return res
    

# Create trial info 
trial_info = run_or_load(
        lambda: trl.create_trial_info(dirpath_data_root),
        'trial_info', recalc=False)

# Create cell info
# TODO: discart cells with unrealisrically high firing rates
cell_info = run_or_load(
        lambda: spk.create_spiketrain_info(dirpath_data_root),
        'cell_info', recalc=False)

# Select correct trials
# TODO: trials with the typical delay length
for n in range(len(trial_info)):
    trial_info[n]['trial_tbl'] = trl.select_correct_trials(trial_info[n]['trial_tbl'])

# Get channels associated with cells
chan_info = lfp.get_cell_channels(cell_info)

lock_ev = 'stim1_t'
time_win = [-1,3]

# Epoch LFPs
fname_out_ep = 'chan_epoched_info_(ev=%s)_(t=%.02f-%.02f)' % (lock_ev, time_win[0], time_win[1])
chan_epoched_info = run_or_load(
        lambda: lfp.epoch_lfp_data_batch(chan_info, trial_info, lock_event=lock_ev, time_win=time_win),
        fname_out_ep, recalc=False)

# Epoch spiketrains
fname_out_spikes_ep = 'cell_epoched_info_(ev=%s)_(t=%.02f-%.02f)' % (lock_ev, time_win[0], time_win[1])
cell_epoched_info = run_or_load(
        lambda: spk.epoch_spike_data_batch(cell_info, trial_info, lock_event=lock_ev, time_win=time_win),
        fname_out_spikes_ep, recalc=False)

'''
# Filter the data
freq_band = [20,35]
fname_out_filt = '%s_(f=%.01f-%.01f)' % (fname_out_ep, freq_band[0], freq_band[1])
chan_filt_info = run_or_load(
        lambda: lfp.filt_epoched_data_batch(chan_epoched_info, freq_band),
        fname_out_filt, recalc=False)

# Hilbert transform
fname_out_hilb = '%s_H' % fname_out_filt
chan_hilb_info = run_or_load(
        lambda: lfp.make_hilbert_batch(chan_filt_info),
        fname_out_hilb, recalc=False)
'''

# TF
win_len = 0.500
#win_overlap = 0.450
#fmax = 100
win_overlap = 0.499
fmax = 60
fname_out_tf = '%s_TF_(wlen=%.03f_wover=%.03f_fmax=%.01f)' % (fname_out_ep, win_len, win_overlap, fmax)
chan_tf_info = run_or_load(
        lambda: lfp.calc_lfp_tf(chan_epoched_info, win_len=win_len, win_overlap=win_overlap, fmax=fmax),
        fname_out_tf, recalc=False)

# Spike-LFP PLV (freq)
t_range = (0.5, 1.2)
fname_out_spPLVf = '%s_spPLVf_(t=%.03f-%.03f)' % (fname_out_tf, t_range[0], t_range[1])
spPLVf_info = run_or_load(
        lambda: spcor.calc_spike_TF_PLV_batch(cell_epoched_info, chan_tf_info, t_range),
        fname_out_spPLVf, recalc=True)

X = spcor.calc_spPLVf_cell_avg(spPLVf_info)    
plt.figure(); plt.plot(X.freq, X.mean(dim='chan_num'))

fbands = [[12,22], [24,36], [14,36]]
tbl_spPLV_fband = spcor.calc_spPLV_fband_chan_unroll(spPLVf_info, fbands)

x = tbl_spPLV_fband.Nspikes / 500
y1 = tbl_spPLV_fband['spPLV_(14-36_Hz)']
#y2 = tbl_spPLV_fband['spPLV_(24-36_Hz)']
d = spk.parse_chan_name(tbl_spPLV_fband.cell_name.tolist())
chan_idx_chan = np.array(tbl_spPLV_fband.chan_id, dtype=int)
chan_idx_cell = np.array(d['chan_id'], dtype=int)
mask = (chan_idx_chan == chan_idx_cell)
plt.figure();
plt.plot(x, y1, '.')
plt.plot(x[mask], y1[mask], '.')
#plt.plot(x+0.1, y2, '.')

'''
# ERP
ERP = lfp.calc_ERP_batch(chan_epoched_info)
plt.figure()
plt.imshow(ERP, aspect='auto', extent=[-1,3,52,0])

# Test filering
Xfilt = np.zeros((52,4000))
for n in range(52):
    fpath_in = chan_filt_info.fpath_filtered.iloc[n]
    Xfilt_cur = xr.load_dataset(fpath_in)['__xarray_dataarray_variable__']
    Xfilt[n,:] = Xfilt_cur.mean(dim='trial_num')
    #plt.imshow(Xfilt, aspect='auto', extent=[-1,3,Xfilt.shape[0],0])
plt.figure()
plt.imshow(Xfilt, aspect='auto', extent=[-1,3,52,0])
'''

'''
fpath_in = chan_hilb_info.fpath_hilbert.iloc[0]
X = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']
#X = X - X.mean(dim='trial_num')
plt.figure()
plt.imshow(np.abs(X), aspect='auto', extent=[-1,3,X.shape[0],0])
'''

'''
X = np.zeros((52,4000))
for n in range(52):
    fpath_in = chan_hilb_info.fpath_hilbert.iloc[n]
    x = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']
    xavg= x.mean(dim='trial_num')
    #x = x - x.mean(dim='trial_num')
    x = abs(x)**2
    X[n,:] = x.mean(dim='trial_num') - abs(xavg)**2
    #plt.imshow(Xfilt, aspect='auto', extent=[-1,3,Xfilt.shape[0],0])
X1 = X / np.mean(X[:,0:999], axis=1, keepdims=True)
plt.figure()
#plt.imshow(X1, aspect='auto', extent=[-1,3,52,0])
plt.plot(X1.mean(axis=0))
'''

# TF
'''
W = lfp.calc_TFpow_batch(chan_tf_info, mode='evoked')
W1 = W / W.mean(dim='time')
Wavg = W1.mean(dim='chan_num')
plt.figure()
Wvis = Wavg[:,:]
plt.imshow(Wvis, aspect='auto', extent=[W.time[0],W.time[-1],W.freq[0],W.freq[-1]], origin='lower')
'''

