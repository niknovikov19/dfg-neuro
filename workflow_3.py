# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import lfp
import pickle as pk
import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trial_manager as trl
import spiketrain_manager as spk
#import firing_rate as fr
import spike_corr as spcor
import useful as usf
import vis


# Root paths for the data and the processing results
dirpath_data_root = 'H:\\WORK\\Camilo\\data'
dirpath_res_root = 'H:\\WORK\\Camilo\\Processing_Pancake_2sess_allchan'


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
    

# Used channels
#chans_used = list(range(1,33))          # area8
#chans_used = list(range(65,97))         # dlPFC


# Create trial info 
trial_info = run_or_load(
        lambda: trl.create_trial_info(dirpath_data_root),
        'trial_info', recalc=False)

# Select correct trials
# TODO: trials with the typical delay length
for n in range(len(trial_info)):
    trial_info[n]['trial_tbl'] = trl.select_correct_trials(trial_info[n]['trial_tbl'])


# Create cell info
# TODO: discard cells with unrealisrically high firing rates
cell_info = run_or_load(
        lambda: spk.create_spiketrain_info(dirpath_data_root),
        'cell_info', recalc=False)


# Get all channels
chan_info = usf.get_all_channels(dirpath_data_root)

# Mask for used channels
#chan_mask = [id in chans_used for id in chan_info.chan_id]


lock_ev = 'stim1_t'
time_win = [-1,3]

# Epoch LFPs
fname_out_LFP_ep = 'chan_all_epoched_info_(ev=%s)_(t=%.02f-%.02f)' % (lock_ev, time_win[0], time_win[1])
chan_epoched_info = run_or_load(
        lambda: lfp.epoch_lfp_data_batch(chan_info, trial_info, lock_event=lock_ev, time_win=time_win),
        fname_out_LFP_ep, recalc=False)

# Epoch spiketrains
fname_out_spikes_ep = 'cell_epoched_info_(ev=%s)_(t=%.02f-%.02f)' % (lock_ev, time_win[0], time_win[1])
cell_epoched_info = run_or_load(
        lambda: spk.epoch_spike_data_batch(cell_info, trial_info, lock_event=lock_ev, time_win=time_win),
        fname_out_spikes_ep, recalc=False)


# TF
win_len = 0.500
win_overlap = 0.450
fmax = 100
#win_overlap = 0.499
#fmax = 60
fname_out_tf = '%s_TF_(wlen=%.03f_wover=%.03f_fmax=%.01f)' % (fname_out_LFP_ep, win_len, win_overlap, fmax)
chan_tf_info = run_or_load(
        lambda: lfp.calc_lfp_tf(chan_epoched_info, win_len=win_len, win_overlap=win_overlap, fmax=fmax),
        fname_out_tf, recalc=False)

# TF ROIs
ROIset_name = 'ROIs_beta_gamma'
ROI_vec = [
        {'name': 'beta_del1',    'flim': [15, 40],  'tlim': [0.5, 1.2]},
        {'name': 'beta_del11',   'flim': [15, 40],  'tlim': [0.5, 0.85]},
        {'name': 'beta_del12',   'flim': [15, 40],  'tlim': [0.85, 1.2]},
        {'name': 'gamma_del1',   'flim': [60, 100], 'tlim': [0.5, 1.2]},
        {'name': 'gamma_del11',  'flim': [60, 100], 'tlim': [0.5, 0.85]},
        {'name': 'gamma_del12',  'flim': [60, 100], 'tlim': [0.85, 1.2]},
        {'name': 'beta_stim',    'flim': [15, 40],  'tlim': [0.05, 0.3]},
        {'name': 'beta1_stim',   'flim': [15, 25],  'tlim': [0.05, 0.3]},
        {'name': 'beta2_stim',   'flim': [25, 40],  'tlim': [0.05, 0.3]},
        {'name': 'gamma_stim',   'flim': [60, 100], 'tlim': [0.05, 0.3]},
        {'name': 'beta_bl',      'flim': [15, 40],  'tlim': [-0.7, -0.2]},
        {'name': 'beta1_bl',     'flim': [15, 25],  'tlim': [-0.7, -0.2]},
        {'name': 'beta2_bl',     'flim': [25, 40],  'tlim': [-0.7, -0.2]},
        {'name': 'gamma_bl',     'flim': [60, 100], 'tlim': [-0.7, -0.2]},
        ]
TFpow_mode = 'induced'
fname_out_tfROI = '%s_(ROIset=%s_pow=%s)' % (fname_out_tf, ROIset_name, TFpow_mode)
tfROI_info = run_or_load(
        lambda: lfp.calc_TF_ROIs(chan_tf_info, ROI_vec, ROIset_name, TFpow_mode),
        fname_out_tfROI, recalc=False)


# Spike-LFP PLV (freq)
t_range = (0.5, 1.2)
fname_out_spPLVf = '%s_spPLVf_(t=%.03f-%.03f)' % (fname_out_tf, t_range[0], t_range[1])
spPLVf_info = run_or_load(
        lambda: spcor.calc_spike_TF_PLV_batch(cell_epoched_info, chan_tf_info, t_range),
        fname_out_spPLVf, recalc=False)

# Firing rate vectors
t_range = (0.5, 1.2)
dt = 10 * 1e-3
fname_out_rvec = '%s_rvec_(t=%i-%i_dt=%i)' % (fname_out_spikes_ep, t_range[0]*1e3, t_range[1]*1e3, dt*1e3)
rvec_info = run_or_load(
        lambda: fr.calc_frate_vec_batch(cell_epoched_info, t_range, dt),
        fname_out_rvec, recalc=False)

# Spike train correlation
nbins_jit=5
niter_jit=50
lag_range=(-15,15)
fname_out_rcov = f'{fname_out_rvec}_cov_(bins={nbins_jit}_iter={niter_jit}_lags={len(lag_range)})'
rcov_info = run_or_load(
        lambda: spcor.calc_rvec_cov_nojit_batch(rvec_info, nbins_jit=nbins_jit, niter_jit=niter_jit, lag_range=lag_range),
        fname_out_rcov, recalc=False)


# Beta in the 1-st half vs. beta in the 2-nd half of the delay (trials for each channel)
vis.show_chan_TFROI_pair(tfROI_info, ('beta_del11', 'beta_del12'))

#X = spcor.calc_spPLVf_cell_avg(spPLVf_info, chans_used)    
#plt.figure(); plt.plot(X.freq, X.mean(dim='chan_num'))

# Load LFP power ROI's
WROI = lfp.calc_TF_ROI_trial_avg(tfROI_info)

x_gamma_stim = np.array(WROI[WROI.ROI_name=='gamma_stim',:]).ravel()
x_gamma_del12 = np.array(WROI[WROI.ROI_name=='gamma_del12',:]).ravel()
x_gamma_bl = np.array(WROI[WROI.ROI_name=='gamma_bl',:]).ravel()
x_beta_del1 = np.array(WROI[WROI.ROI_name=='beta_del1',:]).ravel()
x_beta_del11 = np.array(WROI[WROI.ROI_name=='beta_del11',:]).ravel()
x_beta_del12 = np.array(WROI[WROI.ROI_name=='beta_del12',:]).ravel()
x_beta_bl = np.array(WROI[WROI.ROI_name=='beta_bl',:]).ravel()


x_gamma_stim_ERSP = (x_gamma_stim - x_gamma_bl) / x_gamma_bl
x_gamma_del_ERSP = (x_gamma_del12 - x_gamma_bl) / x_gamma_bl
x_beta_del_ERSP = (x_beta_del12 - x_beta_bl) / x_beta_bl

'''
idx = np.argsort(x_gamma_stim_ERSP)
x_gamma_stim_ERSP = x_gamma_stim_ERSP[idx]
x_gamma_del_ERSP = x_gamma_del_ERSP[idx]
x_beta_del_ERSP = x_beta_del_ERSP[idx]

plt.figure()
plt.plot(x_gamma_stim_ERSP, '.')
plt.plot(x_beta_del_ERSP, '.')
plt.plot(x_gamma_del_ERSP, '.')
'''

plt.figure()
plt.plot(x_beta_del11, x_beta_del12, '.')

# Load firing rates
R = fr.calc_rvec_trial_avg(rvec_info)

# Average over samples
rr = R.mean(dim='sample_num')

# Associate cells with channels
cell_chan_idx = usf.get_chan_idx_by_cell_names(R.cell_name.data, WROI.chan_name.data, cell_info, chan_info)

# Firing rate vs. LFP power
plt.figure()
plt.plot(np.log(rr), np.log(x_gamma_del12[cell_chan_idx]) - np.log(x_gamma_bl[cell_chan_idx]), '.')
plt.plot(np.log(rr), np.log(x_beta_del12[cell_chan_idx]) - np.log(x_beta_bl[cell_chan_idx]), '.')
#plt.plot(np.log(rr), np.log(x_gamma_del12[cell_chan_idx]), '.')
#plt.plot(np.log(rr), np.log(x_beta_del12[cell_chan_idx]), '.')
plt.xlabel('Firing rate')
plt.ylabel('LFP power')
plt.legend(['gamma', 'beta'])


dirpath_out = 'H:\\WORK\\Camilo\\Processing\\Pancake_delay_r_vs_beta_log'
#plot_rate_vs_TFpow_by_cell(rvec_info, tfROI_info, dirpath_out, 'beta_del12')


'''
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

# TFpow plot
'''
W = lfp.calc_TFpow_batch(chan_tf_info, mode='induced')
mask_bl = (W.time < -0.1)
Wbl = W.isel(time=mask_bl).mean(dim='time')
#W1 = W
#W1 = W / W.mean(dim='time')
#W1 = np.log(W) - np.log(Wbl)
W1 = np.log(W)
#W1 = W1.isel(chan_num=chan_mask)
Wavg = W1.mean(dim='chan_num')
plt.figure()
plt.imshow(Wavg, aspect='auto', extent=[W.time[0],W.time[-1],W.freq[0],W.freq[-1]], origin='lower')
#W2 = W1.isel(time = ((W1.time > 0.5) & (W1.time < 1.2))).mean(dim=['time','chan_num'])
#W2 = W1.isel(time = ((W1.time > 1.6) & (W1.time < 10))).mean(dim=['time','chan_num'])
#W2 = W1.mean(dim=['time','chan_num'])
#W2 = np.log(W2)
#plt.plot(W1.freq, W2)
'''


'''
for n in range(len(W1.chan_num)):
    W2 = W1.isel(chan_num=n)
    plt.imshow(W2, aspect='auto', extent=[W.time[0],W.time[-1],W.freq[0],W.freq[-1]], origin='lower')
    plt.title(W1.chan_name.data[n])
    plt.draw()
    plt.waitforbuttonpress()
'''
    

'''
# Load rate covariances
C = xr.load_dataset(rcov_info.iloc[0].fpath_rvec_cov, engine='h5netcdf')['__xarray_dataarray_variable__']

# Average over trials and central lag bins
lag_idx = [14, 15, 16]
Cz = C.isel(sample_num=lag_idx).mean(dim='sample_num').mean(dim='trial_num')

# Normalize and set zeros to the diagonal
Czn = Cz.copy()
Ncell = len(Cz.cell1_num)
for c1 in range(Ncell):
    for c2 in range(Ncell):
        Czn[c1,c2] /= np.sqrt((Czn[c1,c1] * Czn[c2,c2]))
for c in range(Ncell):
    Czn[c,c] = np.nan

# Get channel name for each cell    
ch_vec = np.nan * np.ones(Ncell)
for c in range(Ncell):
    cell_name = Cz.cell1_name[c]
    ci = cell_info[cell_info.cell_name==cell_name]
    ch_vec[c] = ci.chan_id.item()

# Discard pairs from the same channel
Cz2 = Cz.copy()
Czn2 = Czn.copy()
for c1 in range(Ncell):
    for c2 in range(Ncell):
        if ch_vec[c1]==ch_vec[c2]:
            Cz2[c1,c2] = np.nan
            Czn2[c1,c2] = np.nan
    
plt.figure()
plt.imshow(Czn2, vmin=-1, vmax=1)
plt.colorbar()

Czn_vec = Czn2.copy().data.ravel()
Czn_vec = np.sort(Czn_vec[Czn_vec != np.nan])


plt.figure()
plt.plot(Czn_vec, '.')
'''