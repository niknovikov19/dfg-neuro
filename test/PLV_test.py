# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
import pickle as pk
from tqdm import tqdm
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
import TF
import useful as usf

# Root paths for the data and the processing results
dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

# Load Time-Frequency data
fname_dfg_tf = 'dfg_TF_(ev=stim1_t)_(t=-1.00-3.00)_(wlen=0.500_wover=0.450_fmax=100.0)'
fpath_dfg_tf = os.path.join(dirpath_proc, fname_dfg_tf)
dfg_tf = dfg.DataFileGroup(fpath_dfg_tf)

# Load epoched spiketrains
fname_cell_epoched_info = 'cell_epoched_info_(ev=stim1_t)_(t=-1.00-3.00)'
fpath_cell_epoched_info = os.path.join(dirpath_proc, fname_cell_epoched_info)
with open(fpath_cell_epoched_info, 'rb') as fid:
    cell_epoched_info = pk.load(fid)
    
chan_num = 3
cell_num = 41

# Load channel TF data
X_in = dfg_tf.load_inner_data(chan_num)

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
#ncell_max = 10
#cells = cells[:ncell_max]

# Load cell spikes
cell = cells.iloc[cell_num]
with open(cell.fpath_epoched, 'rb') as fid:
    spikes = pk.load(fid)
    spikes = spikes.values

# Time interval in which PLV is calculated
t_range = (0.5, 1.2)

non_phase_locked = True
if non_phase_locked:
    X_TF = X_in.TF - X_in.TF.mean(dim='trial_num')
else:
    X_TF = X_in.TF         

tvec = X_TF.time.values
time_mask = ((tvec >= t_range[0]) & (tvec < t_range[1]))
tvec = tvec[time_mask]
dt = tvec[1] - tvec[0]

fvec = X_TF.freq.values    
fvec = fvec.reshape((len(fvec), 1))

Ntrials = len(X_in.trial_num)
Nfreq = len(fvec)

M0 = np.zeros((Nfreq, Ntrials))
M = np.zeros((Nfreq, Ntrials))

u0 = np.zeros((Nfreq), dtype=np.complex128)
u = np.zeros((Nfreq), dtype=np.complex128)

pbar = tqdm(total=Ntrials)

for trial_num in range(Ntrials):
            
    # Select spikes within the given time interval
    trial_spikes = spikes[trial_num]            
    spike_mask = ((trial_spikes >= t_range[0]) & (trial_spikes < t_range[1]))
    spike_times = trial_spikes[spike_mask]
    Nspikes = len(spike_times)
    if Nspikes == 0:
        #print('No spikes')
        continue
    
    # Spectrogram for a given trial
    X = X_TF[:, :, trial_num].values
    X = X[:, time_mask]
    
    spike_times_rel = spike_times - tvec[0]
    spike_bins = np.floor(spike_times_rel / dt).astype(int)
    spike_dt_vec = spike_times_rel - spike_bins * dt

    spike_dt_vec = spike_dt_vec.reshape((1, len(spike_dt_vec)))
    
    Y0 = np.angle(X[:, spike_bins])
    Y = Y0 + 2 * np.pi * fvec * spike_dt_vec
    
    Y0 = np.mod(Y0, 2 * np.pi)
    Y = np.mod(Y, 2 * np.pi)
    
    S0 = X[:, spike_bins]
    S0 = S0 / np.abs(S0)
    S = S0 * np.exp(2 * np.pi * fvec * spike_dt_vec * 1j)
    
    s0 = np.abs(np.mean(S0, axis=1))
    s = np.abs(np.mean(S, axis=1))
    
    M0[:, trial_num] = s0
    M[:, trial_num] = s
    
    u0 += np.sum(S0, axis=1)
    u += np.sum(S, axis=1)
    
    pbar.update()
    
pbar.close()

# =============================================================================
# plt.figure()
# plt.subplot(2, 1, 1)
# plt.imshow(M0, aspect='auto', origin='lower')
# plt.colorbar()
# plt.ylabel('Frequency')
# plt.subplot(2, 1, 2)
# plt.imshow(M, aspect='auto', origin='lower')
# plt.colorbar()
# plt.xlabel('Trial')
# plt.ylabel('Frequency')
# =============================================================================

u0 /= Ntrials
u /= Ntrials

plt.figure()
plt.plot(fvec, np.abs(u0))
plt.plot(fvec, np.abs(u))

# =============================================================================
# plt.figure()
# plt.subplot(2, 2, 1)
# plt.imshow(Y0, aspect='auto')
# plt.colorbar()
# plt.subplot(2, 2, 2)
# plt.imshow(Y, aspect='auto')
# plt.colorbar()
# plt.subplot(2, 2, 3)
# plt.plot(fvec, s0)
# plt.xlabel('Frequency')
# plt.subplot(2, 2, 4)
# plt.plot(fvec, s)
# plt.xlabel('Frequency')
# =============================================================================


# =============================================================================
# # Spike-triggerred LFP phase
# PLV = X.isel(trial_num=trial_num).interp(
#     time=ROI_spikes, method='nearest', assume_sorted=False)
# PLV /= np.abs(PLV)
# PLV = PLV.mean(dim='time').data
# 
# # Store into output arrays
# data_PLV[:, tROI_num, trial_num, cell_num] = PLV
# data_Nspikes[tROI_num, trial_num, cell_num] = Nspikes
# =============================================================================



