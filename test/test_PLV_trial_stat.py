# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:50:54 2021

@author: Nikita
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import xarray as xr
import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import useful as usf
import trial_manager as trl
import spiketrain_manager as spk
import firing_rate as fr
import lfp
import spike_corr as spcor
import vis

import data_file_group as dfg
import roi_utils as roi
import spike_TF_PLV as spPLV
import useful as usf


# Root paths for the data and the processing results
dirpath_root = r'H:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

Nchan_used = 25
fname_in = f'dfg_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_tROI_fROI_pval_(nchan={Nchan_used})'
fpath_in = os.path.join(dirpath_proc, fname_in)    
dfg_spPLV_tROI_fROI_pval = dfg.DataFileGroup(fpath_in)

pvals_all = np.array([], dtype=np.float)
PLV_all = np.array([], dtype=np.float)
R_all = np.array([], dtype=np.float)
chan_all = np.array([], dtype=np.float)

for n in range(25):

    X = dfg_spPLV_tROI_fROI_pval.load_inner_data(n)
    
    PLV_ROI = X.PLV.data[0,0,:]
    pval_ROI = X.PLV_pval.data[0,0,:]
    R = X.firing_rate.data[0,:]
    
    PLV_all = np.concatenate((PLV_all, PLV_ROI))
    pvals_all = np.concatenate((pvals_all, pval_ROI))
    R_all = np.concatenate((R_all, R))
    chan_all = np.concatenate((chan_all, [n]*len(R)))
    
    alpha = 0.05 / len(R)
    mask = (pval_ROI < alpha)
    PLV_ok = PLV_ROI[mask]
    
    sz = R / 2
    sz = 12
    
    plt.figure(100)
    plt.clf()
    #plt.scatter(np.real(PLV_ROI), np.imag(PLV_ROI), s=sz)    
    plt.scatter(np.real(PLV_ok), np.imag(PLV_ok), s=sz)    
    plt.plot(0, 0, 'k+', markersize=15)
    plt.draw()
    plt.waitforbuttonpress()
    #break

idx = np.argsort(pvals_all)
PLV_all = PLV_all[idx]
pvals_all = pvals_all[idx]
R_all = R_all[idx]
chan_all = chan_all[idx]

pvals_all_cor = pvals_all * len(pvals_all)

mask = (pvals_all < 0.001) & (R_all < 100)
sz = R_all / 4

plt.figure(100)
plt.clf()
plt.scatter(np.real(PLV_all[mask]), np.imag(PLV_all[mask]), s=sz[mask])    
plt.plot(0, 0, 'k+', markersize=15)
plt.draw()

