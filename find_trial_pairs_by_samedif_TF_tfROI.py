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
import xarray as xr
import pickle as pk

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import useful as usf
import trial_manager as trl
import spiketrain_manager as spk
import firing_rate as fr
import lfp
#import spike_corr as spcor
import vis

import data_file_group_2 as dfg
import roi_utils as roi
import spike_TF_PLV as spPLV
import useful as usf


def _dfg_find_trial_pairs_by_samedif_tfpow_inner(X_in, ROIset_same, ROIset_dif,
                                       Nbins=20, thresh_sameROI=1.5,
                                       thresh_difROI=1.5):
    """ Find trial pairs with same / different TF power in two ROIs.
    
    Find trial pairs (for a given channel), such that in each pair
    tf power values in ROIset_same ROI are close to each other and 
    tf power values in ROIset_dif ROI maximally differ from each other
    
    """
    
    # Get ROI's for selecting same / different TF power values
    x_same = usf.xarray_select(X_in.TFpow, ROIset_same)
    x_dif = usf.xarray_select(X_in.TFpow, ROIset_dif)
    #print(type(X_in))
    #print(type(x_same))
    if (x_same.size==0) or (x_dif.size==0):
        raise ValueError('ROI not found')
    #x_same = x_same.data[0]
    #x_dif = x_dif.data[0]
    
    # Sort trials by 'same' ROI values
    idx_same_sorted = x_same.argsort()
    
    trial_idx_loval = np.array([], dtype=np.int64)
    trial_idx_hival = np.array([], dtype=np.int64)
    N = x_same.size
    
    # Bins by sorted 'same' ROI values
    for n in range(Nbins):
        
        # Trials from the current bin
        idx2_bin = np.arange(np.ceil(n*N/Nbins), (n+1)*N/Nbins, dtype=np.int64)
        idx_bin = idx_same_sorted[idx2_bin]
        
        # Sort bin trials by 'dif' ROI values
        x_dif_bin = x_dif[idx_bin]
        idx2_dif_srt_bin = x_dif_bin.argsort()
        idx_dif_srt_bin = idx_bin[idx2_dif_srt_bin]
        
        # Get trial pairs from the bin, sorted by the difference in 'dif' ROI values
        npairs = int(len(idx_bin) / 2)
        idx_bin_loval = idx_dif_srt_bin[0:npairs]
        idx_bin_hival = idx_dif_srt_bin[-npairs:]
        trial_idx_loval = np.append(trial_idx_loval, idx_bin_loval)
        trial_idx_hival = np.append(trial_idx_hival, idx_bin_hival)
    
    # Check that the groups contain different trials
    #a1 = set(trial_idx_loval)
    #a2 = set(trial_idx_hival)
    #a1.intersection(a2)

    # ROI value differences for the selected trial pairs
    diff_sameROI = x_same[trial_idx_hival] - x_same[trial_idx_loval]
    diff_difROI = x_dif[trial_idx_hival] - x_dif[trial_idx_loval]
    
    # Find outliers
    mask_same = (np.abs((diff_sameROI - np.mean(diff_sameROI)))
                 > (thresh_sameROI * np.std(diff_sameROI)))
    mask_dif = (np.abs((diff_difROI - np.mean(diff_difROI)))
                > (thresh_difROI * np.std(diff_difROI)))
    
    '''
    plt.figure()
    plt.plot(diff_sameROI, diff_difROI, '.')
    plt.plot(diff_sameROI[mask_same], diff_difROI[mask_same], '.')
    plt.plot(diff_sameROI[mask_dif], diff_difROI[mask_dif], '.')
    plt.xlabel(str(ROIset_same))
    plt.ylabel(str(ROIset_dif))
    plt.title('TF power difference in trial pairs')
    '''
    
    # Remove outliers
    mask = ~mask_same & ~mask_dif
    trial_idx_loval = trial_idx_loval[mask]
    trial_idx_hival = trial_idx_hival[mask]
    diff_sameROI = diff_sameROI[mask]
    diff_difROI = diff_difROI[mask]
    
    # TODO: p-values
    
    # Initialize output DataArray objects
    Npairs = len(trial_idx_loval)
    coords_out = {'trial_pair': np.arange(Npairs)}
    trial_idx_loval_xr = xr.DataArray(
        trial_idx_loval, coords=coords_out, dims=['trial_pair'])
    trial_idx_hival_xr = xr.DataArray(
        trial_idx_hival, coords=coords_out, dims=['trial_pair'])
    diff_sameROI_xr = xr.DataArray(
        diff_sameROI, coords=coords_out, dims=['trial_pair'])
    diff_difROI_xr = xr.DataArray(
        diff_difROI, coords=coords_out, dims=['trial_pair'])

    # Collect the output dataset
    data_vars = {'trial_id_loval': trial_idx_loval_xr,
                 'trial_id_hival': trial_idx_hival_xr,
                 'diff_sameROI': diff_sameROI_xr,
                 'diff_difROI': diff_difROI_xr}
                 
    X_out = xr.Dataset(data_vars)
    return X_out


def dfg_find_trial_pairs_by_samedif_tfpow(dfg_in, ROIset_same, ROIset_dif,
                                       Nbins=20, thresh_sameROI=1.5,
                                       thresh_difROI=1.5):
    """ Find trial pairs with same / different TF power in two ROIs.
    
    Find trial pairs (for all channels), such that in each pair
    tf power values in ROIset_same ROI are close to each other and 
    tf power values in ROIset_dif ROI maximally differ from each other
    
    """
    
    # Name of the processing step
    proc_step_name = 'Find trial pairs with similar / different TFpow in two ROIs'
    
    # Dictionary of parameters
    param_names = ['ROIset_same', 'ROIset_dif', 'Nbins',
                   'thresh_sameROI', 'thresh_difROI']
    local_vars = locals()
    params = {par_name: local_vars[par_name] for par_name in param_names}
    
    # Name of the dfg's outer table column for the paths to Dataset files
    fpath_data_column = 'fpath_TFpow_trial_pairs'
        
    # Function that converts the parameters' dict to the form suitable
    # for storing into a processing step description
    def gen_proc_step_params(par):
        par_out = {
            'ROIset_same': {
                'desc': 'ROI in which TFpow should be similar in a trial pair',
                'value': str(ROIset_same)},
            'ROIset_dif': {
                'desc': 'ROI in which TFpow should differ in a trial pair',
                'value': str(ROIset_dif)},
            'Nbins': {
                'desc': 'Number of bins to separate trial values of ROIset_same ROI',
                'value': str(Nbins)},            
            'thresh_sameROI': {
                'desc': 'Outlier threshold for ROIset_same values',
                'value': str(thresh_sameROI)},
            'thresh_difROI': {
                'desc': 'Outlier threshold for ROIset_dif values',
                'value': str(thresh_difROI)}
        }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath(fpath_in, params):
        fpath_data_postfix = 'trial_pairs'
        fpath_noext, ext  = os.path.splitext(fpath_in)
        return fpath_noext + '_' + fpath_data_postfix + ext
    
    # Description of the new variables
    vars_new_descs = {
            'trial_id_loval': 'Trial with a small value in ROIset_dif',
            'trial_id_hival': 'Trial with a large value in ROIset_dif',
            'diff_sameROI': 'Value difference between trials in ROIset_same',
            'diff_difROI': 'Value difference between trials in ROIset_dif'
    }
    
    # Description of the new coordinates
    coords_new_descs = {
            'trial_pair': 'Trial pair'
    }
    
    # Call the processing function for each inner dataset of the DataFileGroup
    dfg_out = dfg.apply_dfg_inner_proc(
            dfg_in, _dfg_find_trial_pairs_by_samedif_tfpow_inner, params,
            proc_step_name, gen_proc_step_params, fpath_data_column, gen_fpath,
            vars_new_descs, coords_new_descs)
    
    return dfg_out

