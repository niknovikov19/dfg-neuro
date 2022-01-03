
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

import useful as usf


def find_chan_trial_pairs_tfROI(tfROI_info, chan_name, ROIname_same, ROIname_dif, Nbins=20, thresh_sameROI=1.5, thresh_difROI=1.5):
# Find trial pairs (for a given channel), such that in each pair
# tf power values in 'ROIname_same' ROI are close to each other and 
# tf power values in 'ROIname_dif' ROI maximally differ from each other
    
    # Get channel
    chan_rec = tfROI_info[tfROI_info['chan_name']==chan_name]
    if chan_rec.empty:
        print('find_chan_trial_pairs_tfROI(): cannot find channel in tfROI_info')
        return None
    
    # Load tfROI data
    fpath_tfROI = chan_rec.fpath_ROIs.values[0]
    X = usf.load_xarray(fpath_tfROI)
    if X is None:
        print('find_chan_trial_pairs_tfROI(): cannot load tfROI data')
        return None
    
    x_same = X.where(X.ROI_name==ROIname_same, drop=True)
    x_dif = X.where(X.ROI_name==ROIname_dif, drop=True)
    if (x_same.size==0) or (x_dif.size==0):
        print('find_chan_trial_pairs_tfROI(): invalid ROI name')
        return None
    x_same = x_same.data[0]
    x_dif = x_dif.data[0]
    
    # Sort trials by 'same' ROI values
    idx_same_srt = x_same.argsort()
    
    N = x_same.size
    
    trial_idx_loval = np.array([], dtype=np.int64)
    trial_idx_hival = np.array([], dtype=np.int64)
    
    # Bins by sorted 'same' ROI values
    for n in range(Nbins):
        
        # Trials from the current bin
        idx2_bin = np.arange(np.ceil(n*N/Nbins), (n+1)*N/Nbins, dtype=np.int64)
        idx_bin = idx_same_srt[idx2_bin]
        
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
    mask_same = np.abs((diff_sameROI - np.mean(diff_sameROI))) > (thresh_sameROI * np.std(diff_sameROI))
    mask_dif = np.abs((diff_difROI - np.mean(diff_difROI))) > (thresh_difROI * np.std(diff_difROI))
    
    '''
    plt.figure()
    plt.plot(diff_sameROI, diff_difROI, '.')
    plt.plot(diff_sameROI[mask_same], diff_difROI[mask_same], '.')
    plt.plot(diff_sameROI[mask_dif], diff_difROI[mask_dif], '.')
    plt.xlabel(ROIname_same)
    plt.ylabel(ROIname_dif)
    plt.title('TF power difference in trial pairs')
    '''
    
    # Remove outliers
    mask = ~mask_same & ~mask_dif
    trial_idx_loval = trial_idx_loval[mask]
    trial_idx_hival = trial_idx_hival[mask]
    diff_sameROI = diff_sameROI[mask]
    diff_difROI = diff_difROI[mask]
    
    res = {'trial_idx_loval': trial_idx_loval, 'trial_idx_hival': trial_idx_hival, 'diff_sameROI': diff_sameROI, 'diff_difROI': diff_difROI}
    return res
    
#res = find_chan_trial_pairs_tfROI(tfROI_info, 'Pancake_20130923_1_ch8', 'beta_del12', 'beta_del11')


def _find_trial_pairs_by_samedif_tfpow(X_in, ROIset_same, ROIset_dif,
                                       Nbins=20, thresh_sameROI=1.5,
                                       thresh_difROI=1.5):
    """ Find trial pairs with same / different TF power in two ROIs.
    
    Find trial pairs (for a given channel), such that in each pair
    tf power values in 'ROIname_same' ROI are close to each other and 
    tf power values in 'ROIname_dif' ROI maximally differ from each other
    
    """
    
    # Get ROI's for selecting same / different TF power values
    x_same = usf.xarray_select(X_in, ROIset_same)
    x_dif = usf.xarray_select(X_in, ROIset_dif)
    if (x_same.size==0) or (x_dif.size==0):
        raise ValueError('ROI with the given name not found')
    x_same = x_same.data[0]
    x_dif = x_dif.data[0]
    
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
    mask_same = np.abs((diff_sameROI - np.mean(diff_sameROI))) > (thresh_sameROI * np.std(diff_sameROI))
    mask_dif = np.abs((diff_difROI - np.mean(diff_difROI))) > (thresh_difROI * np.std(diff_difROI))
    
    '''
    plt.figure()
    plt.plot(diff_sameROI, diff_difROI, '.')
    plt.plot(diff_sameROI[mask_same], diff_difROI[mask_same], '.')
    plt.plot(diff_sameROI[mask_dif], diff_difROI[mask_dif], '.')
    plt.xlabel(ROIname_same)
    plt.ylabel(ROIname_dif)
    plt.title('TF power difference in trial pairs')
    '''
    
    # Remove outliers
    mask = ~mask_same & ~mask_dif
    trial_idx_loval = trial_idx_loval[mask]
    trial_idx_hival = trial_idx_hival[mask]
    diff_sameROI = diff_sameROI[mask]
    diff_difROI = diff_difROI[mask]
    
    res = {'trial_idx_loval': trial_idx_loval, 'trial_idx_hival': trial_idx_hival, 'diff_sameROI': diff_sameROI, 'diff_difROI': diff_difROI}
    return res

    
    
# batch function that calls find_trial_pairs_by_chan_tfROI() for each channel?
    
def select_cell_pairs_by_spPLV(spPLVf_info, tROI_name, flim, chan_npairs_max, pthresh):
# 
    print('select_cell_pairs_by_spPLV')

    # Create output table
    #chan_info_out = chan_epoched_info.copy()
    #chan_info_out.insert(len(chan_info_out.columns), 'fpath_tf', '')
    
    Nchan = len(spPLVf_info)
    
    for n in range(Nchan):
        
        print('%i / %i' % (n, Nchan))
        
        chan = spPLVf_info.iloc[n]
        
        # Input and output paths
        fpath_in = chan.fpath_epoched
        postfix = 'TF_(wlen=%.03f_wover=%.03f_fmax=%.01f)' % (win_len, win_overlap, fmax)
        fpath_out = usf.generate_fpath_out(fpath_in, postfix)
        
        # Check whether it is already calculated
        if os.path.exists(fpath_out) and (need_recalc==False):
            print('Already calculated')
            chan_info_out.fpath_tf.iloc[n] = fpath_out
            continue
        
        # Load spPLV data
        X = xr.load_dataset(fpath_in)['__xarray_dataarray_variable__']
    
