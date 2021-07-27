import os
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import useful as usf


def plot_rate_vs_TFpow_by_cell(rvec_info, tfROI_info, dirpath_out, ROI_name, ROI_name_bl=None):
    
    if not os.path.exists(dirpath_out):
        os.mkdir(dirpath_out)
        
    plt.figure()
        
    for n in range(len(rvec_info)):
        
        cell = rvec_info.iloc[n]        
        cell_name = cell.cell_name
        
        # Load firing rates
        R = usf.load_xarray(cell.fpath_rvec)
        
        # Find channel
        chan_name = usf.get_chan_by_cell(cell_name, rvec_info, tfROI_info)
        chan = tfROI_info[tfROI_info.chan_name == chan_name].iloc[0]
        
        # Load TFROI
        W = usf.load_xarray(chan.fpath_ROIs)
        
        # ROI of interest
        w = W[W.ROI_name==ROI_name,:].data.ravel()
        w = np.log(w)
        
        # Baseline correction
        if ROI_name_bl is not None:
            w_bl = W[W.ROI_name==ROI_name_bl,:].data.ravel()
            #w_bl = np.log(w_bl)
            #w -= w_bl
            w = (w - w_bl) / w_bl
            
        # Mean firing rate
        r = R.mean(dim='sample_num').data
        #r = np.log(r)
        
        # Plot
        plt.clf()
        plt.plot(r, w, '.')
        plt.xlabel('Firing rate')
        plt.ylabel(f'{ROI_name}')
        plt.title(cell_name)
        
        # Save
        fpath_out = os.path.join(dirpath_out, f'{cell_name}.png')
        plt.savefig(fpath_out)

        
def show_chan_TFROI_pair(chan_tfROI_info, tfROI_pair):
    
    Nchan = len(chan_tfROI_info)
    
    plt.figure()
    
    for n in range(Nchan):
        
        chan = chan_tfROI_info.iloc[n]
        
        # Load TF ROIs
        fpath_in = chan.fpath_ROIs
        WROI = xr.load_dataset(fpath_in, engine='h5netcdf')['__xarray_dataarray_variable__']
        
        ROIs = np.empty(shape=2, dtype=np.object)
        meds = np.empty(shape=2)
        
        for m in range(2):
        
            ROI_id = np.where(WROI.ROI_name == tfROI_pair[m])[0][0]
            ROI = WROI.sel(ROI_num = ROI_id)
            ROI = np.log(ROI)
            ROIs[m] = ROI
            
            meds[m] = np.median(ROI)
        
        plt.clf();
        plt.plot(ROIs[0], ROIs[1], '.')
        plt.plot([np.min(ROIs[0]), np.max(ROIs[0])], [meds[1]]*2, 'r')
        plt.plot([meds[0]]*2, [np.min(ROIs[1]), np.max(ROIs[1])], 'r')
        plt.xlabel(tfROI_pair[0])
        plt.ylabel(tfROI_pair[1])
        plt.title(chan.chan_name)
        
        plt.draw()
        res = plt.waitforbuttonpress()
        if res==False:
            break


