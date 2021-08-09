# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 05:13:38 2021

@author: Nikita
"""


def calc_ROIs(data_in, fpath_col_name_in, fpath_col_name_out,
              ROIset_name, ROIset_dim_name, ROI_coords, ROI_descs,
              dim_to_merge=None, ROIset_name_merged=None,
              ROI_name_merge_templ='{new}_{old}', force_recalc=False):
    """Groups the data into ROIs over given dimensions
    
    ROI_coords = ('coord1', 'coord2', ...)
    
    ROI_descs = [
        {'name': 'ROIname1', 'coord1': (x1first, x1last), 'coord2': (x2first, x2last), ...} 
        ...
        ]

    """
    
    print(f'calc_ROIs(): ROIset = {ROIset_name}')

    # Create output table for external data
    data_out = data_in.copy()
    data_out.insert(len(data_out.columns), fpath_col_name_out, '')
    
    WROI = None    
    NROI = len(ROI_descs)
    
    # Generate ROI names
    ROI_names, ROI_names2 = [], []
    for ROI in ROI_descs:
        ROI_name_2 = 'ROI'
        for coord in ROI_coords:
            ROI_name_2 += f'_({coord}={ROI[coord][0]}-{ROI[coord][1]})'
        ROI_names.append(ROI['name'])
        ROI_names2.append(ROI_name_2)

    # Get input internal dimensions and coordinates
    fpath_in = data_in[fpath_col_name_in][0]
    X = usf.load_xarray(fpath_in, unpack=False)
    dims_in = list(X.dims.keys())
    coords_in = X.coords
    
    # Output internal dimensions
    dims_out = dims_in.copy()
    for dim in ROI_coords:                    # Remove dimesions collapsed into ROIs
        dims_out.remove(dim)
    dims_out.insert(0, ROIset_dim_name)     # Add ROIset dimension (1-st)
        
    
                    
    for n, rec in enumerate(data_in):
        
        print(f'Process record: {n} / {len(data_in)}')

        # Input and output paths for internal data
        fpath_in = rec[fpath_col_name_in]
        postfix = f'ROIset={ROIset_name}'
        fpath_out = usf.generate_fpath_out(fpath_in, postfix)
        
        # Add output path to the record
        data_out[fpath_col_name_out].iloc[n] = fpath_out
        
        # Check whether the internal data is already calculated
        if os.path.exists(fpath_out) and (force_recalc==False):
            print('Already calculated -- skip')
            continue
        
        # Load internal data
        X = usf.load_xarray(fpath_in, unpack=False)
        
        # Allocate output xarray for internal data
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
            tmask = (W.time >= ROIs[m]['tlim'][0]) & (W.time <= ROIs[m]['tlim'][1])
            fmask = (W.freq >= ROIs[m]['flim'][0]) & (W.freq <= ROIs[m]['flim'][1])
            WROI[m,:] = W1.isel(time=tmask, freq=fmask).mean(dim=['time','freq'])
        
        # Add info about the operation
        WROI.attrs = W.attrs.copy()
        W.attrs['ROIset_name'] = ROIset_name
        W.attrs['ROIs'] = ROIs
        W.attrs['TFpow_mode'] = TFpow_mode
        W.attrs['ROI_fpath_source'] = fpath_in
            
        # Save the transformed data
        WROI.to_netcdf(fpath_out, engine='h5netcdf')
        
    # Add info about the epoching operation to the output table
    data_out.attrs['ROIset_name'] = ROIset_name
    data_out.attrs['ROIs'] = ROIs
    data_out.attrs['TFpow_mode'] = TFpow_mode
    
    return data_out
