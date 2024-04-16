# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 16:44:55 2024

@author: aleks
"""

from time import time

import dask
from dask.distributed import Client
import xarray as xr


client = Client(threads_per_worker=1, n_workers=1)
print(client.dashboard_link)  # Prints the link to the Dask dashboard

fpath_in = r"D:\WORK\Salvador\M1_project\M1_thal_analysis\data\nikita\dfg_test\230508_2774_1426VAL\lfp_epoched_TF_(wlen=0.25_wover=0.225_fmax=100).nc"

t0 = time()
X = xr.open_dataset(fpath_in, chunks='auto')
dt = time() - t0; print(f'dt = {dt}')

t0 = time()
Y = X.TF.mean(dim='trial')
Y = Y.compute().values
dt = time() - t0; print(f'dt = {dt}')

#Y.data.visualize()
fpath = r"D:\WORK\Salvador\M1_project\M1_thal_analysis\data\nikita\dfg_test\230508_2774_1426VAL\lfp_epoched_TF_(wlen=0.25_wover=0.225_fmax=100)_TFpow_().nc"
X = xr.open_dataset(fpath, chunks=None)

#fpath = r'E:\\M1_exp\\Proc\\230508_2774_1426VAL\\lfp_epoched_TF_(wlen=0.25_wover=0.225_fmax=100).nc'
fpath = r"E:\M1_exp\Proc\230508_2774_1426VAL\lfp_epoched.nc"
X = xr.open_dataset(fpath, engine='h5netcdf', chunks={'chan': 42, 'time': -1})
#X = xr.open_dataset(fpath)
#X = xr.open_dataset(fpath, engine='h5netcdf', chunks={})

Y = X.LFP + 2j
#Y = Y.compute()
Y = xr.Dataset({'LFP': Y})
fpath_out = r"D:\WORK\Salvador\M1_project\M1_thal_analysis\data\nikita\dfg_test\test6.nc"
enc = {var: {'chunksizes': tuple([chunk[0] for chunk in Y[var].chunks])}
       for var in Y.data_vars}
Y.to_netcdf(fpath_out, engine='h5netcdf', invalid_netcdf=True, encoding=enc)
Y1 = xr.open_dataset(fpath_out, chunks={})
Y1.close()

import numpy as np
Z_ = np.zeros((1000, 1000), dtype=np.complex128)
#Z = 
