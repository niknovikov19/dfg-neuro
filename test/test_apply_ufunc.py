from pathlib import Path
import sys
from time import time

import numpy as np
from scipy import signal as sig
import xarray as xr

# Creating a sample DataArray with multiple dimensions
dim1 = np.linspace(0, 1, 100)
dim2 = np.linspace(0, 1, 2000)
time_ = np.linspace(0, 1, 1000)
data = np.random.rand(len(dim1), len(dim2), len(time_)) + np.sin(2 * np.pi * 50 * time_)
data[:, :, 0] = (np.arange(len(dim1)).reshape(len(dim1), 1) +
                 np.arange(len(dim2)).reshape(1, len(dim2)) / 1000)

X_in = xr.DataArray(data, dims=['dim1', 'dim2', 'time'],
                    coords={'dim1': dim1, 'dim2': dim2, 'time': time_})
X_in = X_in.chunk({'dim1': None, 'dim2': 100, 'time': -1})

# Define parameters for the spectrogram
fs = 1000  # Sampling frequency in Hz
nperseg = 256  # Length of each segment
noverlap = nperseg // 2  # Overlap between segments

freq_size = nperseg // 2 + 1
time_size = max(1, int(np.ceil((X_in.sizes['time'] - noverlap) / (nperseg - noverlap)))) - 1

def f(X, fs, nperseg, noverlap):
    #print(X.ravel()[0])
    S = sig.spectrogram(X, fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
    #return S
    return S[2]
    
res = xr.apply_ufunc(
    f,  # the function to apply
    X_in,                # input DataArray
    kwargs={'fs': fs, 'nperseg': nperseg, 'noverlap': noverlap},
    input_core_dims=[['time']],  # 'time' is the dimension over which to apply spectrogram
    #output_core_dims=[['freq'], ['time1'], ['freq', 'time1']],  # new dimensions in the output
    output_core_dims=[['freq', 'time1']],  # new dimensions in the output
    output_sizes={'freq': freq_size, 'time1': time_size},
    vectorize=False,  # enable vectorization
    dask='parallelized',  # enable Dask
    #output_dtypes=[float, float, float]
    output_dtypes=[float]
)
res = res.rename({'time1': 'time'})

#res_da = xr.Dataset({'S': res[2], 'ff': res[0], 'tt': res[1]})
res_da = xr.Dataset({'S': res})

# =============================================================================
# S = res[2]
# S.compute()
# 
# ff = res[0][0, 0, :].values
# tt = res[1][0, 0, :].values
# 
# # Reshape output to have the proper dimensions
# S = S.assign_coords(
#     {'freq': ('freq', ff), 'time1': ('time1', tt)})
# #S = S.expand_dims({'dim1': X_in.dim1, 'dim2': X_in.dim2})
# =============================================================================

dirpath_out = Path(r'D:\WORK\Salvador\M1_project\M1_thal_analysis\data\nikita\dask_test')
fpath_out = dirpath_out / 'ufunc_test_1.nc'
encoding = {
    var: {'chunksizes': tuple([chunk[0] for chunk in res_da[var].chunks])} 
    for var in res_da.data_vars
}
t0 = time()
res_da.to_netcdf(fpath_out, engine='h5netcdf', invalid_netcdf=True,
            encoding=encoding)
print(f'{time() - t0}')

Y = xr.open_dataset(fpath_out, engine='h5netcdf', chunks={})
