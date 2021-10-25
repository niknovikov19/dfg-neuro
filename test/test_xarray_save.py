# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr

dims = ['dim0', 'dim1']
sz = (3, 2)
coords = {
        'dim0':     np.arange(sz[0]),    # 0, 1, 2
        'dim1':     np.arange(sz[1]),    # 0, 1
        }

x = np.random.randint(0, 100, size=sz)
dim_coords = [coord_val for coord_name, coord_val in coords.items()
              if coord_name in dims]

X = xr.DataArray(x, coords=coords, dims=dims)
Q = xr.Dataset({'X': X})

Q.attrs[(1, 'a', 3)] = np.arange(10)
Q.attrs[(2, 'b')] = 'gggggg'

Q.to_netcdf('data/xr_save_test_1.nc', engine='h5netcdf')

