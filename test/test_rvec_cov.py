# -*- coding: utf-8 -*-

import os
import sys

import matplotlib.pyplot as plt

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
import rvec_cov

dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')
fpath_in = os.path.join(
    dirpath_proc, 'dfg_rvec_(ev=stim1_t)_(t=-1.00-3.00)_(t=500-1200_dt=10)')
dfg_in = dfg.DataFileGroup(fpath_in)

nbins_jit = 5
niter_jit = 50
lag_range = (-15, 15)
time_range = (0.85, 1.2)

dfg_out = rvec_cov.calc_dfg_rvec_cov_nojit(
    dfg_in, nbins_jit, niter_jit, time_range, lag_range, test_mode=True)

X = dfg_out.load_inner_data(0)
C = X.rcov[49, 6, :, :].transpose().data

plt.figure()
plt.imshow(C, aspect='auto')


# =============================================================================
# fpath1 = r'D:\WORK\Camilo\data\Pancake\20130923\session01\array01\channel014\cell02\epochs_(ev=stim1_t)_(t=-1.00-3.00)_rvec_(t=500-1200_dt=10).nc'
# fpath2 = r'D:\WORK\Camilo\data\Pancake\20130923\session01\array04\channel118\cell01\epochs_(ev=stim1_t)_(t=-1.00-3.00)_rvec_(t=500-1200_dt=10).nc'
# 
# X1_ = xr.open_dataset(fpath1, engine='h5netcdf').__xarray_dataarray_variable__
# X2_ = xr.open_dataset(fpath2, engine='h5netcdf').__xarray_dataarray_variable__
# 
# tvec = X1_.time.values
# tvec_ROI = tvec[tvec > 0.85]
# 
# X1_ = usf.xarray_select_xr(X1_, {'time': tvec_ROI})
# X2_ = usf.xarray_select_xr(X2_, {'time': tvec_ROI})
# 
# X1 = X1_.values
# X2 = X2_.values
# 
# C = None
# Cj = None
# Cnj = None
# 
# Ntrials = X1.shape[0]
# 
# for n in range(Ntrials):
#     
#     x1 = X1[n,:]
#     x2 = X2[n,:]
#     
#     if ~np.any(x1) and ~np.any(x2):
#         continue
#     
#     c, cj, cnj, lags = _calc_rvec_cov_nojit(
#         x1, x2, nbins_jit=5, niter_jit=50, lag_range=(-15, 15))
#     
#     if C is None:
#         Nlags = len(lags)
#         C = np.nan * np.ones((Ntrials, Nlags))
#         Cj = np.nan * np.ones((Ntrials, Nlags))
#         Cnj = np.nan * np.ones((Ntrials, Nlags))
#         
#     C[n,:] = c
#     Cj[n,:] = cj
#     Cnj[n,:] = cnj
# 
# CCnj = np.mean(Cnj, axis=0)
#     
# plt.figure()
# plt.imshow(Cnj, aspect='auto')
# #plt.plot(lags, CCnj)
# =============================================================================