# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 18:47:40 2024

@author: Nikita
"""

import os
import pickle as pk
import time

import matplotlib.pyplot as plt
import numpy as np
#import h5py
import scipy
import scipy.signal as sig
#import sys
#import pandas as pd
import xarray as xr

import data_file_group_2 as dfg
#import firing_rate as fr
import roi_utils as roi
#import spiketrain_manager as spk
import useful as usf


def _calc_dfg_ersp_inner(X_in, baseline, var_name='TF', time_win=None):
    
    # Select the variable to process and
    X_in = X_in[var_name]
    
    # Complex amplitude -> power
    X_pow = np.real(X_in * np.conjugate(X_in))
    
    # Baseline
    X_bl = X_pow.sel(time=slice(*baseline)).mean(dim=['time', 'trial'])
    
    # ERSP
    X_ersp = (X_pow - X_bl) / X_bl
    
    # Select time interval of interest
    if time_win is not None:
        X_ersp = X_ersp.sel(time=slice(*time_win))
    
    return xr.Dataset({'ERSP': X_ersp})


def calc_dfg_ersp(dfg_in, baseline, var_name='TF', time_win=None, 
                  need_recalc=True):
    
    proc_step_name = 'ERSP'
    params = {'baseline': {'val': baseline, 'short': 'bl',
                           'desc': 'Baseline time window, s'},
              'var_name': {'val': var_name, 'short': None,
                           'desc': 'Variable to process'},
              'time_win': {'val': time_win, 'short': 't',
                           'desc': 'Time window to keep'}}
    dfg_out = dfg.apply_dfg_inner_proc_mt_2(
            dfg_in, _calc_dfg_ersp_inner, proc_step_name, params, need_recalc
            )    
    return dfg_out

