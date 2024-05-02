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

from . import data_file_group_2 as dfg
#import firing_rate as fr
#import roi_utils as roi
#import spiketrain_manager as spk
#import useful as usf


def _calc_dfg_ersp_inner(X_in, baseline):
    
    X = {}
    X['total'] = X_in['TFpow']
    X['pl'] = X_in['TFpow_pl']
    X['npl'] = X_in['TFpow'] - X_in['TFpow_pl']
    
    # Baseline
    for key, X_ in X.items():
        if 'trial' in X_.dims:
            dims_avg = ['time', 'trial']
        else:
            dims_avg = ['time']
        X_bl = X_.sel(time=slice(*baseline)).mean(dim=dims_avg)
        X[key] = (X_ - X_bl) / X_bl
    
    res = {'ERSP': X['total'], 'ERSP_pl': X['pl'], 'ERSP_npl': X['npl']}    
    return xr.Dataset(res)


def calc_dfg_ersp(dfg_in, baseline, need_recalc=True):
    
    proc_step_name = 'ERSP'
    params = {'baseline': {'val': baseline, 'short': 'bl',
                           'desc': 'Baseline time window, s'}}
    dfg_out = dfg.apply_dfg_inner_proc_2(
            dfg_in, _calc_dfg_ersp_inner, proc_step_name, params, need_recalc
            )    
    return dfg_out

