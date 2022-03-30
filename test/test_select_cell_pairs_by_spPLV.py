# -*- coding: utf-8 -*-

import os
import sys

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg
from select_cell_pairs_by_spPLV import select_cell_pairs_by_spPLV

dirpath_root = r'D:\WORK\Camilo'
dirpath_data = os.path.join(dirpath_root, 'data')
dirpath_proc = os.path.join(dirpath_root, 'Processing_Pancake_2sess_allchan')

Nchan_used = 25
fname_in = f'tbl_spPLV_(ev=stim1_t)_(TF_0.5_0.4_100)_tROI_fROI_pval_(nchan={Nchan_used})'
fpath_in = os.path.join(dirpath_proc, fname_in)

tbl_spPLV_pval = dfg.DataTable(fpath_in)

tROI_name = 'del1'
fROI_name = 'beta'
pthresh = 0.05
rmax = 50

tbl_res = select_cell_pairs_by_spPLV(tbl_spPLV_pval, tROI_name, fROI_name,
                                     pthresh, rmax)