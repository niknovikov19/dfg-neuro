# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 23:44:52 2021

@author: Nikita
"""

import trial_manager as trm
import os
import pandas as pd
import numpy as np


dirpath_ev_Tao = 'H:\\WORK\\Camilo\\Tao_events'
dirpath_ev_Roger = 'H:\\WORK\\Camilo\\data\\Pancake'

subj_prefix = 'P'
sess_id = '20131014'


# Create trial table from Roger's events
dirpath_in = os.path.join(dirpath_ev_Roger, sess_id, 'session01')
trial_tbl = trm.create_sess_trial_table(dirpath_in)

# Select correct trials only
idx = ((trial_tbl.correct.to_numpy() == True) & (trial_tbl.reward_on_t.to_numpy() != None))    
trial_tbl_cor = trial_tbl.iloc[np.where(idx)]

# Remove fields not existing in Tao's data
trial_tbl_cor = trial_tbl_cor.drop(['stimBlankStart_t', 'trial_end_t'], axis=1)


# Load trial table from Tao's data
fname_in = 'Tao_events_%s%s.csv' % (subj_prefix, sess_id)
fpath_in = os.path.join(dirpath_ev_Tao, fname_in)
trial_tbl_cor_2 = pd.read_csv(fpath_in, delimiter=',', header=0)

   
# Columns to compare        
col_names_cmp = [
        'trial_start_t',
        'fix_start_t',
        'stim1_t',
        'stim1_code',
        'stim2_t',
        'stim2_code',
        'delay_start_t',
        'response_on_t',
        'reward_on_t']

if trial_tbl_cor.shape[0] != trial_tbl_cor_2.shape[0]:
    
    print('Unequal number of trials!\n')
    
else:

    # Compare
    for col_name in col_names_cmp:
        
        x1 = trial_tbl_cor[col_name].to_numpy()
        x2 = trial_tbl_cor_2[col_name].to_numpy()
        
        dvec = np.abs(x1 - x2)
        d = np.max(dvec)
        d_id = np.argmax(dvec)
        
        print('%s:  dmax = %.15f  (id = %i)' % (col_name, d, d_id))
    
    
    
    