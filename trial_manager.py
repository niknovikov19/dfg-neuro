# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 05:12:31 2021

@author: Nikita
"""

import NeuralProcessingTools as npt
import os
import pandas as pd
import re
import numpy as np
#import matplotlib.pyplot as plt


# Create trial table for a session
def create_sess_trial_table(dirpath_session):
    
    # Get events using Roger's code
    dirpath_old = os.getcwd()
    os.chdir(dirpath_session)
    trials = npt.trialstructures.get_trials()
    os.chdir(dirpath_old)
    
    col_names = [
        'trial_id',
        'correct',
        'trial_start_t',
        'fix_start_t',
        'stim1_t',
        'stim1_code',
        'stimBlankStart_t',
        'stim2_t',
        'stim2_code',
        'delay_start_t',
        'response_on_t',
        'reward_on_t',
        'trial_end_t']
    
    # Table of trails for output
    trial_tbl = pd.DataFrame(columns=col_names)
    
    # Table entry
    trial_templ = pd.DataFrame(data=[[None]*len(col_names)], columns=col_names)
    trial = trial_templ
    
    trial_id = -1
    #is_inside_trial = False
    trial_start_flag = False
    fix_start_flag = False
    
    stim_ev_re = re.compile('stimulus_on_([0-9])_([0-9]+)')
    
    for n in range(len(trials.events)):
        
        ev = trials.events[n]
        ev_t = trials.timestamps[n]
        
        # Ignore everything not between fix_start and trial_end
        if (not trial_start_flag) and (not fix_start_flag) and (ev not in ['trial_start', 'fix_start']):
            print('Event outside trial boundaries (id = %i): %s' % (n,ev))
            continue
    
        if (ev == 'trial_start') or ((ev == 'fix_start') and not trial_start_flag):
            
            # Append the previous trial entry to the table
            if trial_id >= 0:
                if trial.reward_on_t[0] is None:
                    trial.correct = False
                trial_tbl = trial_tbl.append(trial)
            
            # Initiate new trial entry
            trial = trial_templ.copy()
            trial_id = trial_id + 1
            #if trial_id==32:
            #    xxx=1
            trial.trial_id = trial_id
            trial.correct = True            # Until we meet 'failure' event
            #trial.correct = False
            if ev == 'trial_start':
                trial.trial_start_t = ev_t
                trial_start_flag = True
            if ev == 'fix_start':
                trial.trial_start_t = ev_t
                trial.fix_start_t = ev_t
                fix_start_flag = True
            #is_inside_trial = True
            
            continue
    
                
        if ev == 'trial_end':
            trial.trial_end_t = ev_t
            #is_inside_trial = False
            trial_start_flag = False
            fix_start_flag = False
            continue
        
        # Failure
        if ev == 'failure':
            trial.correct = False
            #trial.correct = True
            continue
        
        # Parse stimulus event
        re_match = stim_ev_re.match(ev)
        if re_match is not None:
            stim_num = int(re_match.groups()[0])
            stim_code = int(re_match.groups()[1])
            if stim_num == 1:
                trial.stim1_code = stim_code
                trial.stim1_t = ev_t
            if stim_num == 2:
                trial.stim2_code = stim_code
                trial.stim2_t = ev_t
            continue
                
        # Process other events
        col_name = ev + '_t'
        if col_name in col_names:
            trial[col_name] = ev_t
        else:
            print('Unknown event (id = %i): %s' % (n,ev))
            
    # Append the last trial entry to the table
    if trial_id >= 0:
        if trial.reward_on_t[0] is None:
            trial.correct = False
        trial_tbl = trial_tbl.append(trial)
            
    return trial_tbl

# Select correct trials only
def select_correct_trials(trial_tbl):
    idx = ((trial_tbl.correct.to_numpy() == True) & (trial_tbl.reward_on_t.to_numpy() != None))    
    trial_tbl_cor = trial_tbl.iloc[np.where(idx)]
    return trial_tbl_cor

# Create trial tables for all subjects and sessions
def create_trial_info(dirpath_root):
    
    # Initialize the output
    trial_info = []
    
    # Walk through all subjects and sessions
    for dirpath, dirnames, filenames in os.walk(dirpath_root):
        for dirname in dirnames:
            
            # Try to interpret the current folder is a session folder
            sess_id_local = re.match('session([0-9]+)', dirname)
            
            if sess_id_local is not None:

                # Path to the session folder
                dirpath_sess = os.path.join(dirpath, dirname)

                # Subject name and session date    
                dirpath_base, date_str = os.path.split(dirpath)
                dirpath_base, subj_name = os.path.split(dirpath_base)

                # Session id: date + number                
                sess_id = date_str + '_' + str(int(sess_id_local.groups()[0]))
                
                # Create trial table for the current subject + sesssion
                print('Subject: %s  Session: %s\n' % (subj_name, sess_id))
                trial_tbl = create_sess_trial_table(dirpath_sess)
                
                # Append the output
                entry = {'subj_name': subj_name, 'sess_id': sess_id, 'dirpath_sess': dirpath_sess, 'trial_tbl': trial_tbl}
                trial_info.append(entry)
                
    return trial_info





