import NeuralProcessingTools as npt
import os
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

os.chdir('H:\\WORK\\Camilo\\data\\Pancake\\20130923\\session01')

# Get events using Roger's code
trials = npt.trialstructures.get_trials()
#timestamps, trialidx, stimidx = trials.get_timestamps('reward_on')


def create_trial_table(trials):
    
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
    is_inside_trial = False
    
    stim_ev_re = re.compile('stimulus_on_([0-9])_([0-9]+)')
    
    for n in range(len(trials.events)):
        
        ev = trials.events[n]
        ev_t = trials.timestamps[n]
        
        # Ignore everything not between trial_start and trial_end
        if (not is_inside_trial) and (ev != 'trial_start'):
            print('Event outside trial boundaries (id = %i): %s' % (n,ev))
            continue
    
        # Initiate new trial entry   
        if ev == 'trial_start':        
            trial = trial_templ.copy()
            trial_id = trial_id + 1
            trial.trial_id = trial_id
            trial.correct = True            # Until we meet 'failure' event
            trial.trial_start_t = ev_t
            is_inside_trial = True
            continue
    
        # Append the current trial entry to the table        
        if ev == 'trial_end':
            if trial_id >= 0:
                trial.trial_end_t = ev_t
                trial_tbl = trial_tbl.append(trial)
                is_inside_trial = False
            continue
        
        # Failure
        if ev == 'failure':
            trial.correct = False
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
            
    return trial_tbl
            
# Create trials table from Roger's events
trial_tbl = create_trial_table(trials)

# Select correct trials only
idx = ((trial_tbl.correct.to_numpy() == True) & (trial_tbl.reward_on_t.to_numpy() != None))    
trial_tbl_cor = trial_tbl.iloc[np.where(idx)]

# Remove fields not existing in Tao's data
trial_tbl_cor = trial_tbl_cor.drop(['stimBlankStart_t', 'trial_end_t'], axis=1)
   
# Load trials table from Tao's data
fpath_in = 'H:\\WORK\\Camilo\\Tao_events.csv';
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

# Compare
for col_name in col_names_cmp:
    
    x1 = trial_tbl_cor[col_name].to_numpy()
    x2 = trial_tbl_cor_2[col_name].to_numpy()
    
    d = np.max(np.abs(x1 - x2))
    
    print('%s: %.15f' % (col_name,d))
    
    

            
        
        
    
    
