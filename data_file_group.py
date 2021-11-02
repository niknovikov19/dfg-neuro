# -*- coding: utf-8 -*-

import importlib
import itertools
import os
import sys

import numpy as np
import pandas as pd
import pickle
from pprint import pprint
import xarray as xr

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import useful as usf


class DataProcTree:
    "Description of a sequence of data processing steps."
    
    def __init__(self):
        self.proc_steps = {}
        self.branch_id = '(1)'
        self.proc_steps[self.branch_id] = {}
    
    def _get_last_step_num(self):
        step_nums = [int(s) for s in self.proc_steps[self.branch_id].keys()]
        if len(step_nums) == 0:
            return None
        else:
            return max(step_nums)
        
    def _step_id_by_num(self, step_num):
        return str(step_num)
    
    def _step_num_by_id(self, step_id):
        return int(step_id)
        
    def add_process_step(self, step_name, func_name, step_params,
                         data_desc_out):
        step_num_last = self._get_last_step_num()
        step_num = step_num_last + 1 if step_num_last is not None else 0
        step_desc = {
                'name': step_name,
                'function': func_name,
                'params': step_params,
                'data_desc_out': data_desc_out
        }
        step_id = self._step_id_by_num(step_num)
        self.proc_steps[self.branch_id][step_id] = step_desc
        
    def get_last_step_id(self):
        step_num = self._get_last_step_num()
        if step_num is None:
            return None
        else:
            return self._step_id_by_num(step_num)
        
    def get_last_step(self):
        step_id = self.get_last_step_id()
        if step_id is None:
            raise ValueError('Process tree is empty')
        else:
            return self.proc_steps[self.branch_id][step_id]
        
    def get_parent_steps(self, step_id):
        step_num = self._step_num_by_id(step_id)
        step_id_prev = self._step_id_by_num(step_num - 1)
        step_prev = self.proc_steps[self.branch_id][step_id_prev]
        return [step_prev]


class DataFileGroup:
    "Representation of data stored in multiple files."
    
    def __init__(self):
        self.outer_table = pd.DataFrame()
        self.data_proc_tree = DataProcTree()
        
    def get_data_desc(self):
        return self.data_proc_tree.get_last_step()['data_desc_out']
    
    def get_table_entries(self):
        return self.outer_table.iterrows()
    
    def get_inner_data_path(self, table_entry):
        data_desc = self.get_data_desc()
        fpath_data = table_entry[data_desc['fpath_data_column']]
        return fpath_data
    
    def load_inner_data(self, table_entry):
        fpath_data = self.get_inner_data_path(table_entry)
        X = xr.open_dataset(fpath_data, engine='h5netcdf')
        return X
    
    def save_inner_data(self, table_entry, X, fpath_out):
        data_desc = self.get_data_desc()
        table_entry[data_desc['fpath_data_column']] = fpath_out
        X.to_netcdf(fpath_out, engine='h5netcdf')
        
    def make_data_attrs(self):
        # Data description - from the last processing step
        data_desc = self.get_data_desc()
        attrs = {
            'data_desc': data_desc,
            'proc_steps': self.data_proc_tree.proc_steps,
            }
        return attrs
    
    def make_inner_data_attrs(self, table_entry):
        # Data description - from the last processing step
        data_desc = self.get_data_desc()
        # Names and values of the outer coords for the given table entry
        outer_coord_vals = {
                dim_name: table_entry[dim_name]
                for dim_name in data_desc['outer_dims']}
        # Parent files
        last_step_id = self.data_proc_tree.get_last_step_id()
        parent_steps = self.data_proc_tree.get_parent_steps(last_step_id)
        parent_columns = [step['data_desc_out']['fpath_data_column']
                        for step in parent_steps]
        parent_files = [table_entry[col_name] for col_name in parent_columns]
        attrs = {
            'data_desc': data_desc,
            'proc_steps': self.data_proc_tree.proc_steps,
            'outer_coord_vals': outer_coord_vals,
            'fpath_parent': parent_files
            }
        return attrs
    
    def save(self, fpath_out):
        with open(fpath_out, 'wb') as fid:
            pickle.dump(self, fid)
            
    def load(self, fpath_in):
        with open(fpath_in, 'rb') as fid:
            obj = pickle.load(fid)
        self.data_proc_tree = obj.data_proc_tree
        self.outer_table = obj.outer_table
        
