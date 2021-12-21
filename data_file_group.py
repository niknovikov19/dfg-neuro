# -*- coding: utf-8 -*-

import copy
#import importlib
#import itertools
import os
import sys

import numpy as np
import pandas as pd
import pickle
#from pprint import pprint
from tqdm import tqdm
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
        if step_num == 0:
            return []
        step_id_prev = self._step_id_by_num(step_num - 1)
        step_prev = self.proc_steps[self.branch_id][step_id_prev]
        return [step_prev]


class DataFileGroup:
    "Representation of data stored in multiple files."
    
    def __init__(self, fpath_in=None):
        self.outer_table = pd.DataFrame()
        self.data_proc_tree = DataProcTree()
        if fpath_in is not None:
            self.load(fpath_in)
        
    def create(self, root_proc_step):        
        # Create empty outer table with columns for outer coords
        # and a column for paths to inner data files
        data_desc = root_proc_step['data_desc_out']
        outer_coords = list(data_desc['outer_coords'].keys())
        #TODO: check that outer_dims are a subset of outer_coords
        col_names = outer_coords + [data_desc['fpath_data_column']]
        self.outer_table = pd.DataFrame(columns=col_names)        
        # Create the first processing step description
        self.data_proc_tree.add_process_step(
                root_proc_step['name'], root_proc_step['function'],
                root_proc_step['params'], root_proc_step['data_desc_out'])
        
    def create2(self, proc_steps):
        # Create the processing tree
        # TODO: all branches
        for step in proc_steps['(1)'].values():
            self.data_proc_tree.add_process_step(
                    step['name'], step['function'], step['params'],
                    step['data_desc_out'])
        # Create outer table
        data_desc = self.get_data_desc()
        col_names = list(data_desc['outer_coords'].keys())
        for step in proc_steps['(1)'].values():
            fpath_data_column = step['data_desc_out']['fpath_data_column']
            col_names.append(fpath_data_column)
        self.outer_table = pd.DataFrame(columns=col_names) 
        
    def add_entry(self, outer_coords, X, fpath_data):
        # TODO: check that dimension vals' combinations do not repeat
        # Create new entry in the outer table
        new_entry_data = outer_coords.copy()
        new_entry_data.update({self.get_fpath_data_column_name(): ''})
        self.outer_table = self.outer_table.append(new_entry_data,
                                                   ignore_index=True)
        table_entry = self.get_last_table_entry()
        # Save inner data and store the corresponding path into outer_table
        self.set_inner_data_attrs(table_entry, X)
        self.save_inner_data(table_entry, X, fpath_data)        
        
    def get_data_desc(self):
        return self.data_proc_tree.get_last_step()['data_desc_out']

    def get_num_table_entries(self):
        return len(self.outer_table)
    
    def get_table_entries(self):
        return self.outer_table.index.to_list()
    
    def get_last_table_entry(self):
        return self.get_table_entries()[-1]
    
    def get_fpath_data_column_name(self):
        return self.get_data_desc()['fpath_data_column']
    
    def get_inner_data_path(self, table_entry):
        column_name = self.get_fpath_data_column_name()
        return self.outer_table.at[table_entry, column_name]
    
    def load_inner_data(self, table_entry):
        fpath_data = self.get_inner_data_path(table_entry)
        X = xr.open_dataset(fpath_data, engine='h5netcdf')
        return X
    
    def save_inner_data(self, table_entry, X, fpath_out):
        column_name = self.get_fpath_data_column_name()
        self.outer_table.at[table_entry, column_name] = fpath_out
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
                dim_name: self.outer_table.at[table_entry, dim_name]
                for dim_name in data_desc['outer_dims']}
        # Parent files
        last_step_id = self.data_proc_tree.get_last_step_id()
        parent_steps = self.data_proc_tree.get_parent_steps(last_step_id)
        parent_columns = [step['data_desc_out']['fpath_data_column']
                        for step in parent_steps]
        parent_files = [
                self.outer_table.at[table_entry, col_name]
                for col_name in parent_columns]
        attrs = {
            'data_desc': data_desc,
            'proc_steps': self.data_proc_tree.proc_steps,
            'outer_coord_vals': outer_coord_vals,
            'fpath_parent': parent_files
            }
        return attrs
    
    def set_inner_data_attrs(self, table_entry, X):
        attrs = self.make_inner_data_attrs(table_entry)
        X.attrs = usf.flatten_dict(attrs)
        for var in X.data_vars.values():
            var.attrs.clear()
    
    def save(self, fpath_out):
        with open(fpath_out, 'wb') as fid:
            pickle.dump(self, fid)
            
    def init_outer_indices(self):
        num_entries = len(self.outer_table)
        self.outer_table.set_index(np.arange(num_entries), inplace=True)        
            
    def load(self, fpath_in):
        with open(fpath_in, 'rb') as fid:
            obj = pickle.load(fid)
        self.data_proc_tree = obj.data_proc_tree
        self.outer_table = obj.outer_table
        # Fix un-initialized indices
        if np.all(self.outer_table.index == 0):
            self.init_outer_indices()


def make_data_desc(data_desc_old, var_names, fpath_data_column,
                   inner_dim_names, inner_coord_names,
                   vars_new_descs, coords_new_descs):
    
    data_desc_new = {}
    
    # Name of the column in the outer table with the paths to inner data
    data_desc_new['fpath_data_column'] = fpath_data_column
    
    # Outer dims and coords - from the old description
    data_desc_new['outer_dims'] = data_desc_old['outer_dims']
    data_desc_new['outer_coords'] = data_desc_old['outer_coords']
    
    # Names of inner vars, dims, and coords
    data_desc_new['variables'] = {
            var_name: '' for var_name in var_names}
    data_desc_new['inner_dims'] = inner_dim_names
    data_desc_new['inner_coords'] = {
            coord_name: '' for coord_name in inner_coord_names}
    
    # Var descriptions - try take from the argument or from old data desc
    var_descs_out = data_desc_new['variables']
    var_descs_in = data_desc_old['variables']
    for var_name in var_descs_out:
        if var_name in var_descs_in:
            var_descs_out[var_name] = var_descs_in[var_name]
        if vars_new_descs is not None:
            if var_name in vars_new_descs:
                var_descs_out[var_name] = vars_new_descs[var_name]
            
    # Coord descriptions - try take from the argument or from old data desc
    coord_descs_out = data_desc_new['inner_coords']
    coord_descs_in = data_desc_old['inner_coords']
    for coord_name in coord_descs_out:
        if coord_name in coord_descs_in:
            coord_descs_out[coord_name] = coord_descs_in[coord_name]
        if coords_new_descs is not None:
            if coord_name in coords_new_descs:
                coord_descs_out[coord_name] = coords_new_descs[coord_name]
            
    return data_desc_new


def apply_dfg_inner_proc(dfg_in: DataFileGroup,
                         inner_proc: 'function', params: dict, 
                         proc_step_name: str, gen_proc_step_params: 'function',
                         fpath_data_column: str, gen_fpath: 'function',
                         vars_new_descs=None, coords_new_descs=None):
    
    dfg_out = copy.deepcopy(dfg_in)
    
    # Add to the outer table a column that will contain paths
    # to the output Dataset files
    outer_tbl_out = dfg_out.outer_table
    outer_tbl_out.insert(len(outer_tbl_out.columns), fpath_data_column, '')
    
    # Initialize progress bar
    pbar = tqdm(total=dfg_in.get_num_table_entries())
    
    need_init = True

    for entry in dfg_out.get_table_entries():
    
        # Load dataset
        X_in = dfg_in.load_inner_data(entry)
        
        # Perform inner procedure
        X_out = inner_proc(X_in, **params)
        
        # After the first call of the inner procedure: 
        # 1. Create a desciption of the new data based on:
        #    - description of the old data (from dfg_in)
        #    - properties of the newly generated Dataset (X_out)
        #    - additional arguments (vars_new_descs, coords_new_descs,
        #      and fpath_data_column)
        # 2. Create a description of the new processing step (including
        #    the description of the new data)
        # 3. Add the new step to the data processing tree
        # 4. Put a copy of the data description and the data processing tree
        #    into the attributes of the outer table
        if need_init:            
            data_desc_out = make_data_desc(
                    dfg_in.get_data_desc(), list(X_out.data_vars),
                    fpath_data_column, list(X_out.dims), list(X_out.coords),
                    vars_new_descs, coords_new_descs)
            proc_func_name = 'INNER: ' + inner_proc.__name__
            dfg_out.data_proc_tree.add_process_step(
                    proc_step_name, proc_func_name,
                    gen_proc_step_params(params), data_desc_out)
            dfg_out.outer_table.attrs = usf.flatten_dict(
                    dfg_out.make_data_attrs())
            need_init = False
        
        # Set attributes of the new dataset
        dfg_out.set_inner_data_attrs(entry, X_out)
        
        # Save new dataset and store the path into outer_table
        fpath_in = dfg_in.get_inner_data_path(entry)
        fpath_out = gen_fpath(fpath_in, params)
        dfg_out.save_inner_data(entry, X_out, fpath_out)
    
        pbar.update()
   
    pbar.close()
    
    return dfg_out

