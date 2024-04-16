# -*- coding: utf-8 -*-

import concurrent.futures
import copy
#import importlib
#import itertools
import os
from pprint import pprint
import re
import sys

import numpy as np
import pandas as pd
import pickle
#from pprint import pprint
from tqdm import tqdm
import xarray as xr

# =============================================================================
# dirpath_file = os.path.dirname(os.path.abspath(__file__))
# dirpath_pkg = os.path.dirname(dirpath_file)
# sys.path.append(dirpath_pkg)
# =============================================================================

from . import useful as usf


INNER_PROC_MULTITHREAD = 0


import io
import pickle

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "data_file_group_2":
            renamed_module = "dfg.data_file_group_2"
        return super(RenameUnpickler, self).find_class(renamed_module, name)

def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)

# =============================================================================
# 
# =============================================================================


class DataProcTree:
    "Description of a sequence of data processing steps."
    
    def __init__(self):
        self.clear()
        
    def clear(self):
        self.proc_steps = {}
        self.branch_id = '(1)'
        self.proc_steps[self.branch_id] = {}
        
    def __str__(self):
        return str(self.proc_steps)
    def __repr__(self):
        return repr(self.proc_steps)
    
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


def find_max_root_num(branch_id):
    templ = '([\d]+)[\D]+'
    res = re.findall(templ, branch_id)
    root_nums = [int(s) for s in res]
    return np.max(np.array(root_nums))

def increment_root_nums(branch_id, inc):
    def rep_fun(match):
        s = match.group(0)
        x = int(s)
        return str(x + inc)
    templ = '[\d]+'
    return re.sub(templ, rep_fun, branch_id)


def merge_proc_trees(tree1: DataProcTree, tree2: DataProcTree):
    branch_id_1 = tree1.branch_id  
    branch_id_2 = tree2.branch_id
    inc = find_max_root_num(branch_id_1)
    branch_id_2 = increment_root_nums(branch_id_2, inc)
    branch_id_12 = f'({branch_id_1}+{branch_id_2})'    
    tree12 = DataProcTree()
    tree12.proc_steps = copy.deepcopy(tree1.proc_steps)
    for branch_id, proc_step in tree2.proc_steps.items():
        branch_id_new = increment_root_nums(branch_id, inc)
        tree12.proc_steps[branch_id_new] = copy.deepcopy(proc_step)
    tree12.branch_id = branch_id_12
    tree12.proc_steps[branch_id_12] = {}
    return tree12

# =============================================================================
# 
# =============================================================================
    
class DataContainerBase:
    "Base class for the multi-file and table storages."
    
    def __init__(self, fpath_in=None):
        self.outer_table = pd.DataFrame()
        self.data_proc_tree = DataProcTree()
        if fpath_in is not None:
            self.load(fpath_in)

    def _create_outer_table(self, used_proc_steps='all'):
        data_desc = self.get_data_desc()
        col_names = list(data_desc['outer_coords'].keys())
        proc_steps = self.data_proc_tree.proc_steps
        if used_proc_steps == 'all':
            steps = proc_steps['(1)'].values()
        elif used_proc_steps == 'last':
            steps = [self.data_proc_tree.get_last_step()]
        for step in steps:
            step_data_desc = step['data_desc_out']
            if 'fpath_data_column' in step_data_desc:
                col_names += [step_data_desc['fpath_data_column']]
            if 'outer_data_vals' in step_data_desc:
                col_names += list(step_data_desc['outer_data_vals'].keys())
        self.outer_table = pd.DataFrame(columns=col_names)
        
    def _set_proc_tree(self, proc_steps):
        self.data_proc_tree.clear()
        for step in proc_steps['(1)'].values():
            self.data_proc_tree.add_process_step(
                    step['name'], step['function'], step['params'],
                    step['data_desc_out'])
        
    def create2(self, proc_steps, used_proc_steps='all'):
        """
        proc_steps[n]
            name
            function
            params
            data_desc_out
                variables
                    {name: description}
                outer_dims
                    [name]
                outer_coords
                    {name: description}
                ------
                <DataFileGroup only>
                fpath_data_column
                inner_dims
                    [name]
                inner_coords
                    {name: description}
        """
        # Create the processing tree
        # TODO: all branches
        self._set_proc_tree(proc_steps)
        # Create outer table
        self._create_outer_table(used_proc_steps)
        
    def create(self, root_proc_step):
        proc_steps = {'(1)': {'0': root_proc_step}}        
        self.create2(proc_steps)
        
    def subset(self, index):
        dfg_sub = type(self)()
        dfg_sub.data_proc_tree = copy.deepcopy(self.data_proc_tree)
        dfg_sub.outer_table = self.outer_table[index].copy()
        dfg_sub._init_outer_indices()
        return dfg_sub
        
    def get_data_desc(self):
        return self.data_proc_tree.get_last_step()['data_desc_out']

    def get_num_table_entries(self):
        return len(self.outer_table)
    
    def get_table_entries(self):
        return self.outer_table.index.to_list()
    
    def get_last_table_entry(self):
        return self.get_table_entries()[-1]
    
    def get_table_entry_rec(self, table_entry):
        return dict(self.outer_table.loc[table_entry])
    
    def make_data_attrs(self):
        # Data description - from the last processing step
        data_desc = self.get_data_desc()
        attrs = {
            'data_desc': data_desc,
            'proc_steps': self.data_proc_tree.proc_steps,
            }
        return attrs
    
    def get_var_names(self):
        return list(self.get_data_desc()['variables'].keys())
    
    def _init_outer_indices(self):
        num_entries = len(self.outer_table)
        self.outer_table.set_index(np.arange(num_entries), inplace=True)
        
    def save(self, fpath_out):
        with open(fpath_out, 'wb') as fid:
            pickle.dump(self, fid)
            
    def load(self, fpath_in):
        with open(fpath_in, 'rb') as fid:
            #print(fpath_in)
            #obj = pickle.load(fid)
            obj = renamed_load(fid)
        self.data_proc_tree = obj.data_proc_tree
        self.outer_table = obj.outer_table
        # Fix un-initialized indices
        if np.all(self.outer_table.index == 0):
            self._init_outer_indices()
    
    def change_root(self, old_root, new_root):
        fpath_col_name = self.get_fpath_data_column_name()
        for entry in self.get_table_entries():        
            fpath_old = self.outer_table.at[entry, fpath_col_name]
            fpath_new = fpath_old.replace(old_root, new_root)
            self.outer_table.at[entry, fpath_col_name] = fpath_new
            
    def get_table_entries_by_coords(self, outer_coord_vals):
        mask = np.ones((len(self.outer_table)), dtype=bool)        
        for coord_name, coord_val in outer_coord_vals.items():
            mask_cur = (self.outer_table[coord_name] == coord_val)
            mask &= mask_cur
        row_num = np.where(mask)[0]
        return self.outer_table.index[row_num]
    
    def get_table_entry_by_coords(self, outer_coord_vals):
        return self.get_table_entries_by_coords(outer_coord_vals)[0]
    
    def print_proc_tree(self):
        pprint(self.data_proc_tree.proc_steps)
        

# =============================================================================
# 
# =============================================================================


class DataTable(DataContainerBase):
    "Representation of data stored in a table."
    
    def __init__(self, fpath_in=None):
        super().__init__(fpath_in)

    def add_entry(self, outer_coords, X):
        """Create new entry in the outer table.
        
        outer_coords = {name1: value1, ...}
        X = {name1: value1, ...}
        """
        # TODO: check that dimension vals' combinations do not repeat
        new_entry = outer_coords.copy()
        new_entry.update(X)
        self.outer_table = self.outer_table.append(
                new_entry, ignore_index=True)
    
    def get_outer_data(self, table_entry):
        var_names = self.get_var_names()
        return dict(self.outer_table.loc[table_entry][var_names])

   
# =============================================================================
#             
# =============================================================================

class DataFileGroup(DataContainerBase):
    "Representation of data stored in multiple files."
    
    def __init__(self, fpath_in=None, chunks=None):
        super().__init__(fpath_in)
        if chunks is None:
            self.chunks = {}
        else:
            self.chunks = chunks
        
    def add_entry(self, outer_coords, X, fpath_data, save_inner=True):
        # TODO: check that dimension vals' combinations do not repeat
        # Create new entry in the outer table
        new_entry_data = outer_coords.copy()
        new_entry_data.update({self.get_fpath_data_column_name(): ''})
        #self.outer_table = self.outer_table.append(new_entry_data,
        #                                           ignore_index=True)
        new_entry_data_pd = pd.DataFrame([new_entry_data])
        self.outer_table = pd.concat(
            (self.outer_table, new_entry_data_pd), ignore_index=True)
        table_entry = self.get_last_table_entry()
        # Save inner data and store the corresponding path into outer_table
        self.set_inner_data_attrs(table_entry, X)
        self.save_inner_data(table_entry, X, fpath_data, save_inner)
    
    def get_fpath_data_column_name(self):
        return self.get_data_desc()['fpath_data_column']
    
    def get_inner_data_path(self, table_entry):
        column_name = self.get_fpath_data_column_name()
        return self.outer_table.at[table_entry, column_name]
    
    def load_inner_data_by_path(self, fpath_data, h5=True):
                                #chunks=None):
                                #chunks='auto'):
                                #chunks={'trial': -1, 'chan': 'auto', 'freq': -1, 'time': -1}):
        #X = xr.open_dataset(fpath_data, engine='h5netcdf')
        if h5:
            engine = 'h5netcdf'
        else:
            engine = None
        X = xr.open_dataset(fpath_data, engine=engine, chunks=self.chunks)
        return X
    
    def load_inner_data(self, table_entry, h5=True):
        fpath_data = self.get_inner_data_path(table_entry)
        X = self.load_inner_data_by_path(fpath_data, h5)
        return X
    
    def save_inner_data(self, table_entry, X, fpath_out, save_inner=True):
        column_name = self.get_fpath_data_column_name()
        self.outer_table.at[table_entry, column_name] = fpath_out
        if save_inner:
            if list(X.values())[0].chunks is None:
                encoding = None  # not a dask array
            else:
                encoding = {
                    var: {'chunksizes': tuple([chunk[0] for chunk in X[var].chunks])}
                    for var in X.data_vars
                }
            X.to_netcdf(fpath_out, engine='h5netcdf', invalid_netcdf=True,
                        encoding=encoding)
    
    def make_inner_data_attrs(self, table_entry):
        # Data description - from the last processing step
        data_desc = self.get_data_desc()
        # Names and values of the outer coords for the given table entry
        outer_coord_vals = {
                dim_name: self.outer_table.at[table_entry, dim_name]
                for dim_name in usf.list_wrap(data_desc['outer_dims'])}
        # Parent files
        last_step_id = self.data_proc_tree.get_last_step_id()
        parent_steps = self.data_proc_tree.get_parent_steps(last_step_id)
        parent_columns = [step['data_desc_out']['fpath_data_column']
                        for step in parent_steps]
        try:
            parent_files = [
                    self.outer_table.at[table_entry, col_name]
                    for col_name in parent_columns]
        except:
            parent_files = []
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


# =============================================================================
#             
# =============================================================================

def make_data_desc(data_desc_old, var_names, fpath_data_column,
                   inner_dim_names, inner_coord_names,
                   vars_new_descs, coords_new_descs):
    
    data_desc_new = {}
    
    # Name of the column in the outer table with the paths to inner data
    data_desc_new['fpath_data_column'] = fpath_data_column
    
    # Outer dims and coords - from the old description
    data_desc_new['outer_dims'] = usf.list_wrap(data_desc_old['outer_dims'])
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


# =============================================================================
# def _apply_dfg_inner_proc_st(dfg_in: DataFileGroup,
#                          inner_proc, params: dict, 
#                          proc_step_name: str, gen_proc_step_params,
#                          fpath_data_column: str, gen_fpath,
#                          vars_new_descs=None, coords_new_descs=None,
#                          need_recalc=True):
#     
#     # TODO: implement need_recalc
#     
#     dfg_out = copy.deepcopy(dfg_in)
#     
#     # Add to the outer table a column that will contain paths
#     # to the output Dataset files
#     outer_tbl_out = dfg_out.outer_table
#     outer_tbl_out.insert(len(outer_tbl_out.columns), fpath_data_column, '')
#     
#     # Initialize progress bar
#     pbar = tqdm(total=dfg_in.get_num_table_entries())
#     
#     need_init = True
# 
#     for entry in dfg_out.get_table_entries():
#     
#         # Load dataset
#         X_in = dfg_in.load_inner_data(entry)
#         
#         # Perform inner procedure
#         X_out = inner_proc(X_in, **params)
#         
#         # After the first call of the inner procedure: 
#         # 1. Create a desciption of the new data based on:
#         #    - description of the old data (from dfg_in)
#         #    - properties of the newly generated Dataset (X_out)
#         #    - additional arguments (vars_new_descs, coords_new_descs,
#         #      and fpath_data_column)
#         # 2. Create a description of the new processing step (including
#         #    the description of the new data)
#         # 3. Add the new step to the data processing tree
#         # 4. Put a copy of the data description and the data processing tree
#         #    into the attributes of the outer table
#         if need_init:            
#             data_desc_out = make_data_desc(
#                     dfg_in.get_data_desc(), list(X_out.data_vars),
#                     fpath_data_column, list(X_out.dims), list(X_out.coords),
#                     vars_new_descs, coords_new_descs)
#             proc_func_name = 'INNER: ' + inner_proc.__name__
#             dfg_out.data_proc_tree.add_process_step(
#                     proc_step_name, proc_func_name,
#                     gen_proc_step_params(params), data_desc_out)
#             dfg_out.outer_table.attrs = usf.flatten_dict(
#                     dfg_out.make_data_attrs())
#             need_init = False
#         
#         # Set attributes of the new dataset
#         dfg_out.set_inner_data_attrs(entry, X_out)
#         
#         # Save new dataset and store the path into outer_table
#         fpath_in = dfg_in.get_inner_data_path(entry)
#         fpath_out = gen_fpath(fpath_in, params)
#         dfg_out.save_inner_data(entry, X_out, fpath_out)
#     
#         pbar.update()
#    
#     pbar.close()
#     
#     return dfg_out
# =============================================================================


def apply_dfg_inner_proc(dfg_in: DataFileGroup,
                         inner_proc, params: dict, 
                         proc_step_name: str, gen_proc_step_params,
                         fpath_data_column: str, gen_fpath,
                         vars_new_descs=None, coords_new_descs=None,
                         need_recalc=True):
    
    dfg_out = copy.deepcopy(dfg_in)
    
    # Add to the outer table a column that will contain paths
    # to the output Dataset files
    outer_tbl_out = dfg_out.outer_table
    outer_tbl_out.insert(len(outer_tbl_out.columns), fpath_data_column, '')
    
    # Initialize progress bar
    #pbar = tqdm(total=dfg_in.get_num_table_entries())
    
    # Process the first entry to get info required for initializing the output
    X_in_0 = None
    ind0 = dfg_in.outer_table.index[0]
    fpath_in = dfg_in.get_inner_data_path(ind0)
    fpath_out = gen_fpath(fpath_in, params)
    if os.path.exists(fpath_out) and not need_recalc:
        X_out_0 = dfg_out.load_inner_data_by_path(fpath_out)
    else:
        X_in_0 = dfg_in.load_inner_data(ind0)
        X_out_0 = inner_proc(X_in_0, **params)
    
    # Initialize the output:
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
    data_desc_out = make_data_desc(
            dfg_in.get_data_desc(), list(X_out_0.data_vars),
            fpath_data_column, list(X_out_0.dims), list(X_out_0.coords),
            vars_new_descs, coords_new_descs)
    proc_func_name = 'INNER: ' + inner_proc.__name__
    dfg_out.data_proc_tree.add_process_step(
            proc_step_name, proc_func_name,
            gen_proc_step_params(params), data_desc_out)
    dfg_out.outer_table.attrs = usf.flatten_dict(
            dfg_out.make_data_attrs())

    def thread_proc(entry):
        
        #print(f'Entry: {entry}')
        
        # Save new dataset and store the path into outer_table
        fpath_in = dfg_in.get_inner_data_path(entry)
        fpath_out = gen_fpath(fpath_in, params)
        
        # If inner data file already exists - add table entry without touching
        # the inner data. If it does not exist - calculate, save, add entry
        if os.path.exists(fpath_out) and not need_recalc:
            dfg_out.save_inner_data(entry, None, fpath_out, save_inner=False)
        else:
            #X_in = None
            if entry == 0:
                # Precalculated first result
                X_out = X_out_0
                X_in = X_in_0
            else:
                # Load input dataset
                X_in = dfg_in.load_inner_data(entry)        
                # Perform inner procedure
                X_out = inner_proc(X_in, **params)           
            # Set attributes of the new dataset, save it, and add table entry
            dfg_out.set_inner_data_attrs(entry, X_out)
            dfg_out.save_inner_data(entry, X_out, fpath_out)
            # Close input file handler
            if X_in is not None:
                X_in.close()
                X_in = None
    
    # Run processing of each table entry in a separate thread
    if INNER_PROC_MULTITHREAD:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for entry in dfg_out.get_table_entries():
                executor.submit(thread_proc, entry)
    else:
        pbar = tqdm(total=dfg_in.get_num_table_entries())
        for entry in dfg_out.get_table_entries():
            thread_proc(entry)
            pbar.update()
        pbar.close()
    
    return dfg_out


def gen_pathstr_val(x):
    if isinstance(x, dict):
        raise ValueError('Cannot add dict to filename')
    elif isinstance(x, (list, tuple, np.ndarray)):
        return f'{x[0]}_{x[-1]}'
    else:
        return str(x)

def apply_dfg_inner_proc_2(dfg_in: DataFileGroup, inner_proc, 
                       proc_step_desc: tuple, params: dict, need_recalc=True,
                       vars_new_descs=None, coords_new_descs=None,
                       fpath_data_column=None, fpath_prefix=None,
                       gen_fpath_proc=None):
    
    # Name of the processing step
    if isinstance(proc_step_desc, str):
        proc_step_desc = (proc_step_desc, proc_step_desc)
    proc_step_name_short = proc_step_desc[0]
    proc_step_name_full = proc_step_desc[1]
    
    # Name of the dfg's outer table column for the paths to Dataset files
    if fpath_data_column is None:
        fpath_data_column = 'fpath_' + proc_step_name_short
    
    # Dictionary of parameters
    params2 = {name: par['val'] for name, par in params.items()}
    
    # Function for converting params into the formst of apply_dfg_inner_proc()
    def gen_proc_step_params(params2_):
        par_out = {}
        for name, par in params.items():
            par_out[name] = {
                'desc': par['desc'],
                'value': params2_[name]
                }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath_(fpath_in, params2_):
        fpath_noext, ext = os.path.splitext(fpath_in)
        fpath = fpath_noext + '_' + proc_step_name_short
        params_used = [par for par in params.values() if par['short'] is not None]
        if len(params_used):
            fpath += '_('
            for name, par in params_used.items():
                fpath += (par['short'] + '=' + gen_pathstr_val(params2_[name]) + '_')
            fpath = fpath[:-1] + ')'        
        return fpath + ext
    if gen_fpath_proc is None:
        gen_fpath_proc = gen_fpath_
        
    dfg_out = apply_dfg_inner_proc(
            dfg_in, inner_proc, params2, proc_step_name_full,
            gen_proc_step_params, fpath_data_column, gen_fpath_proc,
            vars_new_descs, coords_new_descs, need_recalc
            )
    return dfg_out


def dfg_to_table(dfg_in: DataFileGroup):
    
    data_desc_in = dfg_in.get_data_desc()
    
    # New data description: inner dims / coords become outer dims / coords
    data_desc_out = copy.deepcopy(data_desc_in)
    data_desc_out['outer_dims'] = (
        usf.list_wrap(data_desc_out['outer_dims']) +
        usf.list_wrap(data_desc_out['inner_dims']))
    data_desc_out['outer_coords'].update(data_desc_out['inner_coords'])
    data_desc_out.pop('inner_dims')
    data_desc_out.pop('inner_coords')
    data_desc_out.pop('fpath_data_column')
    
    fpath_col_name = dfg_in.get_fpath_data_column_name()
    
    # Columns of the new table
    cols_out = list(dfg_in.outer_table.columns)
    cols_out.pop(cols_out.index(fpath_col_name))
    cols_out += list(data_desc_in['inner_coords'].keys())
    cols_out += list(data_desc_in['variables'].keys())
    
    entry_list_out = []
    for entry_in in dfg_in.get_table_entries():
        # Row of the old table
        entry_rec_in = dfg_in.get_table_entry_rec(entry_in)
        # Row of the new table (start construction)
        entry_rec_out_0 = entry_rec_in.copy()
        entry_rec_out_0.pop(fpath_col_name)
        # Load inner data
        X = dfg_in.load_inner_data(entry_in)
        coord_man = usf.XrCoordManager(X)
        # Walk through all positions in the inner data and put values from
        # each position into a separate row of the new table
        pos_list = list(coord_man.get_all_positions())
        for pos in tqdm(pos_list):
            # Current inner dims / coords
            pos_coords = coord_man.coords_by_pos(pos)
            pos_dims = coord_man.dims_by_pos(pos)
            # Add current inner coords to the new table row
            entry_rec_out = entry_rec_out_0.copy()
            entry_rec_out.update(pos_coords)
            # Get values of all inner data variables at the current position
            # and add them to the new table row
            for var_name in list(X.data_vars):
                x = X.loc[pos_dims][var_name].values.item()
                entry_rec_out[var_name] = x
            # Add an entry to the list of the the new table rows
            entry_list_out.append(entry_rec_out)
    
    # Description of the processing step
    step_name = 'DataFileGroup to DataTable'
    step_fun = 'dfg_to_table()'
    step_params = {}
    
    # Create output object
    dtbl_out = DataTable()
    dtbl_out.outer_table = pd.DataFrame(entry_list_out, columns=cols_out)
    dtbl_out.data_proc_tree = copy.deepcopy(dfg_in.data_proc_tree)
    dtbl_out.data_proc_tree.add_process_step(
            step_name, step_fun, step_params, data_desc_out)
    dtbl_out._init_outer_indices()
    return dtbl_out
            
            

#cm = usf.XrCoordManager(X)

'''
variables
    {name: description}
outer_dims
    [name]
outer_coords
    {name: description}
------
<DataFileGroup only>
fpath_data_column
inner_dims
    [name]
inner_coords
    {name: description}
'''


def dfg_inner_mean(X, dim_name):
    X.mean(dim=dim_name)
    

def dfg_collapse_dim(dfg_in, dim_name, coord_names, proc_info, postfix=None,
                     dataset_proc=None):
    """ Collapse along a given dimension.
    
    proc_info = [{var_name_old, proc, var_name_new, var_desc_new}]
    
    """
    
    print('dfg_collapse_dim')
    
    # Name of the processing step
    proc_step_name = 'Collapse along a dimension'
    
    # Dictionary of parameters
    param_names = ['dim_name', 'proc_info', 'dataset_proc']
    local_vars = locals()
    params = {par_name: local_vars[par_name] for par_name in param_names}
    
    # Name of the dfg's outer table column for the paths to Dataset files
    data_desc = dfg_in.get_data_desc()
    fpath_data_column_in = data_desc['fpath_data_column']
    if postfix is None:
        fpath_data_column = f'{fpath_data_column_in}_(collapse={dim_name})'
    else:
        fpath_data_column = f'{fpath_data_column_in}_{postfix}'

    # Function that converts the parameters dict to the form suitable
    # for storing into a processing step description
    def gen_proc_step_params(par):
        proc_info_out = copy.deepcopy(par['proc_info'])
        for var_proc_info in proc_info_out:
            if var_proc_info['proc'] is not None:
                var_proc_info['proc'] = var_proc_info['proc'].__name__
        dataset_proc_out = par['dataset_proc']
        if dataset_proc_out is not None:
            dataset_proc_out = dataset_proc_out.__name__
        par_out = {
            'proc_info': {
                'desc': 'Procedures to apply to variables',
                'value': str(proc_info_out)},
            'dataset_proc': {
                'desc': 'Procedure to apply to the whole dataset',
                'value': str(dataset_proc_out)},
            'dim_name': {
                'desc': 'Dimension to collapse along',
                'value': par['dim_name']}
        }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath(fpath_in, params):
        dim_name = params['dim_name']
        if postfix is None:
            fpath_data_postfix = f'(collapse={dim_name})'
        else:
            fpath_data_postfix = postfix
        fpath_noext, ext  = os.path.splitext(fpath_in)
        return fpath_noext + '_' + fpath_data_postfix + ext
    
    # Description of the new variables
    data_desc = dfg_in.get_data_desc()
    vars_new_descs = {}
    for v in proc_info:
        if ((v['var_name_new'] == v['var_name_old']) and
            (v['var_desc_new'] is None)):
                v['var_desc_new'] = data_desc['variables'][v['var_name_old']]
        vars_new_descs[v['var_name_new'] ] = v['var_desc_new']
    
    # Description of the new coordinates
    coords_new_descs = copy.deepcopy(data_desc['inner_coords'])
    for coord_name in coord_names:
        coords_new_descs.pop(coord_name)
    
    def _inner_proc(X, dim_name, proc_info, dataset_proc):
        if dataset_proc is not None:
            return dataset_proc(X, dim_name)
        else:
            X_out = {}
            for v in proc_info:
                X_var_in = X[v['var_name_old']]
                if v['proc'] is not None:
                    var_value = v['proc'](X_var_in, dim_name)
                else:
                    var_value = copy.deepcopy(X_var_in)
                X_out[v['var_name_new']] = var_value
            return xr.Dataset(X_out)
    
    # Call calc_dataset_ROIs() for each inner dataset of the DataFileGroup
    dfg_out = apply_dfg_inner_proc_mt(
            dfg_in, _inner_proc, params, proc_step_name,
            gen_proc_step_params, fpath_data_column, gen_fpath,
            vars_new_descs, coords_new_descs)
    
    return dfg_out


def dfg_elementwise_proc(dfg_in, proc_info, postfix, dataset_proc=None):
    """ Apply an element-wise function.
    
    proc_info = [{var_name_old, proc, var_name_new, var_desc_new}]
    
    """
    
    print('dfg_apply_elementwise_proc')
    
    # Name of the processing step
    proc_step_name = 'Apply an element-wise function'
    
    # Dictionary of parameters
    param_names = ['proc_info', 'dataset_proc']
    local_vars = locals()
    params = {par_name: local_vars[par_name] for par_name in param_names}
    
    # Name of the dfg's outer table column for the paths to Dataset files
    data_desc = dfg_in.get_data_desc()
    fpath_data_column_in = data_desc['fpath_data_column']
    fpath_data_column = f'{fpath_data_column_in}_{postfix}'

    # Function that converts the parameters dict to the form suitable
    # for storing into a processing step description
    def gen_proc_step_params(par):
        proc_info_out = copy.deepcopy(par['proc_info'])
        for var_proc_info in proc_info_out:
            if var_proc_info['proc'] is not None:
                var_proc_info['proc'] = var_proc_info['proc'].__name__
        dataset_proc_out = dataset_proc
        if dataset_proc_out is not None:
            dataset_proc_out = dataset_proc_out.__name__
        par_out = {
            'proc_info': {
                'desc': 'Procedures to apply to variables',
                'value': str(proc_info_out)},
            'dataset_proc': {
                'desc': 'Procedure to apply to the whole dataset',
                'value': str(dataset_proc_out)},
        }
        return par_out
    
    # Function for converting input to output inner data path
    def gen_fpath(fpath_in, params):
        fpath_data_postfix = postfix
        fpath_noext, ext  = os.path.splitext(fpath_in)
        return fpath_noext + '_' + fpath_data_postfix + ext
    
    # Description of the new variables
    data_desc = dfg_in.get_data_desc()
    vars_new_descs = {}
    for v in proc_info:
        if ((v['var_name_new'] == v['var_name_old']) and
            (v['var_desc_new'] is None)):
                v['var_desc_new'] = data_desc['variables'][v['var_name_old']]
        vars_new_descs[v['var_name_new'] ] = v['var_desc_new']
    
    # Description of the new coordinates
    coords_new_descs = copy.deepcopy(data_desc['inner_coords'])
    
    def _inner_proc(X, proc_info, dataset_proc):
        if dataset_proc is not None:
            return dataset_proc(X)
        else:
            X_out = {}
            for v in proc_info:
                X_var_in = X[v['var_name_old']]
                if v['proc'] is not None:
                    var_value = v['proc'](X_var_in)
                else:
                    var_value = copy.deepcopy(X_var_in)
                X_out[v['var_name_new']] = var_value
            return xr.Dataset(X_out)
    
    # Call calc_dataset_ROIs() for each inner dataset of the DataFileGroup
    dfg_out = apply_dfg_inner_proc_mt(
            dfg_in, _inner_proc, params, proc_step_name,
            gen_proc_step_params, fpath_data_column, gen_fpath,
            vars_new_descs, coords_new_descs)
    
    return dfg_out