# -*- coding: utf-8 -*-


#import numpy as np
#from pprint import pprint


DATA_ATTRIB_DELIMITER = '.'


def flatten_dict(dictionary, level = []):
    tmp_dict = {}
    for key, val in dictionary.items():
        if type(val) == dict:
            tmp_dict.update(flatten_dict(val, level + [key]))
        else:
            tmp_dict[DATA_ATTRIB_DELIMITER.join(level + [key])] = val
    return tmp_dict


def unflatten_dict(dictionary):
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(DATA_ATTRIB_DELIMITER)
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict


class DataAttributes:
    
    def __init__(self):
        self.attr = {'data_info': {}, 'proc_steps': {}}
    
    def _get_last_step_num(self):
        step_nums = [int(s) for s in self.attr['proc_steps'].keys()]
        if len(step_nums) == 0:
            return None
        else:
            return max(step_nums)

    def from_xarray_attrs(self, attr):
        if len(attr) == 0:
            self.attr = {'data_info': {}, 'proc_steps': {}}
        else:
            self.attr = unflatten_dict(attr)
        
    def to_xarray_attrs(self):
        return flatten_dict(self.attr)
        
    def add_process_step(self, step_name, func_name, step_params,
                         data_info_out):
        step_num = self._get_last_step_num()
        if step_num is None:
            data_info_in = None
            step_num = 0
        else:
            step_last = self.attr['proc_steps'][str(step_num)]
            data_info_in = step_last['data_info_out']
            step_num += 1
        step_new = {
                'name': step_name,
                'function': func_name,
                'params': step_params,
                'data_info_in': data_info_in,
                'data_info_out': data_info_out
        }
        self.attr['proc_steps'][str(step_num)] = step_new
        self.attr['data_info'] = step_new['data_info_out']
        

