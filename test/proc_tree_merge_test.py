# -*- coding: utf-8 -*-

import os
import re
import sys

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import numpy as np

import data_file_group_2 as dfg


# Test usage of regexp
s = '(1+2)+(1)+(2+3)+(21)'
templ = '([\d]+)[\D]+'
res = re.findall(templ, s)

def find_max_root_num(branch_id):
    templ = '([\d]+)[\D]+'
    res = re.findall(templ, branch_id)
    root_nums = [int(s) for s in res]
    return np.max(np.array(root_nums))

N = find_max_root_num(s)

def increment_root_nums(branch_id, inc):
    def rep_fun(match):
        s = match.group(0)
        x = int(s)
        return str(x + inc)
    templ = '[\d]+'
    return re.sub(templ, rep_fun, branch_id)

res = increment_root_nums(s, 10)