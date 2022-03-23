# -*- coding: utf-8 -*-

#import copy
import os
from pprint import pprint
import sys

dirpath_file = os.path.dirname(os.path.abspath(__file__))
dirpath_pkg = os.path.dirname(dirpath_file)
sys.path.append(dirpath_pkg)

import data_file_group_2 as dfg

t1 = dfg.DataProcTree()
t1.add_process_step('step11', '', {'par11': '11'}, {})
t1.add_process_step('step12', '', {'par12': '12'}, {})

t2 = dfg.DataProcTree()
t2.add_process_step('step21', '', {'par21': '21'}, {})
t2.add_process_step('step22', '', {'par22': '22'}, {})

t3 = dfg.DataProcTree()
t3.add_process_step('step31', '', {'par31': '31'}, {})
t3.add_process_step('step32', '', {'par32': '32'}, {})

t_12 = dfg.merge_proc_trees(t1, t2)
t_12.add_process_step('step12_1', '', {'par12_1': '12_1'}, {})

t_12_3 = dfg.merge_proc_trees(t_12, t3)
t_3_12 = dfg.merge_proc_trees(t3, t_12)

tt = dfg.merge_proc_trees(t_12_3, t_3_12)
pprint(tt.proc_steps)



