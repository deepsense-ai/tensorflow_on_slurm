from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest

from tensorflow_on_slurm import tf_config_from_slurm, _expand_ids, _expand_nodelist, _worker_task_id

class BasicTestData(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.nodelist = 'p[1135,1137-1142,1147-1148,1152]'
        self.first_nodename = 'p1135'
        self.nodename = 'p1140'
        self.nodes_number = 10

class ShortNodenameTestData(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.nodelist = 'p[0900-0910]'
        self.first_nodename = 'p0900'
        self.nodename = 'p0902'
        self.nodes_number = 11

class ShortNodenameTestData2(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.setUp(self)
        self.nodelist = 'p[0900,0910]'
        self.first_nodename = 'p0900'
        self.nodename = 'p0910'
        self.nodes_number = 2

class TensorflowSlurmUtilsTest(object):
    def test_expand_ids(self):
        test_ids = '1-5,7,8-12'
        res = _expand_ids(test_ids)
    
    def test_expand_nodelist(self):
        expanded = _expand_nodelist(self.nodelist)
        self.assertEqual(len(expanded), self.nodes_number)
        self.assertIn(self.nodename, expanded)
    
    def test_first_task_id(self):
        expanded = _expand_nodelist(self.nodelist)
        first_task_id = _worker_task_id(expanded, self.first_nodename)
        self.assertEqual(first_task_id, 0)
        
    def test_other_task_id(self):
        expanded = _expand_nodelist(self.nodelist)
        task_id = _worker_task_id(expanded, self.nodename)
        self.assertIn(task_id, range(self.nodes_number))
        
    def test_tf_config_from_slurm(self):
        os.environ["SLURM_JOB_NODELIST"] = self.nodelist
        os.environ["SLURMD_NODENAME"] = self.nodename
        os.environ["SLURM_JOB_NUM_NODES"] = str(self.nodes_number)
        cluster, my_job_name, my_task_index = tf_config_from_slurm(ps_number=2)

class BasicTestCase(BasicTestData, TensorflowSlurmUtilsTest):
    pass
class ShortNodenameTestCase(ShortNodenameTestData, TensorflowSlurmUtilsTest):
    pass
class ShortNodenameTestCase2(ShortNodenameTestData2, TensorflowSlurmUtilsTest):
    pass

if __name__ == '__main__':
    unittest.main()
