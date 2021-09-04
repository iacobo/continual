"""
Test suite.
"""

import math
import unittest
import itertools
import torch

from utils import models, data_processing
from main import main

BATCH_SIZES = (1,10,100)
SEQ_LENS = (4,12,48)
N_VARS = (2,10,100)
N_CLASSES = (2,10)
N_LAYERS = (1,2,3,4)
HIDDEN_SIZES = (32,64,128)


DEMOGRAPHICS = ['age', 'gender', 'ethnicity', 'region', 'time_year', 'time_season', 'time_month']
OUTCOMES = ['ARF', 'shock', 'mortality']
DATASETS = ['MIMIC', 'eICU']


class TestDataLoadingMethods(unittest.TestCase):
    """
    Data loading tests.
    """

    def test_modalfeatvalfromseq(self):
        """
        Test that mode of correct dim is returned.
        """
        for n_samples in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                for n_feats in N_VARS:
                    for i in range(n_feats):
                        sim_data = torch.randint(0,1,(n_samples,seq_len,n_feats)).clone().detach().numpy()
                        modes = data_processing.get_modes(sim_data,feat=i)
                        self.assertEqual(modes.shape, torch.Size([n_samples]))


# CL task split tests
class TestCLConstructionMethods(unittest.TestCase):
    """
    Test construction of Continual Learning task splits.
    """

    def ttest_taskidsnonoverlap(self):
        for dataset in DATASETS:
            for experiment in OUTCOMES:
                for demographic in DEMOGRAPHICS:
                    # JA: implement
                    tasks = data_processing.load_data(dataset, demographic, experiment)
                    for pair in itertools.combinations(tasks, repeat=2):
                        self.assertTrue(pair[0][:,0].intersection(pair[0][:,0]) == {})

    def ttest_tasktargets(self):
        for dataset in DATASETS:
            for experiment in OUTCOMES:
                for demographic in DEMOGRAPHICS:
                    # JA: implement
                    tasks = data_processing.load_data(dataset, demographic, experiment)
                    for task in tasks:
                        self.assertTrue(len(task[:,-1].unique())==2)

if __name__ == '__main__':
    unittest.main()
