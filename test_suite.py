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
SEQ_LENS = (4,16,32,48,50,100)
N_VARS = (2,10,30,100)
N_CLASSES = (2,3,4,10)

DEMOGRAPHICS = ['age', 'gender', 'ethnicity', 'region', 'time_year', 'time_season', 'time_month']
OUTCOMES = ['ARF', 'shock', 'mortality']
DATASETS = ['MIMIC', 'eICU']

def magnitude(value):
    """
    Return the magnitude of a positive number.
    """
    if value < 0:
        raise ValueError
    if value == 0:
        return 0
    else:
        return int(math.floor(math.log10(value)))

class TestModelMethods(unittest.TestCase):
    """
    Model definition tests.
    """

    def test_modeloutputshape(self):
        """
        Testing model produces correct shape of output for variety of input sizes.
        """
        for batch_size in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                for n_vars in N_VARS:
                    for n_classes in N_CLASSES:
                        batch = torch.randn(batch_size, seq_len, n_vars)
                        simple_models = models.MODELS.values()
                        for model in simple_models:
                            model = model(seq_len=seq_len, n_channels=n_vars, output_size=n_classes)
                            output = model(batch)
                            expected_shape = torch.Size([batch_size, n_classes])
                            self.assertEqual(output.shape, expected_shape)

    def ttest_modelcapacity(self):
        """
        JA: Need to update given parameterisation of model structure.

        Testing different models have same order of magnitude of parameters.
        """
        for seq_len in SEQ_LENS:
            for n_vars in N_VARS:
                for n_classes in N_CLASSES:
                    simple_models = models.MODELS.values()
                    n_params = [sum(p.numel()
                                for p in m(seq_len=seq_len,
                                           n_channels=n_vars,
                                           output_size=n_classes).parameters()
                                if p.requires_grad)
                                for m in simple_models]
                    param_magnitudes = [magnitude(p) for p in n_params]
                    # RNN/LSTM order bigger
                    self.assertTrue(max(param_magnitudes)-min(param_magnitudes)<=1)


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

class TestTrainingMethods(unittest.TestCase):
    """
    Training tests.
    """

    def ttest_hyperparamtune(self):
        """
        """
        raise NotImplementedError

    def ttest_traininggeneric(self):
        """
        """
        raise NotImplementedError

    def ttest_trainingrehearsal(self):
        """
        """
        raise NotImplementedError

    def ttest_trainingregularization(self):
        """
        """
        raise NotImplementedError


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
