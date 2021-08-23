import math
import unittest
import itertools
import torch

from utils import models, data_processing

BATCH_SIZES = (1,10,100)
SEQ_LENS = (4,10,30,50,100)
N_VARS = (2,10,30,100)
N_CLASSES = (2,3,4,10)

def magnitude(value):
    if value == 0:
        return 0
    else:
        return int(math.floor(math.log10(abs(value))))

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
                        input = torch.randn(batch_size, seq_len, n_vars)
                        simple_models = models.MODELS.values()
                        for model in simple_models:
                            model = model(seq_len=seq_len, n_channels=n_vars, output_size=n_classes)
                            output = model(input)
                            expected_shape = torch.Size([batch_size, n_classes])
                            self.assertEqual(output.shape, expected_shape)

    def test_modelcapacity(self):
        """
        Testing different models have same order of magnitude of parameters.
        """
        for seq_len in SEQ_LENS:
            for n_vars in N_VARS:
                for n_classes in N_CLASSES:
                    simple_models = models.MODELS.values()
                    n_params = [sum(p.numel()
                                for p in m(seq_len=seq_len, n_channels=n_vars, output_size=n_classes).parameters()
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
        for n in BATCH_SIZES:
            for seq_len in SEQ_LENS:
                for n_feats in N_VARS:
                    for i in range(n_feats):
                        sim_data = torch.randint(0,1,(n,seq_len,n_feats)).clone().detach().numpy()
                        modes = data_processing.get_modes(sim_data,feat=i)
                        self.assertEqual(modes.shape, torch.Size([n]))

# CL task split tests
class TestCLConstructionMethods(unittest.TestCase):

    def ttest_taskidsnonoverlap(self):
        for dataset in ['MIMIC','eICU']:
            for experiment in ['ARF','shock','mortality']:
                for demographic in ['age','gender','ethnicity','region','time_year','time_season','time_month']:
                    tasks = data_processing(dataset, demographic, experiment)
                    for pair in itertools.combinations(tasks, repeat=2):
                        self.assertTrue(pair[0][:,0].intersection(pair[0][:,0]) == {})

    def ttest_tasktargets(self):
        for dataset in ['MIMIC','eICU']:
            for experiment in ['ARF','shock','mortality']:
                for demographic in ['age','gender','ethnicity','region','time_year','time_season','time_month']:
                    tasks = data_processing(dataset, demographic, experiment)
                    for task in tasks:
                        self.assertTrue(len(task[:,-1].unique())==2)

if __name__ == '__main__':
    unittest.main()
