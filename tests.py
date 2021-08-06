from utils import models, data_processing

import math
import torch
import unittest
import itertools
from pathlib import Path

def magnitude(value):
    if (value == 0): return 0
    return int(math.floor(math.log10(abs(value))))

# Model definition tests
class TestModelMethods(unittest.TestCase):

    def test_modeloutputshape(self):
        """
        Testing model produces correct shape of output for variety of input sizes.
        """
        for batch_size in (1,10,100):
            for seq_len in (5,10,15,30):
                for n_vars in (2,10,30):
                    for n_classes in (2,4):
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
        for seq_len in (5,10,15,30):
            for n_vars in (2,10,30):
                for n_classes in (2,4):
                    simple_models = models.MODELS.values()
                    n_params = [sum(p.numel() for p in m(seq_len=seq_len, n_channels=n_vars, output_size=n_classes).parameters() if p.requires_grad) for m in simple_models]
                    param_magnitudes = [magnitude(p) for p in n_params]
                    # RNN/LSTM order bigger
                    self.assertTrue(max(param_magnitudes)-min(param_magnitudes)<=1)



# Data loading tests: TO SHELVE AFTER FIDDLE INCORPORATED
class TestDataLoadingMethods(unittest.TestCase):

    def ttest_ids(self):
        """
        Checking patient ID's are unique.
        """
        data_dir = Path(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\data\eICU')
        df = data_processing.load_eicu(root=data_dir)
        self.assertEqual(len(df), len(df.groupby('patientunitstayid')))
        
    def ttest_tasktargets(self):
        """
        Check all task splits have examples of both classes.
        """
        data_dir = Path(r'C:\Users\jacob\OneDrive\Documents\code\cl code\ehr\data\eICU')
        demographics = ['age', 'gender', 'ethnicity', 'region', 'hospitaldischargeyear']
        for demo in demographics:
            tasks = data_processing.eicu_to_tensor(demo, root=data_dir)
            for i, task in enumerate(tasks):
                labels = torch.LongTensor([0,1])
                data_labels = task[1].unique()
                self.assertTrue(data_labels.equal(labels), f"Demographic {demo}, task {i} has unexpected labels: {data_labels}")

    def test_interpolation(self):
        """
        Test that interpolation of missing values works.
        """

        sim_data = torch.randn(100,10,30)
        drop = torch.nn.Dropout(p=0.1)
        sim_data = drop(sim_data)
        sim_data[sim_data==0] = float('nan')

        # NEED TO IMPLEMENT
        self.assertTrue(True)

    def test_modalfeatvalfromseq(self):
        """
        Test that mode of correct dim is returned.
        """
        for n in [1,50,100]:
            for seq_len in [1,5,10,50,100]:
                for n_feats in [1,2,5,10,50,100]:
                    for i in range(n_feats):
                        x = torch.randint(0,1,(n,seq_len,n_feats))
                        modes = data_processing.get_modes(x,feat=i)
                        self.assertEqual(modes.shape, torch.Size([n]))

# CL task split tests
class TestCLConstructionMethods(unittest.TestCase):

    def ttest_taskidsnonoverlap(self):
        for dataset in ['MIMIC','eICU']:
            for experiment in ['ARF','shock','mortality']:
                for demographic in ['age', 'gender', 'ethnicity', 'region', 'time_year', 'time_season', 'time_month']:
                    tasks = data_processing(dataset, demographic, experiment)
                    for pair in itertools.combinations(tasks, repeat=2):
                        self.assertTrue(pair[0][:,0].intersection(pair[0][:,0]) == {})

    def ttest_tasktargets(self):
        for dataset in ['MIMIC','eICU']:
            for experiment in ['ARF','shock','mortality']:
                for demographic in ['age', 'gender', 'ethnicity', 'region', 'time_year', 'time_season', 'time_month']:
                    tasks = data_processing(dataset, demographic, experiment)
                    for task in tasks:
                        self.assertTrue(len(task[:,-1].unique())==2)

if __name__ == '__main__':
    unittest.main()
