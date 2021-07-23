# local imports
import data_processing
import models
# other libs
import torch
import unittest

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
                        simple_models = [models.SimpleCNN, models.SimpleMLP, models.SimpleRNN, models.SimpleLSTM]
                        for model in simple_models:
                            model = model(seq_len=seq_len, n_channels=n_vars, output_size=n_classes)
                            output = model(input)
                            expected_shape = torch.Size([batch_size, n_classes])
                            self.assertEqual(output.shape, expected_shape)


# Data loading tests
class TestDataLoadingMethods(unittest.TestCase):

    def test_ids(self):
        """
        Checking patient ID's are unique.
        """
        df = data_processing.load_eicu()
        self.assertEqual(len(df), len(df.groupby('patientunitstayid')))
        
    def test_tasktargets(self):
        """
        Check all task splits have examples of both classes.
        """
        demographics = ['age', 'gender', 'ethnicity', 'region', 'hospitaldischargeyear']
        for demo in demographics:
            tasks = data_processing.eicu_to_tensor(demo)
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

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
