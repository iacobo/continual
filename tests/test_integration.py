"""
Test suite.
"""

import unittest

BATCH_SIZES = (1,10,100)
SEQ_LENS = (4,12,48)
N_VARS = (2,10,100)
N_CLASSES = (2,10)
N_LAYERS = (1,2,3,4)
HIDDEN_SIZES = (32,64,128)


DEMOGRAPHICS = ['age', 'gender', 'ethnicity', 'region', 'time_year', 'time_season', 'time_month']
OUTCOMES = ['ARF', 'shock', 'mortality']
DATASETS = ['MIMIC', 'eICU']


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

if __name__ == '__main__':
    unittest.main()
