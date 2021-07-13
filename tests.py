#%%
import data_processing
import models
import torch

# Data loading tests
def test1():
    df = data_processing.load_eicu()
    assert len(df) == len(df.groupby('patientunitstayid')), "Duplicate `patientunitstayid` in dataset!"
    
def test2():
    demographics = ['age', 'gender', 'ethnicity', 'region', 'hospitaldischargeyear']
    for demo in demographics:
        tasks = data_processing.eicu_to_tensor(demo)
        for i, task in enumerate(tasks):
            assert len(task['Target'].value_counts() == 2), f"Demographic {demo}, task {i} has only one outcome label."

# Model definition tests
def test3():
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
                        assert output.shape == (batch_size, n_classes)

# %%
