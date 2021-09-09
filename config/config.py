"""
Hyperparameter search-space configuration.
"""

from ray import tune
import numpy as np

LOG_WEIGHTS = [1e-3,1e-2,1e-1,1e0,1e1,1e2]
SAMPLE_COUNTS = [32,64]

# Conditional hyper-param functions
def get_dropout_from_n_layers(spec):
    """
    Returns dropout of 0 if n_layers==1
    else random dropout.
    """
    if spec.config.n_layers==1:
        return 0
    else:
        return np.random.choice([0,0.1,0.2,0.4])

# Hyperparameter search-space
config_generic = {
       'lr':tune.choice([1e-4,1e-3,1e-2]),
       'optimizer':tune.choice(['Adam']), #'SGD', #'momentum':tune.choice([0.0, 0.2, 0.4, 0.6, 0.8, 0.9]),
       'train_epochs':15,
       'train_mb_size':tune.choice([16,32,64,128]),
       'hidden_dim':tune.choice([32,64,128]),
       'n_layers':tune.choice([1,2,3]),
       }

config_model = {
       'CNN':{
              'kernel_size':tune.choice([3,5,7]),
              'nonlinearity':tune.choice(['tanh','relu']),
              },
       'MLP':{
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'nonlinearity':tune.choice(['tanh','relu'])
              },
       'Transformer':{
              'n_heads':tune.choice([4,8,16]),
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'nonlinearity':tune.choice(['relu','gelu'])
              },
       'RNN':{
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'nonlinearity':tune.choice(['tanh','relu']),
              'bidirectional':tune.choice([True,False])
              },
       'LSTM':{
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'bidirectional':tune.choice([True,False])
              },
       'GRU':{
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'bidirectional':tune.choice([True,False])
              }
       }

config_cl = {
       'Replay':{
              'mem_size':tune.choice(SAMPLE_COUNTS)
              },
       'GDumb':{
              'mem_size':tune.choice(SAMPLE_COUNTS)
              },
       'EWC':{
              'mode':'separate',
              'ewc_lambda':tune.choice(LOG_WEIGHTS)
              },
       'OnlineEWC':{
              'mode':'online',
              'ewc_lambda':tune.choice(LOG_WEIGHTS),
              'decay_factor':tune.quniform(0,1,0.1)
              },
       'SI':{
              'si_lambda':tune.choice(LOG_WEIGHTS)
              },
       'LwF':{
              'alpha':tune.choice(LOG_WEIGHTS),
              'temperature':tune.quniform(0,3,0.5)
              },
       'GEM':{
              'patterns_per_exp':tune.choice(SAMPLE_COUNTS),
              'memory_strength':tune.quniform(0,1,0.1)
              },
       'AGEM':{
              'patterns_per_exp':tune.choice(SAMPLE_COUNTS),
              'sample_size':tune.choice(SAMPLE_COUNTS)
              }
       #'CoPE':
       }
