"""
Hyperparameter search-space configuration.
"""

from ray import tune
import numpy as np

N_SAMPLES = [128] #[32,64,128]
DECAY_WEIGHTS = [0.2,0.4,0.6,0.8,0.9]
LOG_WEIGHTS = [1e-3,1e-2,1e-1,1e0,1e1,1e2]
HIDDEN_DIMS = [32,64,128]
N_LAYERS = [2,3,4]
N_HEADS = [4,8,16]

# Conditional hyper-param functions
def get_dropout_from_n_layers(spec):
    """
    Returns dropout of 0 if n_layers==1
    else random dropout.
    """
    if spec.config.model.n_layers==1:
        return 0
    else:
        return np.random.choice([0,0.1,0.2,0.4])

# Hyperparameter search-space
config_generic = {
       'lr':tune.grid_search([1e-4,1e-3,1e-2]),
       'optimizer':'SGD', #tune.choice(['Adam','SGD']),
       'momentum':tune.grid_search(DECAY_WEIGHTS),
       'train_epochs':5,
       'train_mb_size':tune.grid_search([16,32,64,128]),
       }

config_model = {
       'CNN':{
              'hidden_dim':tune.grid_search(HIDDEN_DIMS),
              'n_layers':tune.grid_search(N_LAYERS),
              'kernel_size':tune.grid_search([3,5,7]),
              'nonlinearity':tune.grid_search(['tanh','relu']),
              },
       'MLP':{
              'hidden_dim':tune.grid_search(HIDDEN_DIMS),
              'n_layers':tune.grid_search(N_LAYERS),
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'nonlinearity':tune.grid_search(['tanh','relu'])
              },
       'Transformer':{
              'hidden_dim':tune.grid_search(HIDDEN_DIMS),
              'n_layers':tune.grid_search(N_LAYERS),
              'n_heads':tune.grid_search(N_HEADS),
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'nonlinearity':tune.grid_search(['relu','gelu'])
              },
       'RNN':{
              'hidden_dim':tune.grid_search(HIDDEN_DIMS),
              'n_layers':tune.grid_search(N_LAYERS),
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'nonlinearity':tune.grid_search(['tanh','relu']),
              'bidirectional':tune.grid_search([True,False])
              },
       'LSTM':{
              'hidden_dim':tune.grid_search(HIDDEN_DIMS),
              'n_layers':tune.grid_search(N_LAYERS),
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'bidirectional':tune.grid_search([True,False])
              },
       'GRU':{
              'hidden_dim':tune.grid_search(HIDDEN_DIMS),
              'n_layers':tune.grid_search(N_LAYERS),
              'dropout':tune.sample_from(get_dropout_from_n_layers),
              'bidirectional':tune.grid_search([True,False])
              }
       }

config_cl = {
       'Replay':{
              'mem_size':tune.grid_search(N_SAMPLES)
              },
       'GDumb':{
              'mem_size':tune.grid_search(N_SAMPLES)
              },
       'EWC':{
              'mode':'separate',
              'ewc_lambda':tune.grid_search(LOG_WEIGHTS)
              },
       'OnlineEWC':{
              'mode':'online',
              'ewc_lambda':tune.grid_search(LOG_WEIGHTS),
              'decay_factor':tune.grid_search(DECAY_WEIGHTS)
              },
       'SI':{
              'si_lambda':tune.grid_search(LOG_WEIGHTS)
              },
       'LwF':{
              'alpha':tune.grid_search(LOG_WEIGHTS),
              'temperature':tune.grid_search([0.5,1.0,1.5,2.0,2.5,3.0])
              },
       'LFL':{
              'lambda_e':tune.grid_search([LOG_WEIGHTS])
              },
       'GEM':{
              'patterns_per_exp':tune.grid_search(N_SAMPLES),
              'memory_strength':tune.grid_search(DECAY_WEIGHTS)
              },
       'AGEM':{
              'patterns_per_exp':tune.grid_search(N_SAMPLES),
              'sample_size':tune.grid_search(N_SAMPLES)
              }
       #'CoPE':
       }
