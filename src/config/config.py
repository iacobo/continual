"""
Hyperparameter search-space configuration.
"""
#%%
from ray import tune
import numpy as np

N_SAMPLES = [256]
DECAY_WEIGHTS = [0.2,0.4,0.6,0.8,0.9,1]
TEMPERATURES = [0.5,1.0,1.5,2.0,2.5,3.0]
LOG_WEIGHTS = [1e-3,1e-2,1e-1,1e0,1e1,1e2]
HIDDEN_DIMS = [64,128,256]
N_LAYERS = [3,4]
N_HEADS = [12,16,24]

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
       'momentum':0.9, #tune.grid_search(DECAY_WEIGHTS),
       'train_epochs':100,
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
              'mem_size':tune.grid_search([5*n for n in N_SAMPLES])
              # JA: Have edited this in replay definition
              #'storage_policy':storage_policy.ClassBalancedStoragePolicy()
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
              'temperature':tune.grid_search(TEMPERATURES)
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
              'sample_size':tune.grid_search([i*max(N_SAMPLES) for i in range(1,3)])
              }
       #'CoPE':
       }
#%%
import pandas as pd

vals = {'mem_size': N_SAMPLES,
'ewc_lambda': LOG_WEIGHTS,
'decay_factor': DECAY_WEIGHTS,
'si_lambda': LOG_WEIGHTS,
'alpha': LOG_WEIGHTS,
'temperature': TEMPERATURES,
'lambda_e': LOG_WEIGHTS,
'patterns_per_exp': N_SAMPLES,
'memory_strength': DECAY_WEIGHTS,
'sample_size': [i*max(N_SAMPLES) for i in range(1,3)]}

vals2 = {'hidden_dim': HIDDEN_DIMS, 
'n_layers': N_LAYERS,
'nonlinearity': ['tanh','relu', 'gelu*'],
'n_heads': N_HEADS,
'bidirectional': ['True','False']}

models = ['MLP', 'CNN', 'LSTM', 'Transformer']
for k in vals2.keys():
       vals2[k] = (vals2[k], *[k in config_model[model] for model in models])

df_hp = pd.DataFrame(vals.items(), columns=['Hyperparameter', 'Values'])
df_hp = df_hp.set_index(['Hyperparameter'])

#print(df_hp.to_latex())

df_hp = pd.DataFrame(((k,*v) for k,v in vals2.items()), columns=['Hyperparameter', 'Values', *models])
df_hp = df_hp.set_index(['Hyperparameter'])

#print(df_hp.to_latex())


# %%
